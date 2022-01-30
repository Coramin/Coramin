import unittest
import pyomo.environ as pe
import coramin
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExpressionBase
from typing import Sequence, List, Tuple
import numpy as np
import itertools
from pyomo.contrib import appsi
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.util.report_scaling import _check_coefficents
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr import sympy_tools
import io


def _grid_rhs_vars(v_list: Sequence[_GeneralVarData], num_points: int = 30) -> List[Tuple[float, ...]]:
    res = list()
    for v in v_list:
        res.append(np.linspace(v.lb, v.ub, num_points))
    res = list(tuple(float(p) for p in i) for i in itertools.product(*res))
    return res


def _get_rhs_vals(rhs_vars: Sequence[_GeneralVarData],
                  rhs_expr: ExpressionBase,
                  eval_pts: List[Tuple[float, ...]]) -> List[float]:
    rhs_vals = list()
    for pt in eval_pts:
        for v, p in zip(rhs_vars, pt):
            v.fix(p)
        rhs_vals.append(pe.value(rhs_expr))
    for v in rhs_vars:
        v.unfix()
    return rhs_vals


def _get_relaxation_vals(rhs_vars: Sequence[_GeneralVarData],
                         rhs_expr: ExpressionBase,
                         m: _BlockData,
                         rel: coramin.relaxations.BaseRelaxationData,
                         eval_pts: List[Tuple[float, ...]],
                         rel_side: coramin.utils.RelaxationSide,
                         linear: bool = True) -> List[float]:
    opt = appsi.solvers.Gurobi()
    opt.update_config.update_vars = True
    opt.update_config.check_for_new_or_removed_vars = False
    opt.update_config.check_for_new_or_removed_constraints = False
    opt.update_config.check_for_new_or_removed_params = False
    opt.update_config.update_constraints = False
    opt.update_config.update_params = False
    opt.update_config.update_named_expressions = False
    if linear:
        opt.update_config.treat_fixed_vars_as_params = False

    if rel_side == coramin.utils.RelaxationSide.UNDER:
        sense = pe.minimize
    else:
        sense = pe.maximize
    m.obj = pe.Objective(expr=rel.get_aux_var(), sense=sense)

    under_est_vals = list()
    for pt in eval_pts:
        for v, p in zip(rhs_vars, pt):
            v.fix(p)
        res = opt.solve(m)
        assert res.termination_condition == appsi.base.TerminationCondition.optimal
        under_est_vals.append(rel.get_aux_var().value)

    del m.obj
    for v in rhs_vars:
        v.unfix()
    return under_est_vals


def _num_cons(rel):
    cons = list(rel.component_data_objects(pe.Constraint, descend_into=True, active=True))
    return len(cons)


def _check_unbounded(m: _BlockData,
                     rel: coramin.relaxations.BaseRelaxationData,
                     rel_side: coramin.utils.RelaxationSide,
                     linear: bool = True):
    if rel_side == coramin.utils.RelaxationSide.UNDER:
        sense = pe.minimize
    else:
        sense = pe.maximize
    m.obj = pe.Objective(expr=rel.get_aux_var(), sense=sense)

    if linear:
        opt = appsi.solvers.Gurobi()
        opt.gurobi_options['DualReductions'] = 0
    else:
        opt = appsi.solvers.Ipopt()
    opt.config.load_solution = False
    res = opt.solve(m)

    del m.obj

    return res.termination_condition == appsi.base.TerminationCondition.unbounded


def _check_linear(m: _BlockData):
    for c in m.component_data_objects(pe.Constraint, descend_into=True, active=True):
        repn = generate_standard_repn(c.body)
        if not repn.is_linear():
            return False
    return True


def _check_linear_or_convex(rel: coramin.relaxations.BaseRelaxationData):
    for c in rel.component_data_objects(pe.Constraint, descend_into=True, active=True):
        repn = generate_standard_repn(c.body, quadratic=False)
        if repn.is_linear():
            continue

        if c.lower is not None and c.upper is not None:
            return False  # nonlinear equality constraints are not convex

        # reconstruct the expression without the aux_var
        e = repn.constant
        for coef, v in zip(repn.linear_coefs, repn.linear_vars):
            if v is not rel.get_aux_var():
                e += coef*v
        e += repn.nonlinear_expr

        e_vars = list(identify_variables(e))
        assert len(e_vars) == 1
        rhs_var = rel.get_rhs_vars()[0]
        assert e_vars[0] is rhs_var

        der = differentiate(e, wrt=rhs_var, mode=Modes.reverse_symbolic)
        hes = differentiate(der, wrt=rhs_var, mode=Modes.reverse_symbolic)
        if type(hes) not in {int, float}:
            om, se = sympy_tools.sympyify_expression(hes)
            se = se.simplify()
            hes = sympy_tools.sympy2pyomo_expression(se, om)
        hes_lb, hes_ub = compute_bounds_on_expr(hes)

        if c.lower is not None:
            # e should be concave
            if hes_ub > 0:
                return False
        else:
            assert c.upper is not None
            # e should be convex
            if hes_lb < 0:
                return False
    return True


def _check_scaling(m: _BlockData, rel: coramin.relaxations.BaseRelaxationData) -> bool:
    cons_with_large_coefs = dict()
    cons_with_small_coefs = dict()
    for c in m.component_data_objects(pe.Constraint, descend_into=True, active=True):
        _check_coefficents(c, c.body, rel.large_coef, rel.small_coef, cons_with_large_coefs, cons_with_small_coefs)
    passed = len(cons_with_large_coefs) == 0 and len(cons_with_small_coefs) == 0
    return passed


class TestRelaxationBasics(unittest.TestCase):
    def valid_relaxation_helper(self,
                                m: _BlockData,
                                rel: coramin.relaxations.BaseRelaxationData,
                                rhs_expr: ExpressionBase,
                                num_points: int = 30):
        if rel.use_linear_relaxation:
            self.assertTrue(_check_linear(m))
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars, num_points=num_points)
        rhs_vals = _get_rhs_vals(rhs_vars, rhs_expr, sample_points)
        under_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                              coramin.utils.RelaxationSide.UNDER)
        over_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                             coramin.utils.RelaxationSide.OVER)
        rhs_vals = np.array(rhs_vals)
        under_est_vals = np.array(under_est_vals)
        over_est_vals = np.array(over_est_vals)

        self.assertTrue(np.all(rhs_vals >= under_est_vals))
        self.assertTrue(np.all(rhs_vals <= over_est_vals))

    def equal_at_points_helper(self,
                               m: _BlockData,
                               rel: coramin.relaxations.BaseRelaxationData,
                               rhs_expr: ExpressionBase,
                               pts: Sequence[Tuple[float, ...]],
                               check_underestimator: bool = True,
                               check_overestimator: bool = True,
                               linear: bool = True):
        rhs_vars = rel.get_rhs_vars()
        rhs_vals = _get_rhs_vals(rhs_vars, rhs_expr, pts)
        rhs_vals = np.array(rhs_vals)
        if check_underestimator:
            under_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, pts,
                                                  coramin.utils.RelaxationSide.UNDER, linear)
            under_est_vals = np.array(under_est_vals)
            self.assertTrue(np.all(np.isclose(rhs_vals, under_est_vals)))
        if check_overestimator:
            over_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, pts,
                                                 coramin.utils.RelaxationSide.OVER, linear)
            over_est_vals = np.array(over_est_vals)
            self.assertTrue(np.all(np.isclose(rhs_vals, over_est_vals)))

    def nonlinear_relaxation_helper(self,
                                    m: _BlockData,
                                    rel: coramin.relaxations.BaseRelaxationData,
                                    rhs_expr: ExpressionBase,
                                    num_points: int = 30):
        rel.use_linear_relaxation = False
        rel.rebuild()
        if rel.is_rhs_convex() or rel.is_rhs_concave():
            self.assertFalse(_check_linear(m))
            self.assertTrue(_check_linear_or_convex(rel))
        else:
            self.assertTrue(_check_linear(m))
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars, num_points=num_points)
        rhs_vals = _get_rhs_vals(rhs_vars, rhs_expr, sample_points)
        under_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                              coramin.utils.RelaxationSide.UNDER, linear=False)
        over_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                             coramin.utils.RelaxationSide.OVER, linear=False)
        rhs_vals = np.array(rhs_vals)
        under_est_vals = np.array(under_est_vals)
        over_est_vals = np.array(over_est_vals)

        if rel.is_rhs_convex():
            self.assertTrue(np.all(np.isclose(rhs_vals, under_est_vals)))
        else:
            self.assertTrue(np.all(rhs_vals >= under_est_vals))
        if rel.is_rhs_concave():
            self.assertTrue(np.all(np.isclose(rhs_vals, over_est_vals)))
        else:
            self.assertTrue(np.all(rhs_vals <= over_est_vals))

        if rel.is_rhs_convex():
            rel.relaxation_side = coramin.utils.RelaxationSide.OVER
        if rel.is_rhs_concave():
            rel.relaxation_side = coramin.utils.RelaxationSide.UNDER
        rel.rebuild()
        self.assertTrue(_check_linear(m))

        rel.relaxation_side = coramin.utils.RelaxationSide.BOTH
        rel.use_linear_relaxation = True
        rel.rebuild()

    def original_constraint_helper(self,
                                   m: _BlockData,
                                   rel: coramin.relaxations.BaseRelaxationData,
                                   rhs_expr: ExpressionBase):
        rel.rebuild(build_nonlinear_constraint=True)
        self.assertFalse(_check_linear(m))
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars, 15)
        rhs_vals = _get_rhs_vals(rhs_vars, rhs_expr, sample_points)
        under_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                              coramin.utils.RelaxationSide.UNDER, linear=False)
        over_est_vals = _get_relaxation_vals(rhs_vars, rhs_expr, m, rel, sample_points,
                                             coramin.utils.RelaxationSide.OVER, linear=False)

        rhs_vals = np.array(rhs_vals)
        under_est_vals = np.array(under_est_vals)
        over_est_vals = np.array(over_est_vals)

        self.assertTrue(np.all(np.isclose(rhs_vals, under_est_vals)))
        self.assertTrue(np.all(np.isclose(rhs_vals, over_est_vals)))

        rel.rebuild()
        self.valid_relaxation_helper(m, rel, rhs_expr)

    def relaxation_side_helper(self,
                               m: _BlockData,
                               rel: coramin.relaxations.BaseRelaxationData,
                               rhs_expr: ExpressionBase,
                               check_nonlinear_relaxation: bool = True):
        rel.relaxation_side = coramin.utils.RelaxationSide.UNDER
        rel.rebuild()
        sample_points = [tuple(v.lb for v in rel.get_rhs_vars()), tuple(v.ub for v in rel.get_rhs_vars())]
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, True, False)
        self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.OVER))

        rel.relaxation_side = coramin.utils.RelaxationSide.OVER
        rel.rebuild()
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, False, True)
        self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.UNDER))

        if check_nonlinear_relaxation:
            rel.use_linear_relaxation = False

            rel.relaxation_side = coramin.utils.RelaxationSide.UNDER
            rel.rebuild()
            sample_points = [(v.lb, v.ub) for v in rel.get_rhs_vars()]
            self.equal_at_points_helper(m, rel, rhs_expr, sample_points, True, False, False)
            self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.OVER, False))

            rel.relaxation_side = coramin.utils.RelaxationSide.OVER
            rel.rebuild()
            self.equal_at_points_helper(m, rel, rhs_expr, sample_points, False, True, False)
            self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.UNDER, False))

        rel.relaxation_side = coramin.utils.RelaxationSide.UNDER
        rel.rebuild(build_nonlinear_constraint=True)
        sample_points = [(v.lb, v.ub) for v in rel.get_rhs_vars()]
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, True, False, False)
        self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.OVER, False))

        rel.relaxation_side = coramin.utils.RelaxationSide.OVER
        rel.rebuild(build_nonlinear_constraint=True)
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, False, True, False)
        self.assertTrue(_check_unbounded(m, rel, coramin.RelaxationSide.UNDER, False))

        rel.use_linear_relaxation = True
        rel.relaxation_side = coramin.RelaxationSide.BOTH
        rel.rebuild()

    def changing_bounds_helper(self,
                               m: _BlockData,
                               rel: coramin.relaxations.BaseRelaxationData,
                               rhs_expr: ExpressionBase,
                               num_points: int = 10):
        rhs_vars = rel.get_rhs_vars()
        orig_bnds = pe.ComponentMap((v, (v.lb, v.ub)) for v in rhs_vars)
        grid_pts = _grid_rhs_vars(rhs_vars, num_points=num_points)
        for pt in grid_pts:
            for v, p in zip(rhs_vars, pt):
                v.setlb(p)
            rel.rebuild()
            self.assertIn(_num_cons(rel), {2, 3, 4})
            self.valid_relaxation_helper(m, rel, rhs_expr, num_points=num_points)
            self.equal_at_points_helper(m, rel, rhs_expr, [tuple(v.lb for v in rhs_vars), tuple(v.ub for v in rhs_vars)])
            if rel.is_rhs_convex() or rel.is_rhs_concave():
                self.nonlinear_relaxation_helper(m, rel, rhs_expr, num_points)
        for v, (v_lb, v_ub) in orig_bnds.items():
            v.setlb(v_lb)
            v.setub(v_ub)
        rel.rebuild()
        self.assertIn(_num_cons(rel), {2, 3, 4})
        self.valid_relaxation_helper(m, rel, rhs_expr, num_points=num_points)
        self.equal_at_points_helper(m, rel, rhs_expr, [tuple(v.lb for v in rhs_vars), tuple(v.ub for v in rhs_vars)])
        for pt in grid_pts:
            for v, p in zip(rhs_vars, pt):
                v.setub(p)
            rel.rebuild()
            self.assertIn(_num_cons(rel), {2, 3, 4})
            self.valid_relaxation_helper(m, rel, rhs_expr, num_points=num_points)
            self.equal_at_points_helper(m, rel, rhs_expr, [tuple(v.lb for v in rhs_vars), tuple(v.ub for v in rhs_vars)])
            if rel.is_rhs_convex() or rel.is_rhs_concave():
                self.nonlinear_relaxation_helper(m, rel, rhs_expr, num_points)
        for v, (v_lb, v_ub) in orig_bnds.items():
            v.setlb(v_lb)
            v.setub(v_ub)
        rel.rebuild()
        self.assertIn(_num_cons(rel), {2, 3, 4})
        self.valid_relaxation_helper(m, rel, rhs_expr, num_points=num_points)
        self.equal_at_points_helper(m, rel, rhs_expr, [tuple(v.lb for v in rhs_vars), tuple(v.ub for v in rhs_vars)])

    def large_bounds_helper(self, m: _BlockData, rel: coramin.relaxations.BaseRelaxationData, lb=1, ub=1e6):
        orig_bnds = pe.ComponentMap((v, (v.lb, v.ub)) for v in rel.get_rhs_vars())

        for v in rel.get_rhs_vars():
            v.setlb(lb)
            v.setub(ub)
        rel.rebuild()

        scaling_passed = _check_scaling(m, rel)
        self.assertTrue(scaling_passed)

        if rel.is_rhs_convex():
            self.assertTrue(_check_unbounded(m, rel, coramin.utils.RelaxationSide.OVER))
        elif rel.is_rhs_concave():
            self.assertTrue(_check_unbounded(m, rel, coramin.utils.RelaxationSide.UNDER))
        else:
            self.assertTrue(_check_unbounded(m, rel, coramin.utils.RelaxationSide.UNDER))
            self.assertTrue(_check_unbounded(m, rel, coramin.utils.RelaxationSide.OVER))

        for v, (v_lb, v_ub) in orig_bnds.items():
            v.setlb(v_lb)
            v.setub(v_ub)
        rel.rebuild()

    def infinite_bounds_helper(self, m: _BlockData, rel: coramin.relaxations.BaseRelaxationData):
        self.large_bounds_helper(m, rel, None, None)

    def oa_cuts_helper(self,
                       m: _BlockData,
                       rel: coramin.relaxations.BaseRelaxationData,
                       rhs_expr: ExpressionBase):
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars, 5)
        for pt in sample_points:
            rel.add_oa_point(pt)
        rel.rebuild()
        if rel.is_rhs_convex() or rel.is_rhs_concave():
            self.assertEqual(len(rel._cuts), 5)
        self.valid_relaxation_helper(m, rel, rhs_expr)
        if rel.is_rhs_convex():
            check_under = True
        else:
            check_under = False
        if rel.is_rhs_concave():
            check_over = True
        else:
            check_over = False
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, check_under, check_over)
        rel.clear_oa_points()
        rel.rebuild()

    def pw_helper(self,
                  m: _BlockData,
                  rel: coramin.relaxations.BasePWRelaxationData,
                  rhs_expr: ExpressionBase):
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars, 5)
        for pt in sample_points:
            rel.add_oa_point(pt)
            rel.add_partition_point(pt[0])
        rel.rebuild()
        self.valid_relaxation_helper(m, rel, rhs_expr)
        self.equal_at_points_helper(m, rel, rhs_expr, sample_points, True, True)
        rel.clear_oa_points()
        rel.clear_partitions()
        rel.rebuild()

    def util_methods_helper(self,
                            rel: coramin.relaxations.BaseRelaxationData,
                            rhs_expr: ExpressionBase,
                            aux_var: _GeneralVarData,
                            expected_convex: bool,
                            expected_concave: bool):
        # test get_rhs_vars
        expected = ComponentSet(identify_variables(rhs_expr))
        got = ComponentSet(rel.get_rhs_vars())
        diff = expected - got
        self.assertEqual(len(diff), 0)
        diff = got - expected
        self.assertEqual(len(diff), 0)
        self.assertEqual(type(rel.get_rhs_vars()), tuple)

        # test get_rhs_expr
        expected = rhs_expr
        got = rel.get_rhs_expr()
        self.assertTrue(compare_expressions(expected, got))

        # test get_aux_var
        self.assertIs(rel.get_aux_var(), aux_var)

        # test convex/concave
        self.assertEqual(rel.is_rhs_convex(), expected_convex)
        self.assertEqual(rel.is_rhs_concave(), expected_concave)

        # check that pprint does not raise an error
        out = io.StringIO()
        rel.pprint(ostream=out)
        self.assertIn(f'{str(rel.get_aux_var())} == {str(rhs_expr)}', out.getvalue())
        rel.relaxation_side = coramin.RelaxationSide.UNDER
        rel.rebuild()
        out = io.StringIO()
        rel.pprint(ostream=out)
        self.assertIn(f'{str(rel.get_aux_var())} >= {str(rhs_expr)}', out.getvalue())
        rel.relaxation_side = coramin.RelaxationSide.OVER
        rel.rebuild()
        out = io.StringIO()
        rel.pprint(ostream=out)
        self.assertIn(f'{str(rel.get_aux_var())} <= {str(rhs_expr)}', out.getvalue())
        rel.relaxation_side = coramin.RelaxationSide.BOTH
        rel.rebuild()
        rel.pprint(verbose=True)

    def deviation_helper(self,
                         rel: coramin.relaxations.BaseRelaxationData,
                         rhs_expr: ExpressionBase):
        for v in rel.get_rhs_vars():
            v.value = np.random.uniform(v.lb, v.ub)
        rel.get_aux_var().value = pe.value(rhs_expr) + 1
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 1)
        rel.relaxation_side = coramin.RelaxationSide.UNDER
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 0)
        rel.relaxation_side = coramin.RelaxationSide.OVER
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 1)
        rel.relaxation_side = coramin.RelaxationSide.BOTH
        rel.get_aux_var().value = pe.value(rhs_expr) - 1
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 1)
        rel.relaxation_side = coramin.RelaxationSide.UNDER
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 1)
        rel.relaxation_side = coramin.RelaxationSide.OVER
        dev = rel.get_deviation()
        self.assertAlmostEqual(dev, 0)
        rel.relaxation_side = coramin.RelaxationSide.BOTH

    def small_coef_helper(self, m: _BlockData, rel: coramin.relaxations.BaseRelaxationData, rhs_expr: ExpressionBase):
        rel.small_coef = 1e10
        rel.rebuild()
        self.valid_relaxation_helper(m, rel, rhs_expr)
        rel.small_coef = 1e-10
        rel.rebuild()

    def get_base_pyomo_model(self, xlb=-1.5, xub=0.8, ylb=-2, yub=1):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(xlb, xub))
        m.y = pe.Var(bounds=(ylb, yub))
        m.z = pe.Var()
        return m

    def test_quadratic_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        e = m.x**2
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, True, False)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5,), (0.8,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        self.large_bounds_helper(m, m.rel)
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_exp_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        e = pe.exp(m.x)
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=e)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, True, False)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5,), (0.8,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        self.large_bounds_helper(m, m.rel)
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_log_relaxation(self):
        m = self.get_base_pyomo_model(xlb=0.1, xub=2.5)
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        e = pe.log(m.x)
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=e)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, False, True)
        self.equal_at_points_helper(m, m.rel, e, [(0.1,), (2.5,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        m.rel.large_coef = 1e3
        self.large_bounds_helper(m, m.rel, lb=1e-4, ub=1e-3)
        m.rel.large_coef = 1e5
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_univariate_convex_relaxation(self):
        m = self.get_base_pyomo_model(xlb=0.1, xub=2.5)
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        e = m.x * pe.log(m.x)
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=e)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, True, False)
        self.equal_at_points_helper(m, m.rel, e, [(0.1,), (2.5,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        m.rel.large_coef = 0.1
        self.large_bounds_helper(m, m.rel)
        m.rel.large_coef = 1e5
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_cos_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWCosRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        e = pe.cos(m.x)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, False, True)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5,), (0.8,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        self.large_bounds_helper(m, m.rel)
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_sin_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWSinRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        e = pe.sin(m.x)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, False, False)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5,), (0.8,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        self.large_bounds_helper(m, m.rel)
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_atan_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWArctanRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        e = pe.atan(m.x)
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, False, False)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5,), (0.8,)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e)
        self.infinite_bounds_helper(m, m.rel)
        m.rel.large_coef = 0.1
        self.large_bounds_helper(m, m.rel)
        m.rel.large_coef = 1e5
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=True)
        self.deviation_helper(m.rel, e)

    def test_bilinear_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWMcCormickRelaxation()
        m.rel.build(x1=m.x, x2=m.y, aux_var=m.z)
        e = m.x * m.y
        self.valid_relaxation_helper(m, m.rel, e)
        self.util_methods_helper(m.rel, e, m.z, False, False)
        self.equal_at_points_helper(m, m.rel, e, [(-1.5, -2), (0.8, 1), (-1.5, 1), (0.8, -2)])
        self.oa_cuts_helper(m, m.rel, e)
        self.pw_helper(m, m.rel, e)
        self.changing_bounds_helper(m, m.rel, e, num_points=5)
        self.infinite_bounds_helper(m, m.rel)
        self.large_bounds_helper(m, m.rel, lb=-1e6, ub=1e6)
        self.small_coef_helper(m, m.rel, e)
        self.original_constraint_helper(m, m.rel, e)
        with self.assertRaisesRegex(ValueError, "Relaxations of type <class 'coramin.relaxations.custom_block._ScalarPWMcCormickRelaxation'> do not support relaxations that are not linear."):
            self.nonlinear_relaxation_helper(m, m.rel, e)
        self.relaxation_side_helper(m, m.rel, e, check_nonlinear_relaxation=False)
        self.deviation_helper(m.rel, e)
