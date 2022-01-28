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
    return rhs_vals


def _get_relaxation_vals(rhs_vars: Sequence[_GeneralVarData],
                         rhs_expr: ExpressionBase,
                         m: _BlockData,
                         rel: coramin.relaxations.BaseRelaxationData,
                         eval_pts: List[Tuple[float, ...]],
                         rel_side: coramin.utils.RelaxationSide) -> List[float]:
    opt = appsi.solvers.Gurobi()
    opt.update_config.update_vars = True
    opt.update_config.check_for_new_or_removed_vars = False
    opt.update_config.check_for_new_or_removed_constraints = False
    opt.update_config.check_for_new_or_removed_params = False
    opt.update_config.update_constraints = False
    opt.update_config.update_params = False
    opt.update_config.update_named_expressions = False

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
    return under_est_vals


class TestRelaxationBasics(unittest.TestCase):
    def valid_relaxation_helper(self,
                                m: _BlockData,
                                rel: coramin.relaxations.BaseRelaxationData,
                                rhs_expr: ExpressionBase):
        rhs_vars = rel.get_rhs_vars()
        sample_points = _grid_rhs_vars(rhs_vars)
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

    def get_base_pyomo_model(self, xlb=-1.5, xub=0.8, ylb=-2, yub=1):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(xlb, xub))
        m.y = pe.Var(bounds=(ylb, yub))
        m.z = pe.Var()
        return m

    def test_valid_quadratic_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        self.valid_relaxation_helper(m, m.rel, m.x**2)

    def test_valid_exp_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        self.valid_relaxation_helper(m, m.rel, pe.exp(m.x))

    def test_valid_log_relaxation(self):
        m = self.get_base_pyomo_model(xlb=0.1, xub=2.5)
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=pe.log(m.x))
        self.valid_relaxation_helper(m, m.rel, pe.log(m.x))

    def test_valid_convex_relaxation(self):
        m = self.get_base_pyomo_model(xlb=0.1, xub=2.5)
        m.rel = coramin.relaxations.PWUnivariateRelaxation()
        m.rel.build(x=m.x, aux_var=m.z, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=m.x * pe.log(m.x))
        self.valid_relaxation_helper(m, m.rel, m.x * pe.log(m.x))

    def test_valid_cos_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWCosRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        self.valid_relaxation_helper(m, m.rel, pe.cos(m.x))

    def test_valid_sin_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWSinRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        self.valid_relaxation_helper(m, m.rel, pe.sin(m.x))

    def test_valid_atan_relaxation(self):
        m = self.get_base_pyomo_model()
        m.rel = coramin.relaxations.PWArctanRelaxation()
        m.rel.build(x=m.x, aux_var=m.z)
        self.valid_relaxation_helper(m, m.rel, pe.atan(m.x))
