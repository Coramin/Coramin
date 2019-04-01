import unittest
import math
import pyomo.environ as pe
import coramin


class TestUnivariateExp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = pe.ConcreteModel()
        cls.model = model
        model.y = pe.Var()
        model.x = pe.Var(bounds=(-1.5, 1.5))

        model.obj = pe.Objective(expr=model.y, sense=pe.maximize)
        model.pw_exp = coramin.relaxations.PWUnivariateRelaxation()
        model.pw_exp.build(x=model.x, w=model.y, pw_repn='INC', shape=coramin.utils.FunctionShape.CONVEX,
                           relaxation_side=coramin.utils.RelaxationSide.BOTH, f_x_expr=pe.exp(model.x))
        model.pw_exp.add_point(-0.5)
        model.pw_exp.add_point(0.5)
        model.pw_exp.rebuild()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_exp_ub(self):
        model = self.model.clone()

        solver = pe.SolverFactory('glpk')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), math.exp(1.5), 4)

    def test_exp_mid(self):
        model = self.model.clone()
        model.x_con = pe.Constraint(expr=model.x <= 0.3)

        solver = pe.SolverFactory('glpk')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), 1.44, 3)

    def test_exp_lb(self):
        model = self.model.clone()
        model.obj.sense = pe.minimize

        solver = pe.SolverFactory('glpk')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), math.exp(-1.5), 4)


class TestFeasibility(unittest.TestCase):
    def test_univariate_exp(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('glpk')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = math.exp(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_pw_exp(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        m.c.add_point(-0.25)
        m.c.add_point(0.25)
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('glpk')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = math.exp(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_univariate_log(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=pe.log(m.x))
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('glpk')
        for xval in [0.5, 0.75, 1, 1.25, 1.5]:
            pval = math.log(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_pw_log(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=pe.log(m.x))
        m.c.add_point(0.9)
        m.c.add_point(1.1)
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('glpk')
        for xval in [0.5, 0.75, 1, 1.25, 1.5]:
            pval = math.log(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_x_fixed(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.x.fix(0)
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        self.assertEqual(id(m.c.x_fixed_con.body), id(m.y))
        self.assertEqual(m.c.x_fixed_con.lower, 1.0)
        self.assertEqual(m.c.x_fixed_con.upper, 1.0)

    def test_x_sq(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWXSquaredRelaxation()
        m.c.build(x=m.x, w=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH)
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('glpk')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = xval**2
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)
