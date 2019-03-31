import unittest
import math
import pyomo.environ as pe
import coramin


class TestUnivariatePWEnvelopes(unittest.TestCase):
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

