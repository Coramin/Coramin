import pyomo.environ as pyo
import unittest
import coramin


class TestMcCormick(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_mccormick1(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 6))
        model.y = pyo.Var(bounds=(0, 3))
        model.w = pyo.Var()

        model.obj = pyo.Objective(expr=-model.w - 2 * model.x)
        model.con = pyo.Constraint(expr=model.w <= 12)
        model.mc = coramin.relaxations.McCormickRelaxation()
        model.mc.build(x=model.x, y=model.y, w=model.w)

        linsolver = pyo.SolverFactory('glpk')
        linsolver.solve(model)
        self.assertAlmostEqual(pyo.value(model.x), 6.0, 6)
        self.assertAlmostEqual(pyo.value(model.y), 2.0, 6)

    def test_mccormick2(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 6))
        model.y = pyo.Var(bounds=(0, 3))
        model.w = pyo.Var()

        model.obj = pyo.Objective(expr=-model.w - 2 * model.x)
        model.con = pyo.Constraint(expr=model.w <= 12)
        def mc_rule(b):
            b.build(x=model.x, y=model.y, w=model.w)
        model.mc = coramin.relaxations.McCormickRelaxation(rule=mc_rule)

        linsolver = pyo.SolverFactory('glpk')
        linsolver.solve(model)
        self.assertAlmostEqual(pyo.value(model.x), 6.0, 6)
        self.assertAlmostEqual(pyo.value(model.y), 2.0, 6)

    def test_mccormick3_BOTH(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 6))
        model.y = pyo.Var(bounds=(0, 3))
        model.w = pyo.Var()

        model.obj = pyo.Objective(expr=-model.w - 2 * model.x)
        model.con = pyo.Constraint(expr=model.w <= 12)

        def mc_rule(b):
            m = b.parent_block()
            b.build(x=m.x, y=m.y, w=m.w)
        model.mc = coramin.relaxations.McCormickRelaxation(rule=mc_rule)

        linsolver = pyo.SolverFactory('glpk', tee=True)
        linsolver.solve(model)
        self.assertAlmostEqual(pyo.value(model.x), 6.0, 6)
        self.assertAlmostEqual(pyo.value(model.y), 2.0, 6)

    def test_mccormick3_OVER(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 6))
        model.y = pyo.Var(bounds=(0, 3))
        model.w = pyo.Var()

        model.obj = pyo.Objective(expr=-model.w - 2 * model.x)
        model.con = pyo.Constraint(expr=model.w <= 12)

        def mc_rule(b):
            m = b.parent_block()
            b.build(x=m.x, y=m.y, w=m.w, relaxation_side=coramin.utils.RelaxationSide.OVER)
        model.mc = coramin.relaxations.McCormickRelaxation(rule=mc_rule)

        linsolver = pyo.SolverFactory('glpk', tee=True)
        linsolver.solve(model)
        self.assertAlmostEqual(pyo.value(model.x), 6.0, 6)
        self.assertAlmostEqual(pyo.value(model.y), 2.0, 6)

    def test_mccormick3_UNDER(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 6))
        model.y = pyo.Var(bounds=(0, 3))
        model.w = pyo.Var()

        model.obj = pyo.Objective(expr=-model.w - 2 * model.x)
        model.con = pyo.Constraint(expr=model.w <= 12)

        def mc_rule(b):
            m = b.parent_block()
            b.build(x=m.x, y=m.y, w=m.w, relaxation_side=coramin.utils.RelaxationSide.UNDER)
        model.mc = coramin.relaxations.McCormickRelaxation(rule=mc_rule)

        linsolver = pyo.SolverFactory('glpk', tee=True)
        linsolver.solve(model)
        self.assertAlmostEqual(pyo.value(model.w), 12.0, 6)
