import pyomo.environ as pe
import coramin
from coramin.algorithms.ecp_bounder import _ECPBounder
import unittest
import logging


logging.basicConfig(level=logging.INFO)


class TestECPBounder(unittest.TestCase):
    def test_ecp_bounder(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=0.5*(m.x**2 + m.y**2))
        m.c1 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        m.c2 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        coramin.relaxations.relax(m, in_place=True)
        opt = _ECPBounder(subproblem_solver='gurobi_direct')
        res = opt.solve(m)
        self.assertEqual(res.solver.termination_condition, pe.TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def test_ecp_bounder_persistent(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=0.5*(m.x**2 + m.y**2))
        m.c1 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        m.c2 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        coramin.relaxations.relax(m, in_place=True)
        opt = coramin.algorithms.ECPBounder(subproblem_solver='gurobi_persistent')
        opt.set_instance(m)
        res = opt.solve()
        self.assertEqual(res.solver.termination_condition, pe.TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)
