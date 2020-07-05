import unittest
import pyomo.environ as pe
from pyomo.opt import assert_optimal_termination
import coramin


class TestBaseRelaxation(unittest.TestCase):
    def test_push_and_pop_oa_points(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.y)
        m.obj = pe.Objective(expr=m.y)

        opt = pe.SolverFactory('cplex_persistent')
        opt.set_instance(m)
        m.rel.add_persistent_solver(opt)

        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, -0.5)
        self.assertAlmostEqual(m.y.value, -2)

        m.x.value = -1
        m.rel.add_cut(keep_cut=True)
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

        m.rel.push_oa_points()
        m.rel.rebuild()
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

        m.rel.clear_oa_points()
        m.rel.rebuild()
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, -0.5)
        self.assertAlmostEqual(m.y.value, -2)

        m.x.value = -0.5
        m.rel.add_cut(keep_cut=True)
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0.25)
        self.assertAlmostEqual(m.y.value, -0.5)

        m.rel.pop_oa_points()
        m.rel.rebuild()
        res = opt.solve(save_results=False)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

    def test_push_and_pop_partitions(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.y)
        m.obj = pe.Objective(expr=m.y)
        self.assertEqual(m.rel._partitions[m.x], [-2, 1])

        m.rel.add_partition_point(-1)
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])

        m.rel.push_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])

        m.rel.clear_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, 1])

        m.rel.add_partition_point(-0.5)
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -0.5, 1])

        m.rel.pop_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])
