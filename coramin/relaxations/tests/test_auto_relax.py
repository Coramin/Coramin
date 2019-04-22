import pyomo.environ as pe
import coramin
import unittest
from pyomo.contrib.derivatives.differentiate import reverse_sd
from pyomo.core.expr.visitor import identify_variables


class TestAutoRelax(unittest.TestCase):
    def test_product1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var(bounds=(-1,1))
        m.z = pe.Var()
        m.c = pe.Constraint(expr=m.z - m.x*m.y == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 1)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, -1)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)
        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWMcCormickRelaxation))
        self.assertIn(id(rel.x), {id(rel.relaxations.rel0._x), id(rel.relaxations.rel0._y)})
        self.assertIn(id(rel.y), {id(rel.relaxations.rel0._x), id(rel.relaxations.rel0._y)})
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))

    def test_product2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var(bounds=(-1,1))
        m.z = pe.Var()
        m.v = pe.Var()
        m.c1 = pe.Constraint(expr=m.z - m.x*m.y == 0)
        m.c2 = pe.Constraint(expr=m.v - 3*m.x*m.y == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, -1)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 2)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.v], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWMcCormickRelaxation))
        self.assertIn(id(rel.x), {id(rel.relaxations.rel0._x), id(rel.relaxations.rel0._y)})
        self.assertIn(id(rel.y), {id(rel.relaxations.rel0._x), id(rel.relaxations.rel0._y)})
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))

    def test_quadratic(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.c = pe.Constraint(expr=m.x**2 + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**2 == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWXSquaredRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_cubic_convex(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1,2))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.c = pe.Constraint(expr=m.x**3 + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**3 == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 1)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 8)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_cubic_concave(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2,-1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.c = pe.Constraint(expr=m.x**3 + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**3 == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, -8)
        self.assertAlmostEqual(rel.aux_vars[1].ub, -1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertFalse(rel.relaxations.rel0.is_convex())
        self.assertTrue(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_cubic(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.c = pe.Constraint(expr=m.x**3 + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**3 == 0)

        rel = coramin.relaxations.relax(m)

        # this problem should turn into
        #
        # aux2 + y + z = 0        => aux_con[1]
        # w - 3*aux2 = 0          => aux_con[2]
        # aux1 = x**2             => rel0
        # aux2 = x*aux1           => rel1

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 2)

        self.assertAlmostEqual(rel.aux_vars[1].lb, 0)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertAlmostEqual(rel.aux_vars[2].lb, -1)
        self.assertAlmostEqual(rel.aux_vars[2].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[2]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[2]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWXSquaredRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))

        self.assertTrue(hasattr(rel.relaxations, 'rel1'))
        self.assertTrue(isinstance(rel.relaxations.rel1, coramin.relaxations.PWMcCormickRelaxation))
        self.assertIn(id(rel.x), {id(rel.relaxations.rel1._x), id(rel.relaxations.rel1._y)})
        self.assertIn(id(rel.aux_vars[1]), {id(rel.relaxations.rel1._x), id(rel.relaxations.rel1._y)})
        self.assertEqual(id(rel.aux_vars[2]), id(rel.relaxations.rel1._w))

    def test_pow_fractional1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=0.5)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertFalse(rel.relaxations.rel0.is_convex())
        self.assertTrue(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_fractional2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=1.5)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_neg_even1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1,2))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=-2)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0.25)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_neg_even2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2,-1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=-2)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0.25)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_neg_odd1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1,2))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=-3)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, 0.125)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_neg_odd2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2,-1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=-3)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 1)
        self.assertAlmostEqual(rel.aux_vars[1].lb, -1)
        self.assertAlmostEqual(rel.aux_vars[1].ub, -0.125)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[1]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[1]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWUnivariateRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertFalse(rel.relaxations.rel0.is_convex())
        self.assertTrue(rel.relaxations.rel0.is_concave())
        self.assertFalse(hasattr(rel.relaxations, 'rel1'))

    def test_pow_neg(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.w = pe.Var()
        m.p = pe.Param(initialize=-2)
        m.c = pe.Constraint(expr=m.x**m.p + m.y + m.z == 0)
        m.c2 = pe.Constraint(expr=m.w - 3*m.x**m.p == 0)

        rel = coramin.relaxations.relax(m)

        # This model should be relaxed to
        #
        # aux2 + y + z = 0
        # w - 3 * aux2 = 0
        # aux1 = x**2
        # aux1*aux2 = aux3
        # aux3 = 1
        #

        self.assertTrue(hasattr(rel, 'aux_cons'))
        self.assertTrue(hasattr(rel, 'aux_vars'))
        self.assertEqual(len(rel.aux_cons), 2)
        self.assertEqual(len(rel.aux_vars), 3)

        self.assertAlmostEqual(rel.aux_vars[1].lb, 0)
        self.assertAlmostEqual(rel.aux_vars[1].ub, 1)

        self.assertTrue(rel.aux_vars[3].is_fixed())
        self.assertEqual(rel.aux_vars[3].value, 1)

        self.assertEqual(rel.aux_cons[1].lower, 0)
        self.assertEqual(rel.aux_cons[1].upper, 0)
        ders = reverse_sd(rel.aux_cons[1].body)
        self.assertEqual(ders[rel.z], 1)
        self.assertEqual(ders[rel.aux_vars[2]], 1)
        self.assertEqual(ders[rel.y], 1)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[1].body))), 3)

        self.assertEqual(rel.aux_cons[2].lower, 0)
        self.assertEqual(rel.aux_cons[2].upper, 0)
        ders = reverse_sd(rel.aux_cons[2].body)
        self.assertEqual(ders[rel.w], 1)
        self.assertEqual(ders[rel.aux_vars[2]], -3)
        self.assertEqual(len(list(identify_variables(rel.aux_cons[2].body))), 2)

        self.assertTrue(hasattr(rel, 'relaxations'))
        self.assertTrue(hasattr(rel.relaxations, 'rel0'))
        self.assertTrue(isinstance(rel.relaxations.rel0, coramin.relaxations.PWXSquaredRelaxation))
        self.assertEqual(id(rel.x), id(rel.relaxations.rel0._x))
        self.assertEqual(id(rel.aux_vars[1]), id(rel.relaxations.rel0._w))
        self.assertTrue(rel.relaxations.rel0.is_convex())
        self.assertFalse(rel.relaxations.rel0.is_concave())

        self.assertTrue(hasattr(rel.relaxations, 'rel1'))
        self.assertTrue(isinstance(rel.relaxations.rel1, coramin.relaxations.PWMcCormickRelaxation))
        self.assertIn(id(rel.aux_vars[1]), {id(rel.relaxations.rel1._x), id(rel.relaxations.rel1._y)})
        self.assertIn(id(rel.aux_vars[2]), {id(rel.relaxations.rel1._x), id(rel.relaxations.rel1._y)})
        self.assertEqual(id(rel.aux_vars[3]), id(rel.relaxations.rel1._w))

        self.assertFalse(hasattr(rel.relaxations, 'rel2'))
