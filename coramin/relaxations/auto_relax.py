import pyomo.environ as pe
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import (ExpressionValueVisitor,
                                         nonpyomo_leaf_types, value,
                                         identify_variables)
from pyomo.core.expr.numvalue import is_fixed, polynomial_degree
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr, fbbt
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
import logging
from .univariate import PWUnivariateRelaxation, PWXSquaredRelaxation, PWCosRelaxation, PWSinRelaxation, PWArctanRelaxation
from .pw_mccormick import PWMcCormickRelaxation
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from pyomo.gdp import Disjunct

logger = logging.getLogger(__name__)

if not hasattr(math, 'inf'):
    math.inf = float('inf')


"""
The purpose of this file is to perform feasibility based bounds 
tightening. This is a very basic implementation, but it is done 
directly with pyomo expressions. The only functions that are meant to 
be used by users are fbbt, fbbt_con, and fbbt_block. The first set of 
functions in this file (those with names starting with 
_prop_bnds_leaf_to_root) are used for propagating bounds from the  
variables to each node in the expression tree (all the way to the  
root node). The second set of functions (those with names starting 
with _prop_bnds_root_to_leaf) are used to propagate bounds from the 
constraint back to the variables. For example, consider the constraint 
x*y + z == 1 with -1 <= x <= 1 and -2 <= y <= 2. When propagating 
bounds from the variables to the root (the root is x*y + z), we find 
that -2 <= x*y <= 2, and that -inf <= x*y + z <= inf. However, 
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
propagate bounds back to the variables. Since we know that 
1 <= x*y + z <= 1 and -2 <= x*y <= 2, then we must have -1 <= z <= 3. 
However, bounds cannot be improved on x*y, so bounds cannot be 
improved on either x or y.

>>> import pyomo.environ as pe
>>> m = pe.ConcreteModel()
>>> m.x = pe.Var(bounds=(-1,1))
>>> m.y = pe.Var(bounds=(-2,2))
>>> m.z = pe.Var()
>>> from pyomo.contrib.fbbt.fbbt import fbbt
>>> m.c = pe.Constraint(expr=m.x*m.y + m.z == 1)
>>> fbbt(m)
>>> print(m.z.lb, m.z.ub)
-1.0 3.0

"""


class RelaxationException(Exception):
    pass


class RelaxationCounter(object):
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def __str__(self):
        return str(self.count)


def replace_sub_expression_with_aux_var(arg, parent_block):
    if type(arg) in nonpyomo_leaf_types:
        return arg
    elif arg.is_expression_type():
        _var = parent_block.aux_vars.add()
        _con = parent_block.aux_cons.add(_var == arg)
        fbbt(_con)
        return _var
    else:
        return arg


def _relax_leaf_to_root_ProductExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg1, arg2 = values
    degree_1 = degree_map[arg1]
    degree_2 = degree_map[arg2]
    if degree_1 == 0 or degree_2 == 0:
        degree_map[node] = degree_1 + degree_2
        return arg1 * arg2
    elif (id(arg1), id(arg2), 'mul') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg1), id(arg2), 'mul']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        return _aux_var
    else:
        if not hasattr(parent_block, 'aux_vars'):
            parent_block.aux_vars = pe.VarList()
        _aux_var = parent_block.aux_vars.add()
        arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
        arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWMcCormickRelaxation()
        relaxation.set_input(x=arg1, y=arg2, w=_aux_var, relaxation_side=relaxation_side)
        aux_var_map[id(arg1), id(arg2), 'mul'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_SumExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    res = sum(values)
    degree_map[res] = max([degree_map[arg] for arg in values])
    return res


def _relax_leaf_to_root_NegationExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    res = -arg
    degree_map[res] = degree_map[arg]
    return res


_relax_leaf_to_root_map = dict()
_relax_leaf_to_root_map[_expr.ProductExpression] = _relax_leaf_to_root_ProductExpression
_relax_leaf_to_root_map[_expr.SumExpression] = _relax_leaf_to_root_SumExpression
_relax_leaf_to_root_map[_expr.MonomialTermExpression] = _relax_leaf_to_root_ProductExpression
_relax_leaf_to_root_map[_expr.NegationExpression] = _relax_leaf_to_root_NegationExpression


def _relax_root_to_leaf_ProductExpression(node, relaxation_side_map):
    arg1, arg2 = node.args
    relaxation_side_map[arg1] = RelaxationSide.BOTH
    relaxation_side_map[arg2] = RelaxationSide.BOTH


def _relax_root_to_leaf_SumExpression(node, relaxation_side_map):
    relaxation_side = relaxation_side_map[node]

    for arg in node.args:
        relaxation_side_map[arg] = relaxation_side


def _relax_root_to_leaf_NegationExpression(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side = relaxation_side_map[node]
    if relaxation_side == RelaxationSide.BOTH:
        relaxation_side_map[arg] = RelaxationSide.BOTH
    elif relaxation_side == RelaxationSide.UNDER:
        relaxation_side_map[arg] = RelaxationSide.OVER
    else:
        assert relaxation_side == RelaxationSide.OVER
        relaxation_side_map[arg] = RelaxationSide.UNDER


_relax_root_to_leaf_map = dict()
_relax_root_to_leaf_map[_expr.ProductExpression] = _relax_root_to_leaf_ProductExpression
_relax_root_to_leaf_map[_expr.SumExpression] = _relax_root_to_leaf_SumExpression
_relax_root_to_leaf_map[_expr.MonomialTermExpression] = _relax_root_to_leaf_ProductExpression
_relax_root_to_leaf_map[_expr.NegationExpression] = _relax_root_to_leaf_NegationExpression


class _FactorableRelaxationVisitor(ExpressionValueVisitor):
    """
    This walker generates new constraints with nonlinear terms replaced by
    auxiliary variables, and relaxations relating the auxilliary variables to
    the original variables.
    """
    def __init__(self, aux_var_map, parent_block, relaxation_side_map, counter):
        self.aux_var_map = aux_var_map
        self.parent_block = parent_block
        self.relaxation_side_map = relaxation_side_map
        self.counter = counter
        self.degree_map = ComponentMap()

    def visit(self, node, values):
        if node.__class__ in _relax_leaf_to_root_map:
            return _relax_leaf_to_root_map[node.__class__](node, values, self.aux_var_map, self.degree_map,
                                                           self.parent_block, self.relaxation_side_map, self.counter)
        else:
            raise NotImplementedError('Cannot relax an expression of type ' + str(type(node)))

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.degree_map[node] = 0
            return True, node

        if node.is_variable_type():
            self.degree_map[node] = 1
            return True, node

        if not node.is_expression_type():
            self.degree_map[node] = 0
            return True, node

        if node.__class__ in _relax_root_to_leaf_map:
            _relax_root_to_leaf_map[node.__class__](node, self.relaxation_side_map)

        return False, None


def relax(model, descend_into=None):
    m = model.clone()

    if descend_into is None:
        descend_into = (pe.Block, Disjunct)

    aux_var_map = dict()
    counter_dict = dict()

    for c in m.component_data_objects(ctype=Constraint, active=True, descend_into=descend_into, sort=True):
        if polynomial_degree(c.body) <= 1:
            continue

        if c.lower is not None and c.upper is not None:
            relaxation_side = RelaxationSide.BOTH
        elif c.lower is not None:
            relaxation_side = RelaxationSide.OVER
        elif c.upper is not None:
            relaxation_side = RelaxationSide.UNDER
        else:
            raise ValueError('Encountered a constraint without a lower or an upper bound: ' + str(c))

        parent_block = c.parent_block()
        relaxation_side_map = ComponentMap()
        relaxation_side_map[c.body] = relaxation_side

        if parent_block in counter_dict:
            counter = counter_dict[parent_block]
        else:
            parent_block.relaxations = pe.Block()
            parent_block.aux_vars = pe.VarList()
            parent_block.aux_cons = pe.ConstraintList()
            counter = RelaxationCounter()
            counter_dict[parent_block] = counter

        visitor = _FactorableRelaxationVisitor(aux_var_map, parent_block, relaxation_side_map, counter)
        new_body = visitor.dfs_postorder_stack(c.body)
        lb = c.lower
        ub = c.upper
        parent_block.aux_cons.add(pe.inequality(lb, new_body, ub))
        parent_block.del_component(c)

    for _aux_var, relaxation in aux_var_map.values():
        relaxation.rebuild()

    return m
