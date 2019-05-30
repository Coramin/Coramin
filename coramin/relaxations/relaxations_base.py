from pyomo.core.base.block import _BlockData, Block
from .custom_block import declare_custom_block
import weakref
import pyomo.environ as pe
from collections import Iterable
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from coramin.utils.coramin_enums import FunctionShape, RelaxationSide
import warnings
import logging
import math
from ._utils import _get_bnds_list
import sys

pyo = pe
logger = logging.getLogger(__name__)

"""
Base classes for relaxations
"""


@declare_custom_block(name='BaseRelaxation')
class BaseRelaxationData(_BlockData):
    def __init__(self, component):
        _BlockData.__init__(self, component)
        self._persistent_solvers = ComponentSet()
        self._allow_changes = False
        self._relaxation_side = RelaxationSide.BOTH

    def add_component(self, name, val):
        if self._allow_changes:
            _BlockData.add_component(self, name, val)
        else:
            raise RuntimeError('Pyomo components cannot be added to objects of type {0}.'.format(type(self)))

    def _set_input(self, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self._persistent_solvers = persistent_solvers
        if self._persistent_solvers is None:
            self._persistent_solvers = ComponentSet()
        if not isinstance(self._persistent_solvers, Iterable):
            self._persistent_solvers = ComponentSet([self._persistent_solvers])
        else:
            self._persistent_solvers = ComponentSet(self._persistent_solvers)
        self._relaxation_side = relaxation_side
        assert self._relaxation_side in RelaxationSide

    @property
    def use_linear_relaxation(self):
        """
        If this is True, the relaxation will use a linear relaxation. If False, then a nonlinear relaxation may be used.
        Take x^2 for example, the underestimator can be quadratic.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This property should be implemented by subclasses.')

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        raise NotImplementedError('This property setter should be implemented by subclasses.')

    def remove_relaxation(self):
        """
        Remove any auto-created vars/constraints from the relaxation block
        """
        # this default implementation should work for most relaxations
        # it removes all vars and constraints on this block data object
        self._remove_from_persistent_solvers()
        comps = [pe.Block, pe.Constraint, pe.Var, pe.Set, pe.Param]
        for comp in comps:
            comps_to_del = list(self.component_objects([comp], descend_into=False))
            for _comp in comps_to_del:
                self.del_component(_comp)
        for comp in comps:
            comps_to_del = list(self.component_data_objects([comp], descend_into=False))
            for _comp in comps_to_del:
                self.del_component(_comp)

    def rebuild(self):
        """
        Remove any auto-created vars/constraints from the relaxation block and recreate it
        """
        self._allow_changes = True
        self.remove_relaxation()
        self._build_relaxation()
        self._add_to_persistent_solvers()
        self._allow_changes = False

    def rebuild_nonlinear(self):
        self._allow_changes = True
        self.remove_relaxation()
        self._build_nonlinear()
        self._add_to_persistent_solvers()
        self._allow_changes = False

    def _build_relaxation(self):
        """
        Build the auto-created vars/constraints that form the relaxation
        """
        raise NotImplementedError('This should be implemented in the derived class.')

    def _build_nonlinear(self):
        """
        Build the nonlinear constraint.
        """
        raise NotImplementedError('This should be implemented in the derived class.')

    def vars_with_bounds_in_relaxation(self):
        raise NotImplementedError('This method should be implemented in the derived class.')

    def _remove_from_persistent_solvers(self):
        for i in self._persistent_solvers:
            i.remove_block(block=self)

    def _add_to_persistent_solvers(self):
        for i in self._persistent_solvers:
            i.add_block(block=self)

    def add_persistent_solver(self, persistent_solver):
        self._persistent_solvers.add(persistent_solver)

    def remove_persistent_solver(self, persistent_solver):
        self._persistent_solvers.remove(persistent_solver)

    def clear_persistent_solvers(self):
        self._persistent_solvers = ComponentSet()

    def get_abs_violation(self):
        """
        Compute the absolute value of the constraint violation given the current values of the corresponding vars.

        Returns
        -------
        float
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    @property
    def relaxation_side(self):
        return self._relaxation_side

    @relaxation_side.setter
    def relaxation_side(self, val):
        if val not in RelaxationSide:
            raise ValueError('{0} is not a valid member of RelaxationSide'.format(val))
        self._relaxation_side = val

    def _get_pprint_string(self, relational_operator_string):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        if filename is not None:
            output = open(filename, 'w')
            self.pprint(ostream=output, verbose=verbose, prefix=prefix)
            output.close()
            return

        if ostream is None:
            ostream = sys.stdout

        if self.relaxation_side == RelaxationSide.BOTH:
            relational_operator = '=='
        elif self.relaxation_side == RelaxationSide.UNDER:
            relational_operator = '>='
        elif self.relaxation_side == RelaxationSide.OVER:
            relational_operator = '<='
        else:
            raise ValueError('Unexpected relaxation side')
        ostream.write('{0}{1}\n'.format(prefix, self._get_pprint_string(relational_operator)))


@declare_custom_block(name='BasePWRelaxation')
class BasePWRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        BaseRelaxationData.__init__(self, component)

        self._partitions = ComponentMap()
        """ComponentMap: var: list of float"""

        self._saved_partitions = []
        """list of CompnentMap"""

    def rebuild(self):
        """
        Remove any auto-created vars/constraints from the relaxation block and recreate it
        """
        self.clean_partitions()
        BaseRelaxationData.rebuild(self)

    def _set_input(self, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self._partitions = ComponentMap()
        self._saved_partitions = []
        BaseRelaxationData._set_input(self, relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)

    def add_point(self):
        """
        Add a point to the current partitioning. This does not rebuild the relaxation. You must call build_relaxation
        to rebuild the relaxation.
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def _add_point(self, var, value=None):
        if value is not None:
            vlb, vub = tuple(_get_bnds_list(var))
            if (vlb < value) and (value < vub):
                self._partitions[var].append(value)
            else:
                e = 'The value provided to add_point was not between the variables lower \n' + \
                              'and upper bounds. No point was added.'
                warnings.warn(e)
                logger.warning(e)
        else:
            self._partitions[var].append(var.value)

    def push_partitions(self):
        """
        Save the current partitioning and then clear the current partitioning
        """
        self._saved_partitions.append(self._partitions)
        self.clear_partitions()

    def clear_partitions(self):
        """
        Delete any existing partitioning scheme.
        """
        tmp = ComponentMap()
        for var, pts in self._partitions.items():
            tmp[var] = [pe.value(var.lb), pe.value(var.ub)]
        self._partitions = tmp

    def pop_partitions(self):
        """
        Use the most recently saved partitioning.
        """
        self._partitions = self._saved_partitions.pop(-1)

    def clean_partitions(self):
        # discard any points in the partitioning that are not within the variable bounds
        for var, pts in self._partitions.items():
            pts.sort()

        for var, pts in self._partitions.items():
            lb, ub = tuple(_get_bnds_list(var))

            if pts[0] < lb or pts[-1] > ub:
                pts = [v for v in pts if (lb < v < ub)]
                pts.insert(0, lb)
                pts.append(ub)
                self._partitions[var] = pts

    def is_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def is_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def add_cut(self):
        if not hasattr(self, '_cuts'):
            self._allow_changes = True
            self._cuts = pyo.ConstraintList()
            self._allow_changes = False
        expr = self._get_cut_expr()
        if expr is not None:
            new_con = self._cuts.add(expr)
            for i in self._persistent_solvers:
                i.add_constraint(new_con)

    def _get_cut_expr(self):
        raise NotImplementedError('The add_cut method is not implemented for objects of type {0}.'.format(type(self)))

    def get_abs_violation(self):
        return abs(self.get_violation())

    def get_violation(self):
        viol = self._get_violation()
        if viol >= 0 and self._relaxation_side == RelaxationSide.UNDER:
            viol = 0
        elif viol <= 0 and self._relaxation_side == RelaxationSide.OVER:
            viol = 0
        return viol


class ComponentWeakRef(object):
    """
    This object is used to reference components from a block that are not owned by that block.
    """
    # ToDo: Example in the documentation
    def __init__(self, comp):
        self.compref = None
        self.set_component(comp)

    def get_component(self):
        if self.compref is None:
            return None
        return self.compref()

    def set_component(self, comp):
        self.compref = None
        if comp is not None:
            self.compref = weakref.ref(comp)

    def __setstate__(self, state):
        self.set_component(state['compref'])

    def __getstate__(self):
        return {'compref': self.get_component()}
