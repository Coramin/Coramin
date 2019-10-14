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
from pyomo.core.expr import taylor_series_expansion

pyo = pe
logger = logging.getLogger(__name__)

"""
Base classes for relaxations
"""


class _ValueManager(object):
    def __init__(self):
        self.orig_values = pe.ComponentMap()

    def save_values(self, variables):
        self.orig_values = pe.ComponentMap()
        for v in variables:
            self.orig_values[v] = v.value

    def pop_values(self):
        for v, val in self.orig_values.items():
            v.value = val


def _load_var_values(var_value_map):
    for v, val in var_value_map.items():
        v.value = val


@declare_custom_block(name='BaseRelaxation')
class BaseRelaxationData(_BlockData):
    def __init__(self, component):
        _BlockData.__init__(self, component)
        self._persistent_solvers = ComponentSet()
        self._allow_changes = False
        self._relaxation_side = RelaxationSide.BOTH
        self._oa_points = list()  # List of ComponentMap. Each entry in the list specifies a point at which an outer
                                  # approximation cut should be built for convex/concave constraints.
        self._saved_oa_points = list()
        self._cuts = None
        self._use_linear_relaxation = True
        self.large_eval_tol = math.inf

    def add_component(self, name, val):
        if self._allow_changes:
            _BlockData.add_component(self, name, val)
        else:
            raise RuntimeError('Pyomo components cannot be added to objects of type {0}.'.format(type(self)))

    def _set_input(self, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
                   use_linear_relaxation=True, large_eval_tol=math.inf):
        self._oa_points = list()
        self._saved_oa_points = list()
        self._persistent_solvers = persistent_solvers
        if self._persistent_solvers is None:
            self._persistent_solvers = ComponentSet()
        if not isinstance(self._persistent_solvers, Iterable):
            self._persistent_solvers = ComponentSet([self._persistent_solvers])
        else:
            self._persistent_solvers = ComponentSet(self._persistent_solvers)
        self._relaxation_side = relaxation_side
        assert self._relaxation_side in RelaxationSide
        self.use_linear_relaxation = use_linear_relaxation
        self.large_eval_tol = large_eval_tol

    def get_aux_var(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns w

        Returns
        -------
        aux_var: pyomo.core.base.var._GeneralVarData
            The variable representing w in w = f(x) (which is the constraint being relaxed).
        """
        return self._aux_var

    def get_rhs_vars(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    def get_rhs_expr(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    @property
    def use_linear_relaxation(self):
        """
        If this is True, the relaxation will use a linear relaxation. If False, then a nonlinear relaxation may be used.
        Take x^2 for example, the underestimator can be quadratic.

        Returns
        -------
        bool
        """
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        if not val:
            raise ValueError('Relaxations of type {0} do not support relaxations that are not linear.'.format(type(self)))

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

    def rebuild(self, build_nonlinear_constraint=False):
        """
        Remove any auto-created vars/constraints from the relaxation block and recreate it
        """
        self.clean_oa_points()
        self._allow_changes = True
        self.remove_relaxation()
        if build_nonlinear_constraint:
            if self.relaxation_side == RelaxationSide.BOTH:
                self.nonlinear_con = pe.Constraint(expr=self.get_aux_var() == self.get_rhs_expr())
            elif self.relaxation_side == RelaxationSide.UNDER:
                self.nonlinear_con = pe.Constraint(expr=self.get_aux_var() >= self.get_rhs_expr())
            else:
                self.nonlinear_con = pe.Constraint(expr=self.get_aux_var() <= self.get_rhs_expr())
        else:
            self._build_relaxation()
            if self.use_linear_relaxation:
                val_mngr = _ValueManager()
                val_mngr.save_values(self.get_rhs_vars())
                for pt in self._oa_points:
                    _load_var_values(pt)
                    # check_violation has to be False because we are not loading the value of aux_var
                    self.add_cut(keep_cut=False, check_violation=False, add_cut_to_persistent_solvers=False)
                val_mngr.pop_values()
            else:
                if self.is_rhs_convex() and self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}:
                    self.nonlinear_underestimator = pe.Constraint(expr=self.get_aux_var() >= self.get_rhs_expr())
                elif self.is_rhs_concave() and self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
                    self.nonlinear_overestimator = pe.Constraint(expr=self.get_aux_var() <= self.get_rhs_expr())
        self._add_to_persistent_solvers()
        self._allow_changes = False

    def _build_relaxation(self):
        """
        Build the auto-created vars/constraints that form the relaxation
        """
        raise NotImplementedError('This should be implemented in the derived class.')

    def vars_with_bounds_in_relaxation(self):
        """
        This method returns a list of variables whose bounds appear in the constraints defining the relaxation.
        Take the McCormick relaxation of a bilinear term (w = x * y) for example. The McCormick relaxation is

        w >= xl * y + x * yl - xl * yl
        w >= xu * y + x * yu - xu * yu
        w <= xu * y + x * yl - xu * yl
        w <= x * yu + xl * y - xl * yu

        where xl and xu are the lower and upper bounds for x, respectively, and yl and yu are the lower and upper
        bounds for y, respectively. Because xl, xu, yl, and yu appear in the constraints, this method would return

        [x, y]

        As another example, take w >= x**2. A linear relaxation of this constraint just involves linear underestimators,
        which do not depend on the bounds of x or w. Therefore, this method would return an empty list.
        """
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

    def get_deviation(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns

        max{f(x) - w, 0} if relaxation_side is RelaxationSide.UNDER
        max{w - f(x), 0} if relaxation_side is RelaxationSide.OVER
        abs(w - f(x)) if relaxation_side is RelaxationSide.BOTH

        Returns
        -------
        float
        """
        dev = self.get_aux_var().value - pe.value(self.get_rhs_expr())
        if self.relaxation_side is RelaxationSide.BOTH:
            dev = abs(dev)
        elif self.relaxation_side is RelaxationSide.UNDER:
            dev = max(-dev, 0)
        else:
            dev = max(dev, 0)
        return dev

    def is_rhs_convex(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns True if f(x)
        is convex and False otherwise.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def is_rhs_concave(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns True if f(x)
        is concave and False otherwise.

        Returns
        -------
        bool
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
        return 'Relaxation for {0} {1} {2}'.format(self.get_aux_var().name, relational_operator_string, str(self.get_rhs_expr()))

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        if filename is not None:
            output = open(filename, 'w')
            self.pprint(ostream=output, verbose=verbose, prefix=prefix)
            output.close()

        else:
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
            ostream.write('{0}{1}: {2}\n'.format(prefix, self.name, self._get_pprint_string(relational_operator)))

        if verbose:
            super(BaseRelaxationData, self).pprint(filename=filename, ostream=ostream,
                                                   verbose=verbose, prefix=(prefix + '  '))

    def add_oa_point(self, var_values=None):
        """
        Add a point at which an outer-approximation cut for a convex constraint should be added. This does not
        rebuild the relaxation. You must call rebuild() for the constraint to get added.

        Parameters
        ----------
        var_values: pe.ComponentMap
        """
        if var_values is None:
            var_values = pe.ComponentMap()
            for v in self.get_rhs_vars():
                var_values[v] = v.value
        else:
            var_values = pe.ComponentMap(var_values)
        self._oa_points.append(var_values)

    def push_oa_points(self):
        """
        Save the current list of OA points for later use (this clears the current set of OA points until popped.)
        """
        self._saved_oa_points.append(self._oa_points)
        self.clear_oa_points()

    def clear_oa_points(self):
        """
        Delete any existing OA points.
        """
        self._oa_points = list()

    def pop_oa_points(self):
        """
        Use the most recently saved list of OA points
        """
        self._oa_points = self._saved_oa_points.pop(-1)

    def add_cut(self, keep_cut=True, check_violation=False, feasibility_tol=1e-8, add_cut_to_persistent_solvers=True):
        """
        This function will add a linear cut to the relaxation. Cuts are only generated for the convex side of the
        constraint (if the constraint has a convex side). For example, if the relaxation is a PWXSquaredRelaxationData
        for y = x**2, the add_cut will add an underestimator at x.value (but only if y.value < x.value**2). If
        relaxation is a PWXSquaredRelaxationData for y < x**2, then no cut will be added. If relaxation is is a
        PWMcCormickRelaxationData, then no cut will be added.

        Parameters
        ----------
        keep_cut: bool
            If keep_cut is True, then add_oa_point will also be called. Be careful if the relaxation object is relaxing
            the nonconvex side of the constraint. Thus, the cut will be reconstructed when rebuild is called. If
            keep_cut is False, then the cut will be discarded when rebuild is called.
        check_violation: bool
            If True, then a cut is only added if the cut generated would cut off the current point (current values
            of the variables) by more than feasibility_tol.
        feasibility_tol: float
            Only used if check_violation is True
        add_cut_to_persistent_solvers: bool

        Returns
        -------
        new_con: pyomo.core.base.constraint.Constraint
        """
        if keep_cut:
            self.add_oa_point()

        cut_expr = None

        rhs_val = pe.value(self.get_rhs_expr())
        if rhs_val >= self.large_eval_tol or rhs_val < - self.large_eval_tol:
            pass
        else:
            try:
                if self.is_rhs_convex():
                    if self.relaxation_side == RelaxationSide.UNDER or self.relaxation_side == RelaxationSide.BOTH:
                        if check_violation:
                            viol = self.get_aux_var().value - rhs_val
                            if viol < -feasibility_tol:
                                cut_expr = self.get_aux_var() >= taylor_series_expansion(self.get_rhs_expr())
                        else:
                            cut_expr = self.get_aux_var() >= taylor_series_expansion(self.get_rhs_expr())
                elif self.is_rhs_concave():
                    if self.relaxation_side == RelaxationSide.OVER or self.relaxation_side == RelaxationSide.BOTH:
                        if check_violation:
                            viol = self.get_aux_var().value - rhs_val
                            if viol > feasibility_tol:
                                cut_expr = self.get_aux_var() <= taylor_series_expansion(self.get_rhs_expr())
                        else:
                            cut_expr = self.get_aux_var() <= taylor_series_expansion(self.get_rhs_expr())
            except (OverflowError, ZeroDivisionError, ValueError):
                pass
        if cut_expr is not None:
            if not hasattr(self, '_cuts'):
                self._cuts = None
            if self._cuts is None:
                del self._cuts
                self._allow_changes = True
                self._cuts = pe.ConstraintList()
                self._allow_changes = False
            new_con = self._cuts.add(cut_expr)
            if add_cut_to_persistent_solvers:
                for i in self._persistent_solvers:
                    i.add_constraint(new_con)
        else:
            new_con = None

        return new_con

    def clean_oa_points(self):
        # For each OA point, if the point is outside variable bounds, move the point to the variable bounds
        for pts in self._oa_points:
            for v, pt in pts.items():
                lb, ub = tuple(_get_bnds_list(v))
                if pt < lb:
                    pts[v] = lb
                if pt > ub:
                    pts[v] = ub


@declare_custom_block(name='BasePWRelaxation')
class BasePWRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        BaseRelaxationData.__init__(self, component)

        self._partitions = ComponentMap()  # ComponentMap: var: list of float
        self._saved_partitions = list()  # list of CompnentMap

    def rebuild(self, build_nonlinear_constraint=False):
        """
        Remove any auto-created vars/constraints from the relaxation block and recreate it
        """
        self.clean_partitions()
        super(BasePWRelaxationData, self).rebuild(build_nonlinear_constraint=build_nonlinear_constraint)

    def _set_input(self, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
                   use_linear_relaxation=True, large_eval_tol=math.inf):
        self._partitions = ComponentMap()
        self._saved_partitions = list()
        BaseRelaxationData._set_input(self, relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                                      use_linear_relaxation=use_linear_relaxation, large_eval_tol=large_eval_tol)

    def add_parition_point(self):
        """
        Add a point to the current partitioning. This does not rebuild the relaxation. You must call rebuild()
        to rebuild the relaxation.
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def _add_partition_point(self, var, value=None):
        if value is None:
            value = pe.value(var)
        # if the point is outside the variable's bounds, then it will simply get removed when clean_partitions
        # gets called.
        self._partitions[var].append(value)

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

    def get_active_partitions(self):
        ans = ComponentMap()
        for var, pts in self._partitions.items():
            val = pyo.value(var)
            lower = var.lb
            upper = var.ub
            for p in pts:
                if val >= p and p > lower:
                    lower = p
                if val <= p and p < upper:
                    upper = p
            ans[var] = lower, upper
        return ans


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
