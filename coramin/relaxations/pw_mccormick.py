import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .mccormick import _build_mccormick_relaxation
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef
from .custom_block import declare_custom_block
from ._utils import var_info_str, bnds_info_str, x_pts_info_str, check_var_pts

import logging
logger = logging.getLogger(__name__)


def _build_pw_mccormick_relaxation(b, x, y, w, x_pts, relaxation_side=RelaxationSide.BOTH):
    """
    This function creates piecewise envelopes to relax "w = x*y". Note that the partitioning is done on "x" only.
    This is the "nf4r" from Gounaris, Misener, and Floudas (2009).

    Parameters
    ----------
    b: pyo.ConcreteModel or pyo.Block
    x: pyomo.core.base.var._GeneralVarData
        The "x" variable in x*y
    y: pyomo.core.base.var._GeneralVarData
        The "y" variable in x*y
    w: pyomo.core.base.var._GeneralVarData
        The "w" variable that is replacing x*y
    x_pts: list of floats
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the
        last point (x_pts[-1]) is equal to x.ub
    relaxation_side : minlp.minlp_defn.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)

    """
    xlb = pyo.value(x.lb)
    xub = pyo.value(x.ub)
    ylb = pyo.value(y.lb)
    yub = pyo.value(y.ub)

    check_var_pts(x, x_pts=x_pts)
    check_var_pts(y)

    if x.is_fixed() and y.is_fixed():
        b.xy_fixed_eq = pyo.Constraint(expr= w == pyo.value(x) * pyo.value(y))
    elif x.is_fixed():
        b.x_fixed_eq = pyo.Constraint(expr= w == pyo.value(x) * y)
    elif y.is_fixed():
        b.y_fixed_eq = pyo.Constraint(expr= w == x * pyo.value(y))
    elif len(x_pts) == 2:
        _build_mccormick_relaxation(b, x=x, y=y, w=w, relaxation_side=relaxation_side)
    else:
        # create the lambda variables (binaries for the pw representation)
        b.interval_set = pyo.Set(initialize=range(1, len(x_pts)))
        b.lam = pyo.Var(b.interval_set, within=pyo.Binary)

        # create the delta y variables
        b.delta_y = pyo.Var(b.interval_set, bounds=(0, None))

        # create the "sos1" constraint
        b.lam_sos1 = pyo.Constraint(expr=sum(b.lam[n] for n in b.interval_set) == 1.0)

        # create the x interval constraints
        b.x_interval_lb = pyo.Constraint(expr=sum(x_pts[n - 1] * b.lam[n] for n in b.interval_set) <= x)
        b.x_interval_ub = pyo.Constraint(expr=x <= sum(x_pts[n] * b.lam[n] for n in b.interval_set))

        # create the y constraints
        b.y_con = pyo.Constraint(expr=y == ylb + sum(b.delta_y[n] for n in b.interval_set))

        def delta_yn_ub_rule(m, n):
            return b.delta_y[n] <= (yub - ylb) * b.lam[n]

        b.delta_yn_ub = pyo.Constraint(b.interval_set, rule=delta_yn_ub_rule)

        # create the relaxation constraints
        if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
            b.w_lb1 = pyo.Constraint(expr=(w >= yub * x + sum(x_pts[n] * b.delta_y[n] for n in b.interval_set) -
                                           (yub - ylb) * sum(x_pts[n] * b.lam[n] for n in b.interval_set)))
            b.w_lb2 = pyo.Constraint(
                expr=w >= ylb * x + sum(x_pts[n - 1] * b.delta_y[n] for n in b.interval_set))

        if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
            b.w_ub1 = pyo.Constraint(expr=(w <= yub * x + sum(x_pts[n - 1] * b.delta_y[n] for n in b.interval_set) -
                                           (yub - ylb) * sum(x_pts[n - 1] * b.lam[n] for n in b.interval_set)))
            b.w_ub2 = pyo.Constraint(expr=w <= ylb * x + sum(x_pts[n] * b.delta_y[n] for n in b.interval_set))


@declare_custom_block(name='PWMcCormickRelaxation')
class PWMcCormickRelaxationData(BasePWRelaxationData):
    """
    A class for managing McCormick relaxations.

    Parameters
    ----------
    x : pyomo.core.base.var._GeneralVarData
        The "x" variable in x*y
    y : pyomo.core.base.var._GeneralVarData
        The "y" variable in x*y
    w : pyomo.core.base.var._GeneralVarData
        The "w" auxillary variable that is replacing x*y
    relaxation_side : minlp.minlp_defn.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)

    Examples
    --------
    Example 1: w_i = x_i * y_i
    >>> import coramin
    >>> import pyomo.environ as pe
    >>>
    >>> m = pe.ConcreteModel()
    >>> a = [1,2,3]
    >>> m.x = pe.Var(a, bounds=(-1,1))
    >>> m.y = pe.Var(a, bounds=(-1,1))
    >>> m.w = pe.Var(a)
    >>>
    >>> def c_rule(b, i):
    ...     b.build(relaxation_side=RelaxationSide.BOTH, x=m.x[i], y=m.y[i], w=m.w[i])
    >>> m.mcc = coramin.relaxations.PWMcCormickRelaxation(a, rule=c_rule)
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._yref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _y(self):
        return self._yref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        return [self._x, self._y]

    def _set_input(self, kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y')
        w = kwargs.pop('w')
        self._xref.set_component(x)
        self._yref.set_component(y)
        self._wref.set_component(w)
        self._partitions[self._x] = [pyo.value(self._x.lb), pyo.value(self._x.ub)]

    def _build_relaxation(self):
        _build_pw_mccormick_relaxation(b=self, x=self._x, y=self._y, w=self._w, x_pts=self._partitions[self._x],
                                       relaxation_side=self._relaxation_side)

    def add_point(self, value=None):
        """
        This method adds one point to the partitioning of x. If value is not
        specified, a single point will be added to the partitioning of x at the current value of x. If value is
        specified, then value is added to the partitioning of x.

        Parameters
        ----------
        value: float
            The point to be added to the partitioning of x.
        """
        self._add_point(self._x, value)

    def _get_violation(self):
        """
        Get the absolute value of the current constraint violation.

        Returns
        -------
        violation: float
        """
        return self._w.value - self._x.value * self._y.value

    def is_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False

    def is_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False

    @property
    def use_linear_relaxation(self):
        return True

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        if val is not True:
            raise ValueError('PWMcCormickRelaxation only supports linear relaxations.')
