import logging
import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .custom_block import declare_custom_block
from .relaxations_base import BaseRelaxationData, ComponentWeakRef
import math
from ._utils import _get_bnds_list

logger = logging.getLogger(__name__)


@declare_custom_block(name='McCormickRelaxation')
class McCormickRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        BaseRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._yref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._tol = 0

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _y(self):
        return self._yref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def set_input(self, x, y, w, tol=0.0, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._xref.set_component(x)
        self._yref.set_component(y)
        self._wref.set_component(w)
        self._tol = tol

    def build(self, x, y, w, tol=0.0, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self.set_input(x=x, y=y, w=w, tol=tol, relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        x, y, w = self._x, self._y, self._w

        if x is None or y is None or w is None:
            raise ValueError('Trying to call McCormickRelaxation.build_relaxation, but at least one'
                             ' of the input variables required (x, y, and/or w) has not been set or'
                             ' it has been removed from the model. You must call "initialize" with valid'
                             ' input before calling "build_relaxation"')
        if self._relaxation_side is None or self._tol is None:
            raise ValueError('Trying to call McCormickRelaxation.build_relaxation, but key data'
                             ' has not been set. You must call "initialize" before "build_relaxation"')

        _build_mccormick_relaxation(b=self, x=x, y=y, w=w, relaxation_side=self._relaxation_side, tol=self._tol)

    def vars_with_bounds_in_relaxation(self):
        return [self._x, self._y]

    def _get_violation(self):
        """
        Get the absolute value of the current constraint violation.

        Returns
        -------
        violation: float
        """
        return self._w.value - self._x.value * self._y.value

    @property
    def use_linear_relaxation(self):
        return True

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        if val is not True:
            raise ValueError('McCormickRelaxation only supports relaxations.')

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} {2}*{3}'.format(self._w.name, relational_operator_string, self._x.name, self._y.name)


def _build_mccormick_relaxation(b, x, y, w, relaxation_side=RelaxationSide.BOTH, tol=0):
    """
    Construct the McCormick envelopes for w = x*y

        Underestimators:
            w >= xL*y + x*yL - xL*yL
            w >= xU*y + x*yU - xU*yU
        Overestimators:
            w <= xU*y + x*yL - xU*yL
            w <= x*yU + xL*y - xL*yU

    Parameters
    ----------
    b : Pyomo Block object
        The block that we want to use for adding any necessary constraints
    x : Pyomo Var or VarData object
        The "x" variable in x*y
    y : Pyomo Var or VarData object
        The "y" variable in x*y
    w : Pyomo Var or VarData object
        The "w" variable that is replacing x*y
    relaxation_side: RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    tol: float
        When the upper and lower bounds on x/y become too close, numerical issues can arise because both over-estimators
        become similar and both underestimators become similar. To overcome this, when the difference between the
        upper and lower bounds becomes "small enough", we only include one overestimator and one underestimator. If only
        one overestimator and only one underestimator are included, the maximum error in the relaxation is
        (xU - xL) * (yU - yL). If this product is less than tol, then only one overestimator and only one underestimator
        are included.

    Returns
    -------
    N/A
    """
    # extract the bounds on x and y
    xlb, xub = tuple(_get_bnds_list(x))
    ylb, yub = tuple(_get_bnds_list(y))

    # perform error checking on the bounds before constructing the McCormick envelope
    if xub < xlb:
        e = 'Lower bound is larger than upper bound: {0}, lb: {1}, ub: {2}'.format(x, xlb, xub)
        logger.error(e)
        raise ValueError(e)
    if yub < ylb:
        e = 'Lower bound is larger than upper bound: {0}, lb: {1}, ub: {2}'.format(y, ylb, yub)
        logger.error(e)
        raise ValueError(e)

    # check for different compbinations of fixed variables
    if x.is_fixed() and y.is_fixed():
        # x and y are fixed - don't need a McCormick, but rather an equality
        b.xy_fixed_eq = pyo.Constraint(expr= w == pyo.value(x)*pyo.value(y))
    elif x.is_fixed():
        b.x_fixed_eq = pyo.Constraint(expr= w == pyo.value(x) * y)
    elif y.is_fixed():
        b.y_fixed_eq = pyo.Constraint(expr= w == x * pyo.value(y))
    else:
        # bounds are OK and neither of the variables are fixed
        # build the standard envelopes
        b.relaxation = pyo.ConstraintList()
        assert (relaxation_side in RelaxationSide)
        if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
            if xlb != -math.inf and ylb != -math.inf:
                b.relaxation.add(w >= xlb*y + x*ylb - xlb*ylb)
            if (xub - xlb) * (yub - ylb) > tol or xlb == -math.inf or ylb == -math.inf:
                # see the doc string for this method - only adding one over and one under-estimator
                if xub != math.inf and yub != math.inf:
                    b.relaxation.add(w >= xub*y + x*yub - xub*yub)

        if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
            if xub != math.inf and ylb != -math.inf:
                b.relaxation.add(w <= xub*y + x*ylb - xub*ylb)
            if (xub - xlb) * (yub - ylb) > tol or xub == math.inf or ylb == -math.inf:
                # see the doc string for this method - only adding one over and one under-estimator
                if xlb != -math.inf and yub != math.inf:
                    b.relaxation.add(w <= x*yub + xlb*y - xlb*yub)
