import logging
import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .custom_block import declare_custom_block
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef
import math
from ._utils import var_info_str, bnds_info_str, x_pts_info_str, check_var_pts, _get_bnds_list

logger = logging.getLogger(__name__)


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
    xlb = x_pts[0]
    xub = x_pts[-1]
    ylb, yub = tuple(_get_bnds_list(y))

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
    elif xlb == -math.inf or xub == math.inf or ylb == -math.inf or yub == math.inf:
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
    A class for managing McCormick relaxations of bilinear terms (aux_var = x1 * x2).
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._x1ref = ComponentWeakRef(None)
        self._x2ref = ComponentWeakRef(None)
        self._aux_var_ref = ComponentWeakRef(None)
        self._f_x_expr = None

    @property
    def _x1(self):
        return self._x1ref.get_component()

    @property
    def _x2(self):
        return self._x2ref.get_component()

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return [self._x1, self._x2]

    def get_rhs_expr(self):
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        return [self._x1, self._x2]

    def set_input(self, x1, x2, aux_var, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        """
        Parameters
        ----------
        x1 : pyomo.core.base.var._GeneralVarData
            The "x1" variable in x1*x2
        x2 : pyomo.core.base.var._GeneralVarData
            The "x2" variable in x1*x2
        aux_var : pyomo.core.base.var._GeneralVarData
            The "aux_var" auxillary variable that is replacing x1*x2
        relaxation_side : minlp.minlp_defn.RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        """
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=True, large_eval_tol=math.inf)
        self._x1ref.set_component(x1)
        self._x2ref.set_component(x2)
        self._aux_var_ref.set_component(aux_var)
        self._partitions[self._x1] = _get_bnds_list(self._x1)
        self._f_x_expr = x1 * x2

    def build(self, x1, x2, aux_var, relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        """
        Parameters
        ----------
        x1 : pyomo.core.base.var._GeneralVarData
            The "x1" variable in x1*x2
        x2 : pyomo.core.base.var._GeneralVarData
            The "x2" variable in x1*x2
        aux_var : pyomo.core.base.var._GeneralVarData
            The "aux_var" auxillary variable that is replacing x1*x2
        relaxation_side : minlp.minlp_defn.RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        """
        self.set_input(x1=x1, x2=x2, aux_var=aux_var, relaxation_side=relaxation_side,
                       persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        _build_pw_mccormick_relaxation(b=self, x=self._x1, y=self._x2, w=self._aux_var,
                                       x_pts=self._partitions[self._x1],
                                       relaxation_side=self.relaxation_side)

    def add_parition_point(self, value=None):
        """
        This method adds one point to the partitioning of x1. If value is not
        specified, a single point will be added to the partitioning of x1 at the current value of x1. If value is
        specified, then value is added to the partitioning of x1.

        Parameters
        ----------
        value: float
            The point to be added to the partitioning of x1.
        """
        self._add_partition_point(self._x1, value)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False
