import logging
import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .custom_block import declare_custom_block
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef
import math
from ._utils import var_info_str, bnds_info_str, x_pts_info_str, check_var_pts, _get_bnds_list

logger = logging.getLogger(__name__)


def _build_mccormick_relaxation(b, x1, x2, aux_var, relaxation_side=RelaxationSide.BOTH, tol=0):
    """
    Construct the McCormick envelopes for aux_var = x1*x2

        Underestimators:
            aux_var >= x1L*x2 + x1*x2L - x1L*x2L
            aux_var >= x1U*x2 + x1*x2U - x1U*x2U
        Overestimators:
            aux_var <= x1U*x2 + x1*x2L - x1U*x2L
            aux_var <= x1*x2U + x1L*x2 - x1L*x2U

    Parameters
    ----------
    b : Pyomo Block object
        The block that we want to use for adding any necessary constraints
    x1 : Pyomo Var or VarData object
        The "x1" variable in x1*x2
    x2 : Pyomo Var or VarData object
        The "x2" variable in x1*x2
    aux_var : Pyomo Var or VarData object
        The "aux_var" variable that is replacing x1*x2
    relaxation_side: RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    tol: float
        When the upper and lower bounds on x1/x2 become too close, numerical issues can arise because both over-estimators
        become similar and both underestimators become similar. To overcome this, when the difference between the
        upper and lower bounds becomes "small enough", we only include one overestimator and one underestimator. If only
        one overestimator and only one underestimator are included, the maximum error in the relaxation is
        (x1U - x1L) * (x2U - x2L). If this product is less than tol, then only one overestimator and only one underestimator
        are included.

    Returns
    -------
    N/A
    """
    # extract the bounds on x1 and x2
    x1lb, x1ub = tuple(_get_bnds_list(x1))
    x2lb, x2ub = tuple(_get_bnds_list(x2))

    # perform error checking on the bounds before constructing the McCormick envelope
    if x1ub < x1lb:
        e = 'Lower bound is larger than upper bound: {0}, lb: {1}, ub: {2}'.format(x1, x1lb, x1ub)
        logger.error(e)
        raise ValueError(e)
    if x2ub < x2lb:
        e = 'Lower bound is larger than upper bound: {0}, lb: {1}, ub: {2}'.format(x2, x2lb, x2ub)
        logger.error(e)
        raise ValueError(e)

    # check for different compbinations of fixed variables
    if x1.is_fixed() and x2.is_fixed():
        # x1 and x2 are fixed - don't need a McCormick, but rather an equality
        b.x1x2_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1)*pyo.value(x2))
    elif x1.is_fixed():
        b.x1_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1) * x2)
    elif x2.is_fixed():
        b.x2_fixed_eq = pyo.Constraint(expr= aux_var == x1 * pyo.value(x2))
    else:
        # bounds are OK and neither of the variables are fixed
        # build the standard envelopes
        b.relaxation = pyo.ConstraintList()
        assert (relaxation_side in RelaxationSide)
        if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
            if x1lb != -math.inf and x2lb != -math.inf:
                b.relaxation.add(aux_var >= x1lb*x2 + x1*x2lb - x1lb*x2lb)
            if (x1ub - x1lb) * (x2ub - x2lb) > tol or x1lb == -math.inf or x2lb == -math.inf:
                # see the doc string for this method - only adding one over and one under-estimator
                if x1ub != math.inf and x2ub != math.inf:
                    b.relaxation.add(aux_var >= x1ub*x2 + x1*x2ub - x1ub*x2ub)

        if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
            if x1ub != math.inf and x2lb != -math.inf:
                b.relaxation.add(aux_var <= x1ub*x2 + x1*x2lb - x1ub*x2lb)
            if (x1ub - x1lb) * (x2ub - x2lb) > tol or x1ub == math.inf or x2lb == -math.inf:
                # see the doc string for this method - only adding one over and one under-estimator
                if x1lb != -math.inf and x2ub != math.inf:
                    b.relaxation.add(aux_var <= x1*x2ub + x1lb*x2 - x1lb*x2ub)


def _build_pw_mccormick_relaxation(b, x1, x2, aux_var, x1_pts, relaxation_side=RelaxationSide.BOTH):
    """
    This function creates piecewise envelopes to relax "aux_var = x1*x2". Note that the partitioning is done on "x1" only.
    This is the "nf4r" from Gounaris, Misener, and Floudas (2009).

    Parameters
    ----------
    b: pyo.ConcreteModel or pyo.Block
    x1: pyomo.core.base.var._GeneralVarData
        The "x1" variable in x1*x2
    x2: pyomo.core.base.var._GeneralVarData
        The "x2" variable in x1*x2
    aux_var: pyomo.core.base.var._GeneralVarData
        The "aux_var" variable that is replacing x*y
    x1_pts: list of floats
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the
        last point (x_pts[-1]) is equal to x.ub
    relaxation_side : minlp.minlp_defn.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    """
    x1_lb = x1_pts[0]
    x1_ub = x1_pts[-1]
    x2_lb, x2_ub = tuple(_get_bnds_list(x2))

    check_var_pts(x1, x_pts=x1_pts)
    check_var_pts(x2)

    if x1.is_fixed() and x2.is_fixed():
        b.x1_x2_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1) * pyo.value(x2))
    elif x1.is_fixed():
        b.x1_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1) * x2)
    elif x2.is_fixed():
        b.x2_fixed_eq = pyo.Constraint(expr= aux_var == x1 * pyo.value(x2))
    elif len(x1_pts) == 2:
        _build_mccormick_relaxation(b, x1=x1, x2=x2, aux_var=aux_var, relaxation_side=relaxation_side)
    elif x1_lb == -math.inf or x1_ub == math.inf or x2_lb == -math.inf or x2_ub == math.inf:
        _build_mccormick_relaxation(b, x1=x1, x2=x2, aux_var=aux_var, relaxation_side=relaxation_side)
    else:
        # create the lambda_ variables (binaries for the pw representation)
        b.interval_set = pyo.Set(initialize=range(1, len(x1_pts)))
        b.lambda_ = pyo.Var(b.interval_set, within=pyo.Binary)

        # create the delta x2 variables
        b.delta_x2 = pyo.Var(b.interval_set, bounds=(0, None))

        # create the "sos1" constraint
        b.lambda_sos1 = pyo.Constraint(expr=sum(b.lambda_[n] for n in b.interval_set) == 1.0)

        # create the x1 interval constraints
        b.x1_interval_lb = pyo.Constraint(expr=sum(x1_pts[n - 1] * b.lambda_[n] for n in b.interval_set) <= x1)
        b.x1_interval_ub = pyo.Constraint(expr=x1 <= sum(x1_pts[n] * b.lambda_[n] for n in b.interval_set))

        # create the x2 constraints
        b.x2_con = pyo.Constraint(expr=x2 == x2_lb + sum(b.delta_x2[n] for n in b.interval_set))

        def delta_x2n_ub_rule(m, n):
            return b.delta_x2[n] <= (x2_ub - x2_lb) * b.lambda_[n]

        b.delta_x2n_ub = pyo.Constraint(b.interval_set, rule=delta_x2n_ub_rule)

        # create the relaxation constraints
        if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
            b.aux_var_lb1 = pyo.Constraint(expr=(aux_var >= x2_ub * x1 + sum(x1_pts[n] * b.delta_x2[n] for n in b.interval_set) -
                                                 (x2_ub - x2_lb) * sum(x1_pts[n] * b.lambda_[n] for n in b.interval_set)))
            b.aux_var_lb2 = pyo.Constraint(expr=aux_var >= x2_lb * x1 + sum(x1_pts[n - 1] * b.delta_x2[n] for n in b.interval_set))

        if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
            b.aux_var_ub1 = pyo.Constraint(expr=(aux_var <= x2_ub * x1 + sum(x1_pts[n - 1] * b.delta_x2[n] for n in b.interval_set) -
                                                 (x2_ub - x2_lb) * sum(x1_pts[n - 1] * b.lambda_[n] for n in b.interval_set)))
            b.aux_var_ub2 = pyo.Constraint(expr=aux_var <= x2_lb * x1 + sum(x1_pts[n] * b.delta_x2[n] for n in b.interval_set))


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
        _build_pw_mccormick_relaxation(b=self, x1=self._x1, x2=self._x2, aux_var=self._aux_var,
                                       x1_pts=self._partitions[self._x1],
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
