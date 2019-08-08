import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef
import warnings
from .custom_block import declare_custom_block
import numpy as np
import math
import scipy.optimize
from ._utils import var_info_str, bnds_info_str, x_pts_info_str, check_var_pts, _get_bnds_list, _copy_v_pts_without_inf
from pyomo.opt import SolverStatus, TerminationCondition
import logging
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd, reverse_ad
logger = logging.getLogger(__name__)
pe = pyo


def _sin_overestimator_fn(x, LB):
    return np.sin(x) + np.cos(x) * (LB - x) - np.sin(LB)


def _sin_underestimator_fn(x, UB):
    return np.sin(x) + np.cos(-x) * (UB - x) - np.sin(UB)


def _compute_sine_overestimator_tangent_point(vlb):
    assert vlb < 0
    tangent_point, res = scipy.optimize.bisect(f=_sin_overestimator_fn, a=0, b=math.pi / 2, args=(vlb,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = float(np.cos(tangent_point))
        intercept = float(np.sin(vlb) - slope * vlb)
        return tangent_point, slope, intercept
    else:
        e = 'Unable to build relaxation for sin(x)\nBisect info: ' + str(res)
        logger.error(e)
        raise RuntimeError(e)


def _compute_sine_underestimator_tangent_point(vub):
    assert vub > 0
    tangent_point, res = scipy.optimize.bisect(f=_sin_underestimator_fn, a=-math.pi / 2, b=0, args=(vub,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = float(np.cos(-tangent_point))
        intercept = float(np.sin(vub) - slope * vub)
        return tangent_point, slope, intercept
    else:
        e = ('Unable to build relaxation for sin(x)\nBisect info: ' + str(res))
        logger.error(e)
        raise RuntimeError(e)


def _atan_overestimator_fn(x, LB):
    return (1 + x**2) * (np.arctan(x) - np.arctan(LB)) + x - LB


def _atan_underestimator_fn(x, UB):
    return (1 + x**2) * (np.arctan(x) - np.arctan(UB)) + x - UB


def _compute_arctan_overestimator_tangent_point(vlb):
    assert vlb < 0
    tangent_point, res = scipy.optimize.bisect(f=_atan_overestimator_fn, a=0, b=abs(vlb), args=(vlb,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = 1/(1 + tangent_point**2)
        intercept = float(np.arctan(vlb) - slope * vlb)
        return tangent_point, slope, intercept
    else:
        e = 'Unable to build relaxation for arctan(x)\nBisect info: ' + str(res)
        logger.error(e)
        raise RuntimeError(e)


def _compute_arctan_underestimator_tangent_point(vub):
    assert vub > 0
    tangent_point, res = scipy.optimize.bisect(f=_atan_underestimator_fn, a=-vub, b=0, args=(vub,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = 1/(1 + tangent_point**2)
        intercept = float(np.arctan(vub) - slope * vub)
        return tangent_point, slope, intercept
    else:
        e = 'Unable to build relaxation for arctan(x)\nBisect info: ' + str(res)
        logger.error(e)
        raise RuntimeError(e)


class _FxExpr(object):
    def __init__(self, expr, x):
        self._expr = expr
        self._x = x
        self._deriv = reverse_sd(expr)[x]

    def eval(self, _xval):
        _xval = pyo.value(_xval)
        orig_xval = self._x.value
        self._x.value = _xval
        res = pyo.value(self._expr)
        self._x.value = orig_xval
        return res

    def deriv(self, _xval):
        _xval = pyo.value(_xval)
        orig_xval = self._x.value
        self._x.value = _xval
        res = pyo.value(self._deriv)
        self._x.value = orig_xval
        return res

    def __call__(self, _xval):
        return self.eval(_xval)


def _func_wrapper(obj):
    def _func(m, val):
        return obj(val)
    return _func


def pw_univariate_relaxation(b, x, w, x_pts, f_x_expr, pw_repn='INC', shape=FunctionShape.UNKNOWN,
                             relaxation_side=RelaxationSide.BOTH, large_eval_tol=1e8):
    """
    This function creates piecewise envelopes to relax "w=f(x)" where f(x) is univariate and either convex over the
    entire domain of x or concave over the entire domain of x.

    Parameters
    ----------
    b: pyo.Block
    x: pyo.Var
        The "x" variable in f(x)
    w: pyo.Var
        The "w" variable that is replacing f(x)
    x_pts: list of floats
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the last
        point (x_pts[-1]) is equal to x.ub
    f_x_expr: pyomo expression
        An expression for f(x)
    pw_repn: str
        This must be one of the valid strings for the peicewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    shape: FunctionShape
        Specify the shape of the function. Valid values are minlp.FunctionShape.CONVEX or minlp.FunctionShape.CONCAVE
    relaxation_side: RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    large_eval_tol: float
        To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol, 
        at a point in x_pts, then that point is skipped.
    """
    _eval = _FxExpr(expr=f_x_expr, x=x)
    xlb = x_pts[0]
    xub = x_pts[-1]

    check_var_pts(x, x_pts)

    if shape not in {FunctionShape.CONCAVE, FunctionShape.CONVEX}:
        e = 'pw_univariate_relaxation: shape must be either FunctionShape.CONCAVE or FunctionShape.CONVEX'
        logger.error(e)
        raise ValueError(e)

    if x.is_fixed():
        b.x_fixed_con = pyo.Constraint(expr=w == _eval(x.value))
    elif xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == _eval(x.lb))
    else:
        # Do the non-convex piecewise portion if shape=CONCAVE and relaxation_side=Under/BOTH
        # or if shape=CONVEX and relaxation_side=Over/BOTH

        pw_constr_type = None
        if shape == FunctionShape.CONVEX and relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            pw_constr_type = 'UB'
        if shape == FunctionShape.CONCAVE and relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            pw_constr_type = 'LB'

        if pw_constr_type is not None:
            # Build the piecewise side of the envelope
            if x_pts[0] > -math.inf and x_pts[-1] < math.inf:
                can_evaluate_func_at_all_pts = True  # this is primarily for things like log(x) where x.lb = 0
                tmp_pts = list()
                for _pt in x_pts:
                    try:
                        f = _eval(_pt)
                        if f < -large_eval_tol:
                            logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is less than -1e8'.format(str(_pt), str(x), str(f_x_expr)))
                            continue
                        if f > large_eval_tol:
                            logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is greater than 1e8'.format(str(_pt), str(x), str(f_x_expr)))
                            continue
                        tmp_pts.append(_pt)
                    except (ZeroDivisionError, ValueError):
                        pass
                if len(tmp_pts) >= 2 and tmp_pts[0] == x_pts[0] and tmp_pts[-1] == x_pts[-1]:
                    b.pw_linear_under_over = pyo.Piecewise(w, x,
                                                           pw_pts=tmp_pts,
                                                           pw_repn=pw_repn,
                                                           pw_constr_type=pw_constr_type,
                                                           f_rule=_func_wrapper(_eval)
                                                           )

        non_pw_constr_type = None
        if shape == FunctionShape.CONVEX and relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            non_pw_constr_type = 'LB'
        if shape == FunctionShape.CONCAVE and relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            non_pw_constr_type = 'UB'

        x_pts = _copy_v_pts_without_inf(x_pts)

        if non_pw_constr_type is not None:
            # Build the non-piecewise side of the envelope
            b.linear_under_over = pyo.ConstraintList()
            for _x in x_pts:
                try:
                    w_at_pt = _eval(_x)
                    m_at_pt = _eval.deriv(_x)
                    if w_at_pt < -large_eval_tol:
                        logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is less than -1e8'.format(str(_x), str(x), str(f_x_expr)))
                        continue
                    if w_at_pt > large_eval_tol:
                        logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is greater than 1e8'.format(str(_x), str(x), str(f_x_expr)))
                        continue
                    if m_at_pt < -large_eval_tol:
                        logger.warning('Skipping pt {0} for var {1} because the derivative of {2} evaluated at {0} is less than -1e8'.format(str(_x), str(x), str(f_x_expr)))
                        continue
                    if m_at_pt > large_eval_tol:
                        logger.warning('Skipping pt {0} for var {1} because the derivative of {2} evaluated at {0} is greater than 1e8'.format(str(_x), str(x), str(f_x_expr)))
                        continue
                    b_at_pt = w_at_pt - m_at_pt * _x
                    if non_pw_constr_type == 'LB':
                        b.linear_under_over.add(w >= m_at_pt * x + b_at_pt)
                    else:
                        assert non_pw_constr_type == 'UB'
                        b.linear_under_over.add(w <= m_at_pt * x + b_at_pt)
                except (ZeroDivisionError, ValueError):
                    pass


def pw_x_squared_relaxation(b, x, w, x_pts, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                            use_nonlinear_underestimator=False):
    """
    This function creates piecewise envelopes that provide a linear relaxation of "w=x**2".

    Parameters
    ----------
    b: pyo.Block
    x: pyo.Var
        The "x" variable in x**2
    w: pyo.Var
        The "w" variable that is replacing x**2
    x_pts: list of float
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the last
        point (x_pts[-1]) is equal to x.ub
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the
        Piecewise component). Use help(Piecewise) to learn more.
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    use_nonlinear_underestimator: bool
        If False, then piecewise linear underestimators will be built.
        If True, then the nonlinear underestimators will be built ( w >= x**2 )
    """
    # Need to consider the following situations
    #   side    use_nonlinear      pw_under    pw_over     nonlinear_under  side_for_pw     use_nonlin
    #   OVER        False           no          yes         no                  OVER            False
    #   OVER        True            no          no          no              (EXCEPTION)     (EXCEPTION)
    #   UNDER       False           yes         no          no                  UNDER           False
    #   UNDER       True            no          no          yes                 None            True
    #   BOTH        False           yes         yes         no                  BOTH            False
    #   BOTH        True            no          yes         yes                 OVER            True

    # exception for OVER/True
    # change UNDER/True to None/True
    # change BOTH/True  to OVER/True

    check_var_pts(x, x_pts)

    if use_nonlinear_underestimator and (relaxation_side == RelaxationSide.OVER):
        e = 'pw_x_squared_relaxation: if use_nonlinear_underestimator is True, then ' + \
                         'relaxation_side needs to be FunctionShape.UNDER or FunctionShape.BOTH'
        logger.error(e)
        raise ValueError(e)

    if x.is_fixed():
        b.x_fixed_con = pyo.Constraint(expr= w == pyo.value(x)**2)
    else:
        pw_side = relaxation_side
        if pw_side == RelaxationSide.UNDER and use_nonlinear_underestimator is True:
            pw_side = None
        if pw_side == RelaxationSide.BOTH and use_nonlinear_underestimator is True:
            pw_side = RelaxationSide.OVER

        if pw_side is not None:
            b.pw_under_over = pyo.Block()
            pw_univariate_relaxation(b.pw_under_over, x, w, x_pts, f_x_expr=x**2, pw_repn=pw_repn, shape=FunctionShape.CONVEX, relaxation_side=pw_side)

        if use_nonlinear_underestimator:
            b.underestimator = pyo.Constraint(expr= w >= x**2)


def pw_cos_relaxation(b, x, w, x_pts, relaxation_side=RelaxationSide.BOTH, pw_repn='INC',
                      use_quadratic_overestimator=False):
    """
    This function creates a block with the constraints necessary to relax w = cos(x)
    for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    b: pyo.Block
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in cos(x). The lower bound on x must greater than or equal to
        -pi/2 and the upper bound on x must be less than or equal to pi/2.
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing cos(x)
    x_pts: list of float
        A list of floating point numbers to define the points over which the piecewise
        representation will be generated. This list must be ordered, and it is expected
        that the first point (x_pts[0]) is equal to x.lb and the last point (x_pts[-1])
        is equal to x.ub
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    pw_repn: str
        This must be one of the valid strings for the peicewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    use_quadratic_overestimator: bool
        If False, then linear overestimators will be built. If True, then a
        quadratic overestimator will be used. Note that a piecewise version of the
        quadratic overestimator is not supported.
    """
    _eval = _FxExpr(expr=pyo.cos(x), x=x)

    check_var_pts(x, x_pts)

    xlb = x_pts[0]
    xub = x_pts[-1]

    if x.is_fixed():
        b.x_fixed_con = pyo.Constraint(expr=w == _eval(x.value))
        return

    if xlb < -np.pi / 2.0:
        return

    if xub > np.pi / 2.0:
        return

    if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
        if use_quadratic_overestimator:
            ub = max([abs(xlb), abs(xub)])
            b.overestimator = pyo.Constraint(expr=w <= 1 - ((1-_eval(ub))/ub**2)*x**2)
        else:
            b.overestimator = pyo.Block()
            pw_univariate_relaxation(b=b.overestimator, x=x, w=w, x_pts=x_pts, f_x_expr=pyo.cos(x),
                                     shape=FunctionShape.CONCAVE, pw_repn=pw_repn,
                                     relaxation_side=RelaxationSide.OVER)

    if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
        b.underestimator = pyo.Block()
        pw_univariate_relaxation(b=b.underestimator, x=x, w=w, x_pts=x_pts, f_x_expr=pyo.cos(x),
                                 shape=FunctionShape.CONCAVE, pw_repn=pw_repn,
                                 relaxation_side=RelaxationSide.UNDER)


def pw_sin_relaxation(b, x, w, x_pts, relaxation_side=RelaxationSide.BOTH, pw_repn='INC', safety_tol=1e-10):
    """
    This function creates piecewise relaxations to relax "w=sin(x)" for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    b: pyo.Block
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in sin(x). The lower bound on x must greater than or equal to
        -pi/2 and the upper bound on x must be less than or equal to pi/2.
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing sin(x)
    x_pts: list of float
        A list of floating point numbers to define the points over which the piecewise
        representation will be generated. This list must be ordered, and it is expected
        that the first point (x_pts[0]) is equal to x.lb and the last point (x_pts[-1])
        is equal to x.ub
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    pw_repn: str
        This must be one of the valid strings for the peicewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    safety_tol: float
        amount to lift the overestimator or drop the underestimator. This is used to ensure none of the feasible
        region is cut off by error in computing the over and under estimators.
    """
    check_var_pts(x, x_pts)
    expr = pyo.sin(x)

    xlb = x_pts[0]
    xub = x_pts[-1]

    if x.is_fixed() or xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == (pyo.value(expr)))
        return

    if xlb < -np.pi / 2.0:
        return

    if xub > np.pi / 2.0:
        return

    if x_pts[0] >= 0:
        pw_univariate_relaxation(b=b, x=x, w=w, x_pts=x_pts, f_x_expr=expr,
                                 shape=FunctionShape.CONCAVE, relaxation_side=relaxation_side, pw_repn=pw_repn)
        return
    if x_pts[-1] <= 0:
        pw_univariate_relaxation(b=b, x=x, w=w, x_pts=x_pts, f_x_expr=expr,
                                 shape=FunctionShape.CONVEX, relaxation_side=relaxation_side, pw_repn=pw_repn)
        return

    OE_tangent_x, OE_tangent_slope, OE_tangent_intercept = _compute_sine_overestimator_tangent_point(xlb)
    UE_tangent_x, UE_tangent_slope, UE_tangent_intercept = _compute_sine_underestimator_tangent_point(xub)
    non_piecewise_overestimators_pts = []
    non_piecewise_underestimator_pts = []

    if relaxation_side == RelaxationSide.OVER:
        if OE_tangent_x < xub:
            new_x_pts = [i for i in x_pts if i < OE_tangent_x]
            new_x_pts.append(xub)
            non_piecewise_overestimators_pts = [OE_tangent_x]
            non_piecewise_overestimators_pts.extend(i for i in x_pts if i > OE_tangent_x)
            x_pts = new_x_pts
    elif relaxation_side == RelaxationSide.UNDER:
        if UE_tangent_x > xlb:
            new_x_pts = [xlb]
            new_x_pts.extend(i for i in x_pts if i > UE_tangent_x)
            non_piecewise_underestimator_pts = [i for i in x_pts if i < UE_tangent_x]
            non_piecewise_underestimator_pts.append(UE_tangent_x)
            x_pts = new_x_pts

    b.non_piecewise_overestimators = pyo.ConstraintList()
    b.non_piecewise_underestimators = pyo.ConstraintList()
    for pt in non_piecewise_overestimators_pts:
        b.non_piecewise_overestimators.add(w <= math.sin(pt) + safety_tol + (x - pt) * math.cos(pt))
    for pt in non_piecewise_underestimator_pts:
        b.non_piecewise_underestimators.add(w >= math.sin(pt) - safety_tol + (x - pt) * math.cos(pt))

    intervals = []
    for i in range(len(x_pts)-1):
        intervals.append((x_pts[i], x_pts[i+1]))

    b.interval_set = pyo.Set(initialize=range(len(intervals)), ordered=True)
    b.x = pyo.Var(b.interval_set)
    b.w = pyo.Var(b.interval_set)
    if len(intervals) == 1:
        b.lam = pyo.Param(b.interval_set, mutable=True)
        b.lam[0].value = 1.0
    else:
        b.lam = pyo.Var(b.interval_set, within=pyo.Binary)
    b.x_lb = pyo.ConstraintList()
    b.x_ub = pyo.ConstraintList()
    b.x_sum = pyo.Constraint(expr=x == sum(b.x[i] for i in b.interval_set))
    b.w_sum = pyo.Constraint(expr=w == sum(b.w[i] for i in b.interval_set))
    b.lam_sum = pyo.Constraint(expr=sum(b.lam[i] for i in b.interval_set) == 1)
    b.overestimators = pyo.ConstraintList()
    b.underestimators = pyo.ConstraintList()

    for i, tup in enumerate(intervals):
        x0 = tup[0]
        x1 = tup[1]

        b.x_lb.add(x0 * b.lam[i] <= b.x[i])
        b.x_ub.add(b.x[i] <= x1 * b.lam[i])

        # Overestimators
        if relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            if x0 < 0 and x1 <= 0:
                slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                intercept = math.sin(x0) - slope * x0
                b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            elif (x0 < 0) and (x1 > 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_sine_overestimator_tangent_point(x0)
                if tangent_x <= x1:
                    b.overestimators.add(b.w[i] <= tangent_slope * b.x[i] + (tangent_intercept + safety_tol) * b.lam[i])
                    b.overestimators.add(b.w[i] <= math.cos(x1) * b.x[i] + (math.sin(x1) - x1 * math.cos(x1) + safety_tol) * b.lam[i])
                else:
                    slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                    intercept = math.sin(x0) - slope * x0
                    b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            else:
                b.overestimators.add(b.w[i] <= math.cos(x0) * b.x[i] + (math.sin(x0) - x0 * math.cos(x0) + safety_tol) * b.lam[i])
                b.overestimators.add(b.w[i] <= math.cos(x1) * b.x[i] + (math.sin(x1) - x1 * math.cos(x1) + safety_tol) * b.lam[i])

        # Underestimators
        if relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            if x0 >= 0 and x1 > 0:
                slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                intercept = math.sin(x0) - slope * x0
                b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            elif (x1 > 0) and (x0 < 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_sine_underestimator_tangent_point(x1)
                if tangent_x >= x0:
                    b.underestimators.add(b.w[i] >= tangent_slope * b.x[i] + (tangent_intercept - safety_tol) * b.lam[i])
                    b.underestimators.add(b.w[i] >= math.cos(x0) * b.x[i] + (math.sin(x0) - x0 * math.cos(x0) - safety_tol) * b.lam[i])
                else:
                    slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                    intercept = math.sin(x0) - slope * x0
                    b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            else:
                b.underestimators.add(b.w[i] >= math.cos(x0) * b.x[i] + (math.sin(x0) - x0 * math.cos(x0) - safety_tol) * b.lam[i])
                b.underestimators.add(b.w[i] >= math.cos(x1) * b.x[i] + (math.sin(x1) - x1 * math.cos(x1) - safety_tol) * b.lam[i])

    return x_pts


def pw_arctan_relaxation(b, x, w, x_pts, relaxation_side=RelaxationSide.BOTH, pw_repn='INC', safety_tol=1e-10):
    """
    This function creates piecewise relaxations to relax "w=sin(x)" for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    b: pyo.Block
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in sin(x). The lower bound on x must greater than or equal to
        -pi/2 and the upper bound on x must be less than or equal to pi/2.
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing sin(x)
    x_pts: list of float
        A list of floating point numbers to define the points over which the piecewise
        representation will be generated. This list must be ordered, and it is expected
        that the first point (x_pts[0]) is equal to x.lb and the last point (x_pts[-1])
        is equal to x.ub
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    pw_repn: str
        This must be one of the valid strings for the peicewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    safety_tol: float
        amount to lift the overestimator or drop the underestimator. This is used to ensure none of the feasible
        region is cut off by error in computing the over and under estimators.
    """
    check_var_pts(x, x_pts)
    expr = pyo.atan(x)
    _eval = _FxExpr(expr, x)

    xlb = x_pts[0]
    xub = x_pts[-1]

    if x.is_fixed() or xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == pyo.value(expr))
        return

    if x_pts[0] >= 0:
        pw_univariate_relaxation(b=b, x=x, w=w, x_pts=x_pts, f_x_expr=expr,
                                 shape=FunctionShape.CONCAVE, relaxation_side=relaxation_side, pw_repn=pw_repn)
        return
    if x_pts[-1] <= 0:
        pw_univariate_relaxation(b=b, x=x, w=w, x_pts=x_pts, f_x_expr=expr,
                                 shape=FunctionShape.CONVEX, relaxation_side=relaxation_side, pw_repn=pw_repn)
        return

    if xlb == -math.inf or xub == math.inf:
        return

    OE_tangent_x, OE_tangent_slope, OE_tangent_intercept = _compute_arctan_overestimator_tangent_point(xlb)
    UE_tangent_x, UE_tangent_slope, UE_tangent_intercept = _compute_arctan_underestimator_tangent_point(xub)
    non_piecewise_overestimators_pts = []
    non_piecewise_underestimator_pts = []

    if relaxation_side == RelaxationSide.OVER:
        if OE_tangent_x < xub:
            new_x_pts = [i for i in x_pts if i < OE_tangent_x]
            new_x_pts.append(xub)
            non_piecewise_overestimators_pts = [OE_tangent_x]
            non_piecewise_overestimators_pts.extend(i for i in x_pts if i > OE_tangent_x)
            x_pts = new_x_pts
    elif relaxation_side == RelaxationSide.UNDER:
        if UE_tangent_x > xlb:
            new_x_pts = [xlb]
            new_x_pts.extend(i for i in x_pts if i > UE_tangent_x)
            non_piecewise_underestimator_pts = [i for i in x_pts if i < UE_tangent_x]
            non_piecewise_underestimator_pts.append(UE_tangent_x)
            x_pts = new_x_pts

    b.non_piecewise_overestimators = pyo.ConstraintList()
    b.non_piecewise_underestimators = pyo.ConstraintList()
    for pt in non_piecewise_overestimators_pts:
        b.non_piecewise_overestimators.add(w <= math.atan(pt) + safety_tol + (x - pt) * _eval.deriv(pt))
    for pt in non_piecewise_underestimator_pts:
        b.non_piecewise_underestimators.add(w >= math.atan(pt) - safety_tol + (x - pt) * _eval.deriv(pt))

    intervals = []
    for i in range(len(x_pts)-1):
        intervals.append((x_pts[i], x_pts[i+1]))

    b.interval_set = pyo.Set(initialize=range(len(intervals)))
    b.x = pyo.Var(b.interval_set)
    b.w = pyo.Var(b.interval_set)
    if len(intervals) == 1:
        b.lam = pyo.Param(b.interval_set, mutable=True)
        b.lam[0].value = 1.0
    else:
        b.lam = pyo.Var(b.interval_set, within=pyo.Binary)
    b.x_lb = pyo.ConstraintList()
    b.x_ub = pyo.ConstraintList()
    b.x_sum = pyo.Constraint(expr=x == sum(b.x[i] for i in b.interval_set))
    b.w_sum = pyo.Constraint(expr=w == sum(b.w[i] for i in b.interval_set))
    b.lam_sum = pyo.Constraint(expr=sum(b.lam[i] for i in b.interval_set) == 1)
    b.overestimators = pyo.ConstraintList()
    b.underestimators = pyo.ConstraintList()

    for i, tup in enumerate(intervals):
        x0 = tup[0]
        x1 = tup[1]

        b.x_lb.add(x0 * b.lam[i] <= b.x[i])
        b.x_ub.add(b.x[i] <= x1 * b.lam[i])

        # Overestimators
        if relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            if x0 < 0 and x1 <= 0:
                slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                intercept = math.atan(x0) - slope * x0
                b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            elif (x0 < 0) and (x1 > 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_arctan_overestimator_tangent_point(x0)
                if tangent_x <= x1:
                    b.overestimators.add(b.w[i] <= tangent_slope * b.x[i] + (tangent_intercept + safety_tol) * b.lam[i])
                    b.overestimators.add(b.w[i] <= _eval.deriv(x1) * b.x[i] + (math.atan(x1) - x1 * _eval.deriv(x1) + safety_tol) * b.lam[i])
                else:
                    slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                    intercept = math.atan(x0) - slope * x0
                    b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            else:
                b.overestimators.add(b.w[i] <= _eval.deriv(x0) * b.x[i] + (math.atan(x0) - x0 * _eval.deriv(x0) + safety_tol) * b.lam[i])
                b.overestimators.add(b.w[i] <= _eval.deriv(x1) * b.x[i] + (math.atan(x1) - x1 * _eval.deriv(x1) + safety_tol) * b.lam[i])

        # Underestimators
        if relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            if x0 >= 0 and x1 > 0:
                slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                intercept = math.atan(x0) - slope * x0
                b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            elif (x1 > 0) and (x0 < 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_arctan_underestimator_tangent_point(x1)
                if tangent_x >= x0:
                    b.underestimators.add(b.w[i] >= tangent_slope * b.x[i] + (tangent_intercept - safety_tol) * b.lam[i])
                    b.underestimators.add(b.w[i] >= _eval.deriv(x0) * b.x[i] + (math.atan(x0) - x0 * _eval.deriv(x0) - safety_tol) * b.lam[i])
                else:
                    slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                    intercept = math.atan(x0) - slope * x0
                    b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            else:
                b.underestimators.add(b.w[i] >= _eval.deriv(x0) * b.x[i] + (math.atan(x0) - x0 * _eval.deriv(x0) - safety_tol) * b.lam[i])
                b.underestimators.add(b.w[i] >= _eval.deriv(x1) * b.x[i] + (math.atan(x1) - x1 * _eval.deriv(x1) - safety_tol) * b.lam[i])

    return x_pts


@declare_custom_block(name='PWXSquaredRelaxation')
class PWXSquaredRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = x**2.

    Parameters
    ----------
    x: pyomo.core.base.var._GeneralVarData
        The "x" variable in w=x**2.
    w: pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing x**2
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    use_linear_relaxation: bool
        If True, then linear underestimators will be built.
        If False, then  quadratic underestimators will be built ( w >= x**2 )
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._pw_repn = 'INC'
        self._use_linear_relaxation = True

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        v = []
        if self._relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            v.append(self._x)
        return v

    def set_input(self, x, w, pw_repn='INC', use_linear_relaxation=True, relaxation_side=RelaxationSide.BOTH,
                  persistent_solvers=None):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._xref.set_component(x)
        self._wref.set_component(w)
        self._pw_repn = pw_repn
        self.use_linear_relaxation = use_linear_relaxation
        self._partitions[self._x] = _get_bnds_list(self._x)

    def build(self, x, w, pw_repn='INC', use_linear_relaxation=True, relaxation_side=RelaxationSide.BOTH,
              persistent_solvers=None):
        self.set_input(x=x, w=w, pw_repn=pw_repn, use_linear_relaxation=use_linear_relaxation,
                       relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        pw_x_squared_relaxation(self, x=self._x, w=self._w, x_pts=self._partitions[self._x],
                                pw_repn=self._pw_repn, relaxation_side=self._relaxation_side,
                                use_nonlinear_underestimator=(not self._use_linear_relaxation))

    def _get_cut_expr(self):
        """
        Add a linear cut on the convex side of the constraint based on the current
        values of the variables. There is no need to call rebuild. This
        method directly adds a constraint to the block. A new point will NOT be added
        to the partitioning! This method does not change the partitioning!
        The current relaxation is not discarded and rebuilt. A constraint is simply added.
        """
        expr = None
        viol = self.get_violation()
        if viol >= 0:
            e = 'Cannot add cut; constraint is violated in the wrong direction; no constraint will be added.'
            warnings.warn(e)
            logger.warning(e)
        else:
            xval = pyo.value(self._x)
            expr = self._w >= 2*xval*self._x - xval**2

        return expr

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
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._w.value - self._x.value ** 2

    def is_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return True

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
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        self._use_linear_relaxation = val

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} {2}**2'.format(self._w.name, relational_operator_string, self._x.name)


@declare_custom_block(name='PWUnivariateRelaxation')
class PWUnivariateRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = f(x) where f(x) is either convex
    or concave.

    Parameters
    ----------
    x: pyomo.core.base.var._GeneralVarData
        The "x" variable in w=f(x).
    w: pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing f(x)
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    relaxation_side: RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    shape: FunctionShape
        Options are FunctionShape.CONVEX and FunctionShape.CONCAVE
    f_x_expr: pyomo expression
        The pyomo expression representing f(x)
    large_eval_tol: float
        To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol, 
        at a point in x_pts, then that point is skipped.
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._pw_repn = 'INC'
        self._function_shape = FunctionShape.UNKNOWN
        self._f_x_expr = None
        self.large_eval_tol = None

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        v = []
        if self._relaxation_side is RelaxationSide.BOTH:
            v.append(self._x)
        elif self._relaxation_side is RelaxationSide.UNDER and self._function_shape is FunctionShape.CONCAVE:
            v.append(self._x)
        elif self._relaxation_side is RelaxationSide.OVER and self._function_shape is FunctionShape.CONVEX:
            v.append(self._x)
        return v

    def set_input(self, x, w, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  persistent_solvers=None, large_eval_tol=1e8):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._pw_repn = pw_repn
        self._function_shape = shape
        self._f_x_expr = f_x_expr
        self.large_eval_tol = large_eval_tol

        self._xref.set_component(x)
        self._wref.set_component(w)
        self._partitions[self._x] = _get_bnds_list(self._x)

    def build(self, x, w, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              persistent_solvers=None, large_eval_tol=1e8):
        self.set_input(x=x, w=w, shape=shape, f_x_expr=f_x_expr, pw_repn=pw_repn, relaxation_side=relaxation_side,
                       persistent_solvers=persistent_solvers, large_eval_tol=large_eval_tol)
        self.rebuild()

    def _build_relaxation(self):
        pw_univariate_relaxation(b=self, x=self._x, w=self._w, x_pts=self._partitions[self._x], f_x_expr=self._f_x_expr,
                                 pw_repn=self._pw_repn, shape=self._function_shape,
                                 relaxation_side=self._relaxation_side, large_eval_tol=self.large_eval_tol)

    def _get_cut_expr(self):
        """
        Add a linear cut on the convex side of the constraint based on the current
        values of the variables. There is no need to call rebuild. This
        method directly adds a constraint to the block. A new point will NOT be added
        to the partitioning! This method does not change the partitioning!
        The current relaxation is not discarded and rebuilt. A constraint is simply added.
        """
        expr = None
        viol = self.get_violation()
        if ((viol > 0 and self._function_shape == FunctionShape.CONVEX) or
                (viol < 0 and self._function_shape == FunctionShape.CONCAVE)):
            e = 'Cannot add cut; constraint is violated in the wrong direction; no constraint will be added.'
            warnings.warn(e)
            logger.warning(e)
        else:
            _eval = _FxExpr(self._f_x_expr)
            if self._function_shape == FunctionShape.CONVEX:
                xval = self._x.value
                expr = self._w >= _eval(xval) + _eval.deriv(xval) * (self._x - xval)
            else:
                assert self._function_shape == FunctionShape.CONCAVE
                xval = self._x.value
                expr = self._w <= _eval(xval) + _eval.deriv(xval) * (self._x - xval)

        return expr

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
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._w.value - pyo.value(self._f_x_expr)

    def is_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONVEX

    def is_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONCAVE

    @property
    def use_linear_relaxation(self):
        return True

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        if val is not True:
            raise ValueError('PWUnivariateRelaxation only supports linear relaxations.')

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} {2}'.format(self._w.name, relational_operator_string, str(self._f_x_expr))


@declare_custom_block(name='PWCosRelaxation')
class PWCosRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = cos(x) for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in w=cos(x).
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing cos(x)
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    use_linear_relaxation: bool
        If False, then linear overestimators will be built. If True, then a
        quadratic overestimator will be used. Note that a piecewise version of the
        quadratic overestimator is not supported.
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._use_linear_relaxation = True
        self._pw_repn = 'INC'

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        v = []
        if self._relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER} or (not self._use_linear_relaxation):
            v.append(self._x)
        return v

    def set_input(self, x, w, pw_repn='INC', use_linear_relaxation=True,
                  relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):

        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._pw_repn = pw_repn
        self._use_linear_relaxation = use_linear_relaxation

        self._xref.set_component(x)
        self._wref.set_component(w)
        self._partitions[self._x] = _get_bnds_list(self._x)

    def build(self, x, w, pw_repn='INC', use_linear_relaxation=True,
              relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self.set_input(x=x, w=w, pw_repn=pw_repn, use_linear_relaxation=use_linear_relaxation,
                       relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        pw_cos_relaxation(b=self, x=self._x, w=self._w, x_pts=self._partitions[self._x],
                          relaxation_side=self._relaxation_side, pw_repn=self._pw_repn,
                          use_quadratic_overestimator=(not self._use_linear_relaxation))

    def _get_cut_expr(self):
        """
        Add a linear cut on the convex side of the constraint based on the current
        values of the variables. There is no need to call build_relaxation. This
        method directly adds a constraint to the block. A new point will NOT be added
        to the partitioning! This method does not change the partitioning!
        The current relaxation is not discarded and rebuilt. A constraint is simply added.
        """
        expr = None
        viol = self.get_violation()
        if viol <= 0:
            e = 'Cannot add cut; constraint is violated in the wrong direction; no constraint will be added.'
            warnings.warn(e)
            logger.warning(e)
        else:
            xval = pyo.value(self._x)
            expr = self._w <= pyo.cos(xval) - pyo.sin(xval) * (self._x - xval)

        return expr

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
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._w.value - float(np.cos(pyo.value(self._x)))

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
        return True

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        self._use_linear_relaxation = val

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} cos({2})'.format(self._w.name, relational_operator_string, self._x.name)


@declare_custom_block(name='PWSinRelaxation')
class PWSinRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = sin(x) for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in w=cos(x).
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing cos(x)
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._pw_repn = 'INC'

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        v = []
        xlb = pyo.value(self._x.lb)
        xub = pyo.value(self._x.ub)
        if self._relaxation_side is RelaxationSide.BOTH:
            v.append(self._x)
        elif xlb < 0 and xub > 0:
            v.append(self._x)
        elif xlb >= 0:
            if self._relaxation_side is RelaxationSide.UNDER:
                v.append(self._x)
        else:
            assert xub <= 0
            if self._relaxation_side is RelaxationSide.OVER:
                v.append(self._x)
        return v

    def set_input(self, x, w, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._pw_repn = pw_repn
        self._xref.set_component(x)
        self._wref.set_component(w)
        self._partitions[self._x] = _get_bnds_list(self._x)

    def build(self, x, w, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self.set_input(x=x, w=w, pw_repn=pw_repn, relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        pw_sin_relaxation(b=self, x=self._x, w=self._w, x_pts=self._partitions[self._x],
                          relaxation_side=self._relaxation_side, pw_repn=self._pw_repn)

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
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._w.value - float(np.sin(pyo.value(self._x)))

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
            raise ValueError('PWSinRelaxation only supports linear relaxations.')

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} sin({2})'.format(self._w.name, relational_operator_string, self._x.name)


@declare_custom_block(name='PWArctanRelaxation')
class PWArctanRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = arctan(x).

    Parameters
    ----------
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in w=arctan(x).
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing arctan(x)
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._pw_repn = 'INC'

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    def vars_with_bounds_in_relaxation(self):
        v = []
        xlb = pyo.value(self._x.lb)
        xub = pyo.value(self._x.ub)
        if self._relaxation_side is RelaxationSide.BOTH:
            v.append(self._x)
        elif xlb < 0 and xub > 0:
            v.append(self._x)
        elif xlb >= 0:
            if self._relaxation_side is RelaxationSide.UNDER:
                v.append(self._x)
        else:
            assert xub <= 0
            if self._relaxation_side is RelaxationSide.OVER:
                v.append(self._x)
        return v

    def set_input(self, x, w, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self._pw_repn = pw_repn
        self._xref.set_component(x)
        self._wref.set_component(w)
        self._partitions[self._x] = _get_bnds_list(self._x)

    def build(self, x, w, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None):
        self.set_input(x=x, w=x, pw_repn=pw_repn, relaxation_side=relaxation_side, persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        pw_arctan_relaxation(b=self, x=self._x, w=self._w, x_pts=self._partitions[self._x],
                             relaxation_side=self._relaxation_side, pw_repn=self._pw_repn)

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
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._w.value - float(np.arctan(pyo.value(self._x)))

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
            raise ValueError('PWArctanRelaxation only supports linear relaxations.')

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} arctan({2})'.format(self._w.name, relational_operator_string, self._x.name)
