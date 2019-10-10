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
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad, reverse_sd
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
                             relaxation_side=RelaxationSide.BOTH, large_eval_tol=math.inf):
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
    if shape not in {FunctionShape.CONCAVE, FunctionShape.CONVEX}:
        e = 'pw_univariate_relaxation: shape must be either FunctionShape.CONCAVE or FunctionShape.CONVEX'
        logger.error(e)
        raise ValueError(e)

    if relaxation_side is RelaxationSide.BOTH:
        if shape is FunctionShape.CONVEX:
            logger.warning('pw_univariate_relaxation does not handle the underestimators for convex functions')
        elif shape is FunctionShape.CONCAVE:
            logger.warning('pw_univariate_relaxation does not handle the overestimators for concave functions')
    elif relaxation_side is RelaxationSide.UNDER:
        if shape is FunctionShape.CONVEX:
            logger.warning('pw_univariate_relaxation does not handle the underestimators for convex functions')
    else:
        if shape is FunctionShape.CONCAVE:
            logger.warning('pw_univariate_relaxation does not handle the overestimators for concave functions')

    _eval = _FxExpr(expr=f_x_expr, x=x)
    xlb = x_pts[0]
    xub = x_pts[-1]

    check_var_pts(x, x_pts)

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
                        if f <= -large_eval_tol:
                            logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is less than {3}'.format(str(_pt), str(x), str(f_x_expr), -large_eval_tol))
                            continue
                        if f >= large_eval_tol:
                            logger.warning('Skipping pt {0} for var {1} because {2} evaluated at {0} is greater than {3}'.format(str(_pt), str(x), str(f_x_expr), large_eval_tol))
                            continue
                        tmp_pts.append(_pt)
                    except (ZeroDivisionError, ValueError, OverflowError):
                        pass
                if len(tmp_pts) >= 2 and tmp_pts[0] == x_pts[0] and tmp_pts[-1] == x_pts[-1]:
                    b.pw_linear_under_over = pyo.Piecewise(w, x,
                                                           pw_pts=tmp_pts,
                                                           pw_repn=pw_repn,
                                                           pw_constr_type=pw_constr_type,
                                                           f_rule=_func_wrapper(_eval)
                                                           )


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


@declare_custom_block(name='PWUnivariateRelaxation')
class PWUnivariateRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of aux_var = f(x) where f(x) is either convex
    or concave.
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._aux_var_ref = ComponentWeakRef(None)
        self._pw_repn = 'INC'
        self._function_shape = FunctionShape.UNKNOWN
        self._f_x_expr = None

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return [self._x]

    def get_rhs_expr(self):
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        v = []
        if self.relaxation_side is RelaxationSide.BOTH:
            v.append(self._x)
        elif self.relaxation_side is RelaxationSide.UNDER and self.is_rhs_concave():
            v.append(self._x)
        elif self.relaxation_side is RelaxationSide.OVER and self.is_rhs_convex():
            v.append(self._x)
        return v

    def set_input(self, x, aux_var, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        shape: FunctionShape
            Options are FunctionShape.CONVEX and FunctionShape.CONCAVE
        f_x_expr: pyomo expression
            The pyomo expression representing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=use_linear_relaxation, large_eval_tol=large_eval_tol)
        self._pw_repn = pw_repn
        self._function_shape = shape
        self._f_x_expr = f_x_expr

        self._xref.set_component(x)
        self._aux_var_ref.set_component(aux_var)
        bnds_list = _get_bnds_list(self._x)
        self._partitions[self._x] = bnds_list

    def build(self, x, aux_var, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        shape: FunctionShape
            Options are FunctionShape.CONVEX and FunctionShape.CONCAVE
        f_x_expr: pyomo expression
            The pyomo expression representing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        self.set_input(x=x, aux_var=aux_var, shape=shape, f_x_expr=f_x_expr, pw_repn=pw_repn,
                       relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                       large_eval_tol=large_eval_tol, use_linear_relaxation=True)
        self.rebuild()

    def _build_relaxation(self):
        if self.is_rhs_convex() and self.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONVEX,
                                     relaxation_side=RelaxationSide.OVER, large_eval_tol=self.large_eval_tol)
        elif self.is_rhs_concave() and self.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONCAVE,
                                     relaxation_side=RelaxationSide.UNDER, large_eval_tol=self.large_eval_tol)

    def add_partition_point(self, value=None):
        """
        This method adds one point to the partitioning of x. If value is not
        specified, a single point will be added to the partitioning of x at the current value of x. If value is
        specified, then value is added to the partitioning of x.

        Parameters
        ----------
        value: float
            The point to be added to the partitioning of x.
        """
        self._add_partition_point(self._x, value)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONVEX

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONCAVE

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        self._use_linear_relaxation = val

    def clean_oa_points(self):
        new_oa_points = list()
        lb, ub = tuple(_get_bnds_list(self._x))
        for pts in self._oa_points:
            pt = pts[self._x]
            if pt > lb and pt < ub:
                new_oa_points.append(pts)
        if lb > -math.inf:
            new_oa_points.append(pe.ComponentMap([(self._x, lb)]))
        if ub < math.inf:
            new_oa_points.append(pe.ComponentMap([(self._x, ub)]))
        self._oa_points = new_oa_points


@declare_custom_block(name='PWXSquaredRelaxation')
class PWXSquaredRelaxationData(PWUnivariateRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of aux_var = x**2.
    """
    def set_input(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        super(PWXSquaredRelaxationData, self).set_input(x=x, aux_var=aux_var, shape=FunctionShape.CONVEX,
                                                        f_x_expr=x**2, pw_repn=pw_repn,
                                                        relaxation_side=relaxation_side,
                                                        persistent_solvers=persistent_solvers,
                                                        large_eval_tol=large_eval_tol,
                                                        use_linear_relaxation=use_linear_relaxation)

    def build(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        self.set_input(x=x, aux_var=aux_var,
                       pw_repn=pw_repn,
                       relaxation_side=relaxation_side,
                       persistent_solvers=persistent_solvers,
                       large_eval_tol=large_eval_tol,
                       use_linear_relaxation=use_linear_relaxation)
        self.rebuild()


@declare_custom_block(name='PWCosRelaxation')
class PWCosRelaxationData(PWUnivariateRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = cos(x) for -pi/2 <= x <= pi/2.
    """
    def set_input(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        super(PWXSquaredRelaxationData, self).set_input(x=x, aux_var=aux_var, shape=FunctionShape.CONCAVE,
                                                        f_x_expr=pe.cos(x), pw_repn=pw_repn,
                                                        relaxation_side=relaxation_side,
                                                        persistent_solvers=persistent_solvers,
                                                        large_eval_tol=large_eval_tol,
                                                        use_linear_relaxation=use_linear_relaxation)

    def build(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              persistent_solvers=None, large_eval_tol=math.inf, use_linear_relaxation=True):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        super(PWXSquaredRelaxationData, self).build(x=x, aux_var=aux_var, shape=FunctionShape.CONCAVE,
                                                    f_x_expr=pe.cos(x), pw_repn=pw_repn,
                                                    relaxation_side=relaxation_side,
                                                    persistent_solvers=persistent_solvers,
                                                    large_eval_tol=large_eval_tol,
                                                    use_linear_relaxation=use_linear_relaxation)

    def rebuild(self, build_nonlinear_constraint=False):
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= -math.pi/2 and ub <= math.pi/2:
            super(PWCosRelaxationData, self).rebuild(build_nonlinear_constraint=build_nonlinear_constraint)
        else:
            self.remove_relaxation()

    def is_rhs_convex(self):
        return False

    def is_rhs_concave(self):
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb > -math.pi/2 and ub < math.pi/2:
            return True
        else:
            return False


@declare_custom_block(name='PWSinRelaxation')
class PWSinRelaxationData(PWUnivariateRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = sin(x) for -pi/2 <= x <= pi/2.
    """

    def set_input(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
                  use_linear_relaxation=True):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=use_linear_relaxation, large_eval_tol=math.inf)
        self._pw_repn = pw_repn
        self._xref.set_component(x)
        self._aux_var_ref.set_component(aux_var)
        self._partitions[self._x] = _get_bnds_list(self._x)
        self._f_x_expr = pe.sin(x)

    def build(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
              use_linear_relaxation=True):
        self.set_input(x=x, aux_var=aux_var, pw_repn=pw_repn, relaxation_side=relaxation_side,
                       persistent_solvers=persistent_solvers, use_linear_relaxation=use_linear_relaxation)
        self.rebuild()

    def rebuild(self, build_nonlinear_constraint=False):
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= -math.pi / 2 and ub <= math.pi / 2:
            super(PWSinRelaxationData, self).rebuild(build_nonlinear_constraint=build_nonlinear_constraint)
        else:
            self.remove_relaxation()

    def _build_relaxation(self):
        if self.is_rhs_convex() and self.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONVEX,
                                     relaxation_side=RelaxationSide.OVER, large_eval_tol=self.large_eval_tol)
        elif self.is_rhs_concave() and self.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONCAVE,
                                     relaxation_side=RelaxationSide.UNDER, large_eval_tol=self.large_eval_tol)
        if (not self.is_rhs_convex()) and (not self.is_rhs_concave()):
            pw_sin_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                              relaxation_side=self.relaxation_side, pw_repn=self._pw_repn)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= -math.pi / 2 and ub <= 0:
            return True
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= 0 and ub <= math.pi / 2:
            return True
        return False


@declare_custom_block(name='PWArctanRelaxation')
class PWArctanRelaxationData(PWUnivariateRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = arctan(x).
    """

    def set_input(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
                  use_linear_relaxation=True):
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=use_linear_relaxation, large_eval_tol=math.inf)
        self._pw_repn = pw_repn
        self._xref.set_component(x)
        self._aux_var_ref.set_component(aux_var)
        self._partitions[self._x] = _get_bnds_list(self._x)
        self._f_x_expr = pe.atan(x)

    def build(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH, persistent_solvers=None,
              use_linear_relaxation=True):
        self.set_input(x=x, aux_var=aux_var, pw_repn=pw_repn, relaxation_side=relaxation_side,
                       persistent_solvers=persistent_solvers, use_linear_relaxation=use_linear_relaxation)
        self.rebuild()

    def _build_relaxation(self):
        if self.is_rhs_convex() and self.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONVEX,
                                     relaxation_side=RelaxationSide.OVER, large_eval_tol=self.large_eval_tol)
        elif self.is_rhs_concave() and self.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            pw_univariate_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                     f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONCAVE,
                                     relaxation_side=RelaxationSide.UNDER, large_eval_tol=self.large_eval_tol)
        if (not self.is_rhs_convex()) and (not self.is_rhs_concave()):
            pw_arctan_relaxation(b=self, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                 relaxation_side=self.relaxation_side, pw_repn=self._pw_repn)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if ub <= 0:
            return True
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= 0:
            return True
        return False
