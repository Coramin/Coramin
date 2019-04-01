import pyomo.environ as pe
import logging
import warnings

logger = logging.getLogger(__name__)
pyo = pe


def var_info_str(v):
    s = '\tVar: {0}\n'.format(v)
    return s


def bnds_info_str(vlb, vub):
    s = '\tLB: {0}\n'.format(vlb)
    s += '\tUB: {0}\n'.format(vub)
    return s


def x_pts_info_str(_x_pts):
    s = '\tx_pts: {0}\n'.format(_x_pts)
    return s


def check_var_pts(x, x_pts=None):
    xlb = pe.value(x.lb)
    xub = pe.value(x.ub)

    raise_error = False
    raise_warning = False
    msg = None

    if xub < xlb:
        msg = 'Lower bound is larger than upper bound:\n' + var_info_str(x) + bnds_info_str(xlb, xub)
        raise_error = True

    if x_pts is not None:
        ordered = all(x_pts[i] <= x_pts[i+1] for i in range(len(x_pts)-1))
        if not ordered:
            msg = 'x_pts must be ordered:\n' + var_info_str(x) + bnds_info_str(xlb, xub) + x_pts_info_str(x_pts)
            raise_error = True

        if xlb != x_pts[0] or xub != x_pts[-1]:
            msg = ('end points of the x_pts list must be equal to the bounds on the x variable:\n' + var_info_str(x) +
                   bnds_info_str(xlb, xub) + x_pts_info_str(x_pts))
            raise_error = True

    if abs(pyo.value(x.ub) - pyo.value(x.lb)) <= 1e-8:
        msg = ('The difference between the variables lower and upper bounds is small:\n' + var_info_str(x) +
               bnds_info_str(xlb, xub))
        raise_warning = True

    if raise_error:
        logger.error(msg)
        raise ValueError(msg)

    if raise_warning:
        logger.warning(msg)
        warnings.warn(msg)
