import pyomo.environ as pyo
from coramin.utils import FunctionShape, RelaxationSide
from coramin.relaxations.relaxations_base import BasePWRelaxationData, ComponentWeakRef
import warnings
from coramin.relaxations.univariate import PWXSquaredRelaxationData, pw_x_squared_relaxation
from coramin.relaxations.pw_mccormick import PWMcCormickRelaxationData, _build_pw_mccormick_relaxation
from coramin.relaxations.custom_block import declare_custom_block
import coramin.relaxations._utils as _utils

import logging
logger = logging.getLogger(__name__)


def _build_pw_soc_relaxation(b, x, y, z, w, x_pts, y_pts, z_pts, relaxation_side=RelaxationSide.BOTH,
                             use_nonlinear_underestimator=False, pw_repn='INC', safety_tol=1e-10):
    xlb = pyo.value(x.lb)
    xub = pyo.value(x.ub)
    ylb = pyo.value(y.lb)
    yub = pyo.value(y.ub)
    zlb = pyo.value(z.lb)
    zub = pyo.value(z.ub)
    wlb = pyo.value(w.lb)
    wub = pyo.value(w.ub)

    _utils.check_var_pts(x, x_pts)
    _utils.check_var_pts(y, y_pts)
    _utils.check_var_pts(z, z_pts)
    _utils.check_var_pts(w)

    if wlb < 0:
        e = ('Lower bound is negative for w; not a valid second-order cone.\n' + _utils.var_info_str(
             w) + _utils.bnds_info_str(wlb, wub))
        logger.error(e)
        raise ValueError(e)

    if zlb < 0:
        e = ('Lower bound is negative for z; not a valid second-order cone.\n' + _utils.var_info_str(
             z) + _utils.bnds_info_str(zlb, zub))
        logger.error(e)
        raise ValueError(e)

    if use_nonlinear_underestimator and relaxation_side == RelaxationSide.OVER:
        e = ('Piecewise SOC relaxation: if use_nonlinear_underestimator is True, then relaxation_side needs' +
             ' to be FunctionShape.UNDER or FunctionShape.BOTH')
        logger.error(e)
        raise ValueError(e)

    if relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        def get_v_sq_bounds(vlb, vub):
            if (vlb <= 0) and (vub >= 0):
                v_sq_lb = 0
                v_sq_ub = max([vlb**2, vub**2])
            else:
                v_sq_lb = min([vlb**2, vub**2])
                v_sq_ub = max([vlb**2, vub**2])
            return v_sq_lb, v_sq_ub

        b.x_sq = pyo.Var(bounds=get_v_sq_bounds(xlb, xub))
        b.y_sq = pyo.Var(bounds=get_v_sq_bounds(ylb, yub))
        b.zw = pyo.Var()
        b.eq_x_sq_y_sq_zw = pyo.Constraint(expr=b.x_sq + b.y_sq == b.zw)
        b.x_sq_relaxation = pyo.Block()
        b.y_sq_relaxation = pyo.Block()
        b.zw_mcc_relaxation = pyo.Block()
        pw_x_squared_relaxation(b=b.x_sq_relaxation, x=x, w=b.x_sq, x_pts=x_pts, pw_repn=pw_repn,
                                relaxation_side=RelaxationSide.OVER, use_nonlinear_underestimator=False)
        pw_x_squared_relaxation(b=b.y_sq_relaxation, x=y, w=b.y_sq, x_pts=y_pts, pw_repn=pw_repn,
                                relaxation_side=RelaxationSide.OVER, use_nonlinear_underestimator=False)
        _build_pw_mccormick_relaxation(b=b.zw_mcc_relaxation, x=z, y=w, w=b.zw, x_pts=z_pts,
                                       relaxation_side=RelaxationSide.UNDER)

    if relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        if use_nonlinear_underestimator:
            b.underestimator = pyo.Constraint(expr=x**2 + y**2 <= z*w)
        else:
            b.underestimator = pyo.ConstraintList()
            for _x in x_pts:
                for _y in y_pts:
                    for _z in z_pts:
                        if abs(_z) <= 1e-8:
                            continue
                        _w = (_x**2 + _y**2) / _z
                        assert abs(_x**2 + _y**2 - _z*_w) <= 1e-12
                        b.underestimator.add(_x**2 + _y**2 + 2*_x*(x-_x) + 2*_y*(y-_y) <= _w*z + _z*w - _z*_w + safety_tol)


def _project_point_onto_cone(x_val, y_val, z_val, w_val):
    """
    Try to find a point on the cone x0, y0, z0, w0 such that

    z0 = z_val
    w0 = w_val
    x0, y0 lie on the circle x**2 + y**2 == z_val*w_val and are as close as possible to x_val, y_val

    In order to find such a point, we can require that x0 and y0 lie on the circle and that (x0,y0) and
    (x_val, y_val) line on a line that passes through the origin. Therefore, we must solve the equations

    x0**2 + y0**2 = z_val*w_val
    y0 = (y_val/x_val) * x0

    The solution to these equations is

    x0 = +/- ((z_val * w_val * x_val**2) / (x_val**2 + y_val**2))**0.5
    y0 = (y_val/x_val) * x0

    However, x0 should take on the sign of x_val and y0 should take on the sign of y_val. Therefore,

    x0 = sign(x_val) * ((z_val * w_val * x_val**2) / (x_val**2 + y_val**2))**0.5
    y0 = (y_val/x_val) * x0

    Parameters
    ----------
    x_val: float
    y_val: float
    z_val: float
    w_val: float

    Returns
    -------
    x0: float
    y0: float
    z0: float
    w0: float
    """
    x_val = float(x_val)
    y_val = float(y_val)
    z_val = float(z_val)
    w_val = float(w_val)

    if z_val < 1e-8 or w_val < 1e-8:
        x0 = x_val
        y0 = y_val
        zw_val = x0**2 + y0**2
        if z_val > w_val:
            z0 = z_val
            w0 = zw_val / z0
        elif w_val > z_val:
            w0 = w_val
            z0 = zw_val / w0
        else:
            raise NotImplementedError('this case is not supported yet.')
    else:
        z0 = z_val
        w0 = w_val

        x0 = x_val * ((z_val * w_val) / (x_val**2 + y_val**2)) ** 0.5
        y0 = y_val * ((z_val * w_val) / (x_val**2 + y_val**2)) ** 0.5

    assert abs(x0**2 + y0**2 - z0*w0) <= 1e-12

    return x0, y0, z0, w0


@declare_custom_block(name='PWSOCRelaxation')
class PWSOCRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of x**2 + y**2 == z*w where z and w are positive.

    Parameters
    ----------
    x: pyomo.core.base.var._GeneralVarData
        The "x" variable in x**2 + y**2 == z*w
    y: pyomo.core.base.var._GeneralVarData
        The "y" variable in x**y + y**2 == z*w
    z: pyomo.core.base.var._GeneralVarData
        The "z" variable in x**2 + y**2 == z*w
    w: pyomo.core.base.var._GeneralVarData
        The "w" variable in x**y + y**2 == z*w
    pw_repn: str
        This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more. This is only used for the overestimators of the quadratic terms.
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        Under: x**2 + y**2 <= z*w
        Over: x**2 + y**2 >= z*w
    use_linear_relaxation: bool
        If True, then linear underestimators will be built for x**2 + y**2 <= z*w.
        If False, then the soc underestimator will be built ( x**2 + y**2 <= z*y )
    """
    def __init__(self, component):

        BasePWRelaxationData.__init__(self, component)
        self._xref = ComponentWeakRef(None)
        self._yref = ComponentWeakRef(None)
        self._wref = ComponentWeakRef(None)
        self._zref = ComponentWeakRef(None)
        self._pw_repn = 'INC'
        self._use_linear_relaxation = False

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _y(self):
        return self._yref.get_component()

    @property
    def _w(self):
        return self._wref.get_component()

    @property
    def _z(self):
        return self._zref.get_component()

    def vars_with_bounds_in_relaxation(self):
        res = []
        if self._relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            res.append(self._x)
            res.append(self._y)
            res.append(self._w)
            res.append(self._z)
        return res

    def _set_input(self, kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y')
        w = kwargs.pop('w')
        z = kwargs.pop('z')

        self._xref.set_component(x)
        self._yref.set_component(y)
        self._wref.set_component(w)
        self._zref.set_component(z)

        xlb = pyo.value(self._x.lb)
        xub = pyo.value(self._x.ub)
        ylb = pyo.value(self._y.lb)
        yub = pyo.value(self._y.ub)
        zlb = pyo.value(self._z.lb)
        zub = pyo.value(self._z.ub)

        self._partitions[self._x] = [xlb, xub]
        self._partitions[self._y] = [ylb, yub]
        self._partitions[self._z] = [zlb, zub]

        self._pw_repn = kwargs.pop('pw_repn', 'INC')
        self._use_linear_relaxation = kwargs.pop('use_linear_relaxation', True)

    def _build_relaxation(self):
        _build_pw_soc_relaxation(b=self, x=self._x, y=self._y, z=self._z, w=self._w, x_pts=self._partitions[self._x],
                                 y_pts=self._partitions[self._y], z_pts=self._partitions[self._z],
                                 relaxation_side=self._relaxation_side,
                                 use_nonlinear_underestimator=(not self._use_linear_relaxation),
                                 pw_repn=self._pw_repn)

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
            yval = pyo.value(self._y)
            zval = pyo.value(self._z)
            wval = pyo.value(self._w)
            x0, y0, z0, w0 = _project_point_onto_cone(xval, yval, zval, wval)
            if x0**2 + y0**2 + 2*x0*(xval-x0) + 2*y0*(yval-y0) <= w0*zval + z0*wval - z0*w0:
                raise RuntimeError('cut does not cut off point!')
            expr = x0**2 + y0**2 + 2*x0*(self._x - x0) + 2*y0*(self._y - y0) <= w0*self._z + z0*self._w - z0*w0

        return expr

    def add_point(self, xval=None, yval=None, zval=None):
        self._add_point(self._x, xval)
        self._add_point(self._y, yval)
        self._add_point(self._z, zval)

    def _get_violation(self):
        """
        Get the signed constraint violation.

        Returns
        -------
        float
        """
        return self._z.value*self._w.value - self._x.value**2 - self._y.value**2

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
