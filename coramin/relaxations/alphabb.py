import pyomo.environ as pyo
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from coramin.utils.coramin_enums import RelaxationSide
from coramin.relaxations.custom_block import declare_custom_block
from coramin.relaxations.relaxations_base import BaseRelaxationData, ComponentWeakRef


def _hessian(xs, f_x_expr):
    hess = ComponentMap()
    df_dx_map = reverse_sd(f_x_expr)
    for x in xs:
        ddf_ddx_map = reverse_sd(df_dx_map[x])
        hess[x] = ddf_ddx_map
    return hess


def _compute_alpha(xs, f_x_expr):
    hess = _hessian(xs, f_x_expr)
    alpha = 0.0
    for i, x in enumerate(xs):
        a_ii_expr = hess[x][x]
        a_ii = compute_bounds_on_expr(a_ii_expr)
        tot = a_ii[0]
        for j, y in enumerate(xs):
            if i == j:
                continue
            a_ij_expr = hess[x][y]
            a_ij = compute_bounds_on_expr(a_ij_expr)
            tot -= max(abs(a_ij[0]), abs(a_ij[1]))
        tot = - 0.5 * tot
        if tot > alpha:
            alpha = tot
    return alpha


def _build_alphabb_relaxation(xs, f_x_expr, alpha):
    return f_x_expr + alpha * pyo.quicksum((x - x.lb)*(x - x.ub) for x in xs)


def _build_multivariate_underestimator(xs, f):
    df = reverse_sd(f)
    result = pyo.value(f)
    for x in xs:
        result += pyo.value(df[x]) * (x - x.value)
    return result


def _build_alphabb_constraints(b, xs, w, f_x_expr, alpha, linear=False, xs_pts=None):
    alphabb_expr = _build_alphabb_relaxation(xs, f_x_expr, alpha)
    if not linear:
        b.con = pyo.Constraint(expr=w >= alphabb_expr)
    else:
        b.underestimators = pyo.ConstraintList()
        for pt in xs_pts:
            for v, p in zip(xs, pt):
                v.value = p
            underestimator = _build_multivariate_underestimator(xs, alphabb_expr)
            b.underestimators.add(w >= underestimator)


@declare_custom_block(name='AlphaBBRelaxation')
class AlphaBBRelaxationData(BaseRelaxationData):
    """

    Parameters
    ----------
    x: pyomo.core.base.var._GeneralVarData or list of pyomo.core.base.var._GeneralVarData
        The "x" variable or variables in w=f(x)
    w: pyomo.core.base.var._GeneralVarData
        The auxiliary variable replacing f(x)
    f_x_expr: pyomo expression
        The pyomo expression representing f(x)
    compute_alpha: func
        Callback that given f(x) returns alpha
    """
    def __init__(self, component):
        super().__init__(component)
        self._xs = None
        self._wref = ComponentWeakRef(None)
        self._f_x_expr = None
        self._linear = False
        self._points = []
        self._compute_alpha = None

    @property
    def _w(self):
        return self._wref.get_component()

    def add_point(self):
        pts = [pyo.value(x) for x in self._xs]
        self._points.append(pts)

    def set_input(self, xs, w, f_x_expr, compute_alpha=_compute_alpha, persistent_solvers=None):
        self._set_input(relaxation_side=RelaxationSide.UNDER, persistent_solvers=persistent_solvers)
        self._compute_alpha = compute_alpha

        if not isinstance(xs, list):
            xs = [xs]

        self._xs = xs
        self._wref.set_component(w)
        self._f_x_expr = f_x_expr

    def build(self, xs, w, f_x_expr, compute_alpha=_compute_alpha, persistent_solvers=None):
        self.set_input(xs=xs, w=w, f_x_expr=f_x_expr, compute_alpha=compute_alpha,
                       persistent_solvers=persistent_solvers)
        self.rebuild()

    def _build_relaxation(self):
        w = self._wref.get_component()
        alpha = self._compute_alpha(self._xs, self._f_x_expr)
        _build_alphabb_constraints(
            b=self,
            xs=self._xs,
            w=w,
            f_x_expr=self._f_x_expr,
            alpha=alpha,
            linear=self._linear,
            xs_pts=self._points,
        )

    def _get_violation(self):
        viol = self._w.value - pyo.value(self._f_x_expr)
        return min(viol, 0.0)

    def get_abs_violation(self):
        return abs(self.get_violation())

    def vars_with_bounds_in_relaxation(self):
        return self._xs

    def is_convex(self):
        return True

    def is_concave(self):
        return False

    @property
    def use_linear_relaxation(self):
        return self._linear

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, value):
        self._linear = value

    @property
    def relaxation_side(self):
        return RelaxationSide.UNDER

    @relaxation_side.setter
    def relaxation_side(self, val):
        if val != RelaxationSide.UNDER:
            raise ValueError('relaxation_side must be RelaxationSide.UNDER')

    def _get_pprint_string(self, relational_operator_string):
        return 'Relaxation for {0} {1} {2}'.format(self._w.name, relational_operator_string, str(self._f_x_expr))
