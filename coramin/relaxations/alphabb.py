import pyomo.environ as pyo
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from coramin.utils.coramin_enums import FunctionShape
from coramin.relaxations.custom_block import declare_custom_block
from coramin.relaxations.multivariate import MultivariateRelaxationData
import math


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


@declare_custom_block(name='AlphaBBRelaxation')
class AlphaBBRelaxationData(MultivariateRelaxationData):
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
        self._compute_alpha = None
        self._alphabb_rhs = None

    def get_rhs_expr(self):
        return self._alphabb_rhs

    def set_input(self, aux_var, f_x_expr, compute_alpha=_compute_alpha, persistent_solvers=None,
                  large_eval_tol=math.inf, use_linear_relaxation=True):
        super().set_input(aux_var=aux_var, shape=FunctionShape.CONVEX, f_x_expr=f_x_expr,
                          persistent_solvers=persistent_solvers, large_eval_tol=large_eval_tol,
                          use_linear_relaxation=use_linear_relaxation)
        self._compute_alpha = compute_alpha

    def build(self, aux_var, f_x_expr, compute_alpha=_compute_alpha, persistent_solvers=None,
              large_eval_tol=math.inf, use_linear_relaxation=True):
        self.set_input(aux_var=aux_var, f_x_expr=f_x_expr, compute_alpha=compute_alpha,
                       persistent_solvers=persistent_solvers, large_eval_tol=large_eval_tol,
                       use_linear_relaxation=use_linear_relaxation)
        self.rebuild()

    def _build_relaxation(self):
        alpha = self._compute_alpha(self.get_rhs_vars(), self._f_x_expr)
        self._alphabb_rhs = _build_alphabb_relaxation(xs=self.get_rhs_vars(),
                                                      f_x_expr=self._f_x_expr,
                                                      alpha=alpha)

    def vars_with_bounds_in_relaxation(self):
        return self.get_rhs_vars()
