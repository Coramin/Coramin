from coramin.utils.coramin_enums import EigenValueBounder, RelaxationSide
from coramin.relaxations.custom_block import declare_custom_block
from coramin.relaxations.relaxations_base import BaseRelaxationData, ComponentWeakRef
from coramin.relaxations.hessian import Hessian
from typing import Optional, Tuple
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExpressionBase
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib import appsi
from pyomo.core.base.param import ScalarParam


@declare_custom_block(name='AlphaBBRelaxation')
class AlphaBBRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        super().__init__(component)
        self._xs: Optional[Tuple[_GeneralVarData]] = None
        self._aux_var_ref = ComponentWeakRef(None)
        self._f_x_expr: Optional[ExpressionBase] = None
        self._alphabb_rhs: Optional[ExpressionBase] = None
        self._hessian: Optional[Hessian] = None
        self._alpha: Optional[ScalarParam] = None

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self) -> Tuple[_GeneralVarData, ...]:
        return self._xs

    def get_rhs_expr(self) -> ExpressionBase:
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        if self.is_rhs_convex():
            return list()
        else:
            return list(self._xs)

    def is_rhs_convex(self):
        return self._hessian.get_minimum_eigenvalue() >= 0

    def is_rhs_concave(self):
        return self._hessian.get_maximum_eigenvalue() <= 0

    def _has_convex_underestimator(self):
        return self.relaxation_side == RelaxationSide.UNDER

    def _has_concave_overestimator(self):
        return self.relaxation_side == RelaxationSide.OVER

    def _get_expr_for_oa(self):
        return self._alphabb_rhs

    def _remove_relaxation(self):
        del self._alpha, self._alphabb_rhs
        self._alpha = None
        self._alphabb_rhs = None

    def remove_relaxation(self):
        super().remove_relaxation()
        self._remove_relaxation()

    def set_input(
        self,
        aux_var: _GeneralVarData,
        f_x_expr: ExpressionBase,
        relaxation_side: RelaxationSide,
        use_linear_relaxation: bool = True,
        large_coef: float = 1e5,
        small_coef: float = 1e-10,
        safety_tol: float = 1e-10,
        eigenvalue_bounder: EigenValueBounder = EigenValueBounder.LinearProgram,
        eigenvalue_opt: Optional[appsi.base.Solver] = None,
        hessian: Optional[Hessian] = None,
    ):
        super().set_input(
            relaxation_side=relaxation_side,
            use_linear_relaxation=use_linear_relaxation,
            large_coef=large_coef,
            small_coef=small_coef,
            safety_tol=safety_tol
        )
        self._xs = tuple(identify_variables(f_x_expr, include_fixed=False))
        self._aux_var_ref.set_component(aux_var)
        self._f_x_expr = f_x_expr
        if hessian is None:
            hessian = Hessian(
                expr=f_x_expr, opt=eigenvalue_opt, method=eigenvalue_bounder
            )
        self._hessian = hessian

    def build(
        self,
        aux_var: _GeneralVarData,
        f_x_expr: ExpressionBase,
        relaxation_side: RelaxationSide,
        use_linear_relaxation: bool = True,
        large_coef: float = 1e5,
        small_coef: float = 1e-10,
        safety_tol: float = 1e-10,
        eigenvalue_bounder: EigenValueBounder = EigenValueBounder.LinearProgram,
        eigenvalue_opt: appsi.base.Solver = None,
    ):
        self.set_input(
            aux_var=aux_var,
            f_x_expr=f_x_expr,
            relaxation_side=relaxation_side,
            use_linear_relaxation=use_linear_relaxation,
            large_coef=large_coef,
            small_coef=small_coef,
            safety_tol=safety_tol,
            eigenvalue_bounder=eigenvalue_bounder,
            eigenvalue_opt=eigenvalue_opt,
        )
        self.rebuild()

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, value):
        self._use_linear_relaxation = value

    @property
    def relaxation_side(self):
        return BaseRelaxationData.relaxation_side.fget(self)

    @relaxation_side.setter
    def relaxation_side(self, val):
        if val != self.relaxation_side:
            raise ValueError('Cannot change the relaxation side of an AlphaBBRelaxation')

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        if self.relaxation_side == RelaxationSide.UNDER:
            alpha = max(0, -0.5 * self._hessian.get_minimum_eigenvalue())
        else:
            alpha = max(0, 0.5*self._hessian.get_maximum_eigenvalue())
            alpha = -alpha
        if self._alpha is None:
            self._alpha = ScalarParam(mutable=True)
            self._alphabb_rhs = (
                self.get_rhs_expr()
                + self._alpha
                * sum((x - x.lb) * (x - x.ub) for x in self.get_rhs_vars())
            )
        self._alpha.value = alpha

        super().rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                        ensure_oa_at_vertices=ensure_oa_at_vertices)
