from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from coramin.relaxations.custom_block import declare_custom_block
from coramin.relaxations.relaxations_base import BaseRelaxationData, ComponentWeakRef
from pyomo.core.expr.visitor import identify_variables
import math
import pyomo.environ as pe
from coramin.relaxations._utils import _get_bnds_list


@declare_custom_block(name='MultivariateRelaxation')
class MultivariateRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        super(MultivariateRelaxationData, self).__init__(component)
        self._xs = None
        self._aux_var_ref = ComponentWeakRef(None)
        self._f_x_expr = None
        self._function_shape = FunctionShape.UNKNOWN

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return self._xs

    def get_rhs_expr(self):
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        return list()

    def set_input(self, aux_var, shape, f_x_expr, persistent_solvers=None, large_eval_tol=math.inf,
                  use_linear_relaxation=True):
        """
        Parameters
        ----------
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxiliary variable replacing f(x)
        shape: FunctionShape
            Either FunctionShape.CONVEX or FunctionShape.CONCAVE
        f_x_expr: pyomo expression
            The pyomo expression representing f(x)
        persistent_solvers: list
            List of persistent solvers that should be updated when the relaxation changes
        large_eval_tol: float
            To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol,
            at a point in x_pts, then that point is skipped.
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        if shape not in {FunctionShape.CONVEX, FunctionShape.CONCAVE}:
            raise ValueError('MultivariateRelaxation only supports concave or convex functions.')
        self._function_shape = shape
        if shape == FunctionShape.CONVEX:
            relaxation_side = RelaxationSide.UNDER
        else:
            relaxation_side = RelaxationSide.OVER
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=use_linear_relaxation, large_eval_tol=large_eval_tol)
        self._xs = tuple(identify_variables(f_x_expr, include_fixed=False))
        self._aux_var_ref.set_component(aux_var)
        self._f_x_expr = f_x_expr
        lb_oa_pt = pe.ComponentMap()
        ub_oa_pt = pe.ComponentMap()
        should_use_lb_oa_pt = True
        should_use_ub_oa_pt = True
        for v in self._xs:
            lb, ub = tuple(_get_bnds_list(v))
            if lb <= -math.inf:
                should_use_lb_oa_pt = False
            else:
                lb_oa_pt[v] = lb
            if ub >= math.inf:
                should_use_ub_oa_pt = False
            else:
                ub_oa_pt[v] = ub
        if should_use_lb_oa_pt:
            self.add_oa_point(var_values=lb_oa_pt)
        if should_use_ub_oa_pt:
            self.add_oa_point(var_values=ub_oa_pt)

    def build(self, aux_var, shape, f_x_expr, persistent_solvers=None, large_eval_tol=math.inf,
              use_linear_relaxation=True):
        self.set_input(aux_var=aux_var, shape=shape, f_x_expr=f_x_expr,
                       persistent_solvers=persistent_solvers, large_eval_tol=large_eval_tol,
                       use_linear_relaxation=use_linear_relaxation)
        self.rebuild()

    def _build_relaxation(self):
        pass

    def is_rhs_convex(self):
        return self._function_shape is FunctionShape.CONVEX

    def is_rhs_concave(self):
        return self._function_shape is FunctionShape.CONCAVE

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, value):
        self._use_linear_relaxation = value

    @property
    def relaxation_side(self):
        return self._relaxation_side

    @relaxation_side.setter
    def relaxation_side(self, val):
        if self.is_rhs_convex():
            if val != RelaxationSide.UNDER:
                raise ValueError('MultivariateRelaxations only support underestimators for convex functions')
        if self.is_rhs_concave():
            if val != RelaxationSide.OVER:
                raise ValueError('MultivariateRelaxations only support overestimators for concave functions')
