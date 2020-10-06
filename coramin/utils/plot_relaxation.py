import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import pyomo.environ as pe
from .pyomo_utils import get_objective
from .coramin_enums import RelaxationSide
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver


def _solve_loop(m, x, w, x_list, using_persistent_solver, solver):
    w_list = list()
    for _xval in x_list:
        x.fix(_xval)
        if using_persistent_solver:
            solver.update_var(x)
            res = solver.solve(load_solutions=False, save_results=False)
        else:
            res = solver.solve(m, load_solutions=False)
        if res.solver.termination_condition != pe.TerminationCondition.optimal:
            raise RuntimeError(
                'Could not produce plot because solver did not terminate optimally. Termination condition: ' + str(
                    res.solver.termination_condition))
        if using_persistent_solver:
            solver.load_vars([w])
        else:
            m.solutions.load_from(res)
        w_list.append(w.value)
    return w_list


def plot_relaxation(m, relaxation, solver, show_plot=True, num_pts=100):
    if len(relaxation.get_rhs_vars()) != 1:
        raise NotImplementedError('Plotting is not supported for {0}.'.format(type(relaxation)))
    
    using_persistent_solver = isinstance(solver, PersistentSolver)

    x = relaxation.get_rhs_vars()[0]
    w = relaxation.get_aux_var()

    if not x.has_lb() or not x.has_ub():
        raise ValueError('rhs var must have bounds')

    orig_xval = x.value
    orig_wval = w.value
    xlb = pe.value(x.lb)
    xub = pe.value(x.ub)

    orig_obj = get_objective(m)
    if orig_obj is not None:
        orig_obj.deactivate()

    x_list = np.linspace(xlb, xub, num_pts)
    x_list = [float(i) for i in x_list]
    w_true = list()

    rhs_expr = relaxation.get_rhs_expr()
    for _x in x_list:
        x.value = float(_x)
        w_true.append(pe.value(rhs_expr))
    plt.plot(x_list, w_true, label=str(rhs_expr))

    m._plotting_objective = pe.Objective(expr=w)
    if using_persistent_solver:
        solver.set_instance(m)

    if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        w_min = _solve_loop(m, x, w, x_list, using_persistent_solver, solver)
        plt.plot(x_list, w_min, label='underestimator')

    del m._plotting_objective
    m._plotting_objective = pe.Objective(expr=w, sense=pe.maximize)
    if using_persistent_solver:
        solver.set_objective(m._plotting_objective)

    if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        w_max = _solve_loop(m, x, w, x_list, using_persistent_solver, solver)
        plt.plot(x_list, w_max, label='overestimator')

    plt.legend()
    plt.xlabel(x.name)
    plt.ylabel(w.name)
    if show_plot:
        plt.show()

    x.unfix()
    x.value = orig_xval
    w.value = orig_wval
    del m._plotting_objective
    if orig_obj is not None:
        orig_obj.activate()
