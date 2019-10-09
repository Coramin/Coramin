import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
import coramin


def plot_feasible_region_2d(relaxation, show_plot=True, num_pts=30, tol=1e-4):
    if not isinstance(relaxation, (coramin.relaxations.PWXSquaredRelaxation,
                                   coramin.relaxations.PWUnivariateRelaxation,
                                   coramin.relaxations.PWCosRelaxation,
                                   coramin.relaxations.PWSinRelaxation,
                                   coramin.relaxations.PWArctanRelaxation)):
        raise NotImplementedError('Plotting is not supported for {0}.'.format(type(relaxation)))

    x = relaxation.get_rhs_vars()[0]
    w = relaxation.get_aux_var()

    rel_name = relaxation.local_name
    x_name = x.local_name
    w_name = w.local_name

    rel_parent = relaxation.parent_block()
    x_parent = x.parent_block()
    w_parent = w.parent_block()

    rel_parent.del_component(relaxation)
    x_parent.del_component(x)
    w_parent.del_component(w)

    orig_xlb = x.lb
    orig_xub = x.ub
    orig_wlb = w.lb
    orig_wub = w.ub

    fbbt(relaxation)

    xlb = pe.value(x.lb)
    xub = pe.value(x.ub)
    wlb = pe.value(w.lb)
    wub = pe.value(w.ub)

    x_list = np.linspace(xlb, xub, num_pts)
    w_list = np.linspace(wlb, wub, num_pts)

    w_true = list()
    rhs_expr = relaxation.get_rhs_expr()
    for _x in x_list:
        x.value = float(_x)
        w_true.append(pe.value(rhs_expr))
    plt.plot(x_list, w_true)

    x_list = [float(i) for i in x_list]
    w_list = [float(i) for i in w_list]

    m = pe.ConcreteModel()
    m.x = x
    m.w = w
    m.b = relaxation
    opt = pe.SolverFactory('gurobi_persistent')
    opt.options['BarQCPConvTol'] = 1e-8
    opt.options['FeasibilityTol'] = 1e-8
    opt.set_instance(m)

    feasible_x = list()
    feasible_w = list()

    for _xval in x_list:
        for _wval in w_list:
            x.fix(_xval)
            if hasattr(m, 'obj'):
                del m.obj
            m.obj = pe.Objective(expr=(w - _wval)**2)
            opt.update_var(x)
            opt.set_objective(m.obj)
            res = opt.solve(load_solutions=False, save_results=False)
            if res.solver.termination_condition != pe.TerminationCondition.optimal:
                raise RuntimeError('Could not produce plot because a feasiblity test failed.')
            opt.load_vars([w])
            if abs(pe.value(w.value) - _wval) <= tol:
                feasible_x.append(_xval)
                feasible_w.append(_wval)

    plt.scatter(feasible_x, feasible_w, 5.0)
    plt.xlabel('x')
    plt.ylabel('y')
    if show_plot:
        plt.show()

    m.del_component(relaxation)
    m.del_component(x)
    m.del_component(w)
    del m
    rel_parent.add_component(rel_name, relaxation)
    x_parent.add_component(x_name, x)
    w_parent.add_component(w_name, w)

    x.unfix()
    x.setlb(orig_xlb)
    x.setub(orig_xub)
    w.setlb(orig_wlb)
    w.setub(orig_wub)
