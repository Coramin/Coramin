import pyomo.environ as pyo
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition as TC
import warnings
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.solvers.plugins.solvers.GUROBI import GUROBISHELL
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.solvers.plugins.solvers.CPLEX import CPLEXSHELL
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.cplex_persistent import CPLEXPersistent
from pyomo.solvers.plugins.solvers.GLPK import GLPKSHELL
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import logging
import traceback
import numpy as np
try:
    import coramin.utils.mpi_utils as mpiu
    mpi_available = True
except ImportError:
    mpi_available = False


logger = logging.getLogger(__name__)


_mip_solver_types = {GUROBISHELL, GurobiDirect, GurobiPersistent, CPLEXSHELL, CPLEXDirect, CPLEXPersistent, GLPKSHELL}
_acceptable_termination_conditions = {TC.optimal, TC.globallyOptimal}
_acceptable_solver_status = {SolverStatus.ok}


def _bt_cleanup(model, solver, vardatalist, initial_var_values, deactivated_objectives, lower_bounds=None, upper_bounds=None):
    """
    Cleanup the changes made to the model during bounds tightening.
    Reactivate any deactivated objectives.
    Remove an objective upper bound constraint if it was added.
    If lower_bounds or upper_bounds is provided, update the bounds of the variables in self.vars_to_tighten.

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
    solver: pyomo solver object
    vardatalist: list of pyo.Var
    initial_var_values: ComponentMap
    deactivated_objectives: list of pyo.Objective
    lower_bounds: list of float
        Only needed if you want to update the bounds of the variables. Should be in the same order as
        self.vars_to_tighten.
    upper_bounds: list of float
        Only needed if you want to update the bounds of the variables. Should be in the same order as
        self.vars_to_tighten.
    """
    for v in model.component_data_objects(ctype=pyo.Var, active=None, sort=True, descend_into=True):
        v.value = initial_var_values[v]

    # remove the obj upper bound constraint
    using_persistent_solver = False
    if hasattr(model, '__objective_ineq'):
        if isinstance(solver, PersistentSolver):
            solver.remove_constraint(model.__objective_ineq)
            using_persistent_solver = True
        del model.__objective_ineq

    # reactivate the objectives that we deactivated
    for obj in deactivated_objectives:
        obj.activate()
        if using_persistent_solver:
            solver.set_objective(obj)

    if lower_bounds is not None and upper_bounds is not None:
        for i, v in enumerate(vardatalist):
            lb = lower_bounds[i]
            ub = upper_bounds[i]
            v.setlb(lb)
            v.setub(ub)
            if using_persistent_solver:
                solver.update_var(v)
    elif lower_bounds is not None:
        for i, v in enumerate(vardatalist):
            lb = lower_bounds[i]
            v.setlb(lb)
            if using_persistent_solver:
                solver.update_var(v)
    elif upper_bounds is not None:
        for i, v in enumerate(vardatalist):
            ub = upper_bounds[i]
            v.setub(ub)
            if using_persistent_solver:
                solver.update_var(v)


def _tighten_bnds(model, solver, vardatalist, lb_or_ub):
    """
    Tighten the lower bounds of all variables in vardatalist (or self.vars_to_tighten if vardatalist is None).

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
    solver: pyomo solver object
    vardatalist: list of _GeneralVarData
    lb_or_ub: str
        'lb' or 'ub'

    Returns
    -------
    new_bounds: list of float
    """
    # solve for the new bounds
    new_bounds = list()
    for v in vardatalist:
        # solve for lower var bound
        if lb_or_ub == 'lb':
            model.__obj_bounds_tightening = pyo.Objective(expr=v, sense=pyo.minimize)
        else:
            assert lb_or_ub == 'ub'
            model.__obj_bounds_tightening = pyo.Objective(expr=-v, sense=pyo.minimize)
        if isinstance(solver, PersistentSolver):
            solver.set_objective(model.__obj_bounds_tightening)
            results = solver.solve(tee=False, load_solutions=False, save_results=False)
            if ((results.solver.status in _acceptable_solver_status) and
                    (results.solver.termination_condition in _acceptable_termination_conditions)):
                if type(solver) in _mip_solver_types:
                    if lb_or_ub == 'lb':
                        new_bnd = results.problem.lower_bound
                    else:
                        new_bnd = -results.problem.lower_bound
                else:
                    solver.load_vars([v])
                    new_bnd = pyo.value(v.value)
            else:
                if lb_or_ub == 'lb':
                    new_bnd = pyo.value(v.lb)
                else:
                    new_bnd = pyo.value(v.ub)
                msg = 'Warning: Bounds tightening for lb for var {0} was unsuccessful. The lb was not changed.'.format(
                    v)
                warnings.warn(msg)
                logger.warning(msg)
        else:
            results = solver.solve(model, tee=False, load_solutions=False)
            if ((results.solver.status in _acceptable_solver_status) and
                    (results.solver.termination_condition in _acceptable_termination_conditions)):
                if type(solver) in _mip_solver_types:
                    if lb_or_ub == 'lb':
                        new_bnd = results.problem.lower_bound
                    else:
                        new_bnd = -results.problem.lower_bound
                else:
                    model.solutions.load_from(results)
                    new_bnd = pyo.value(v.value)
            else:
                if lb_or_ub == 'lb':
                    new_bnd = pyo.value(v.lb)
                else:
                    new_bnd = pyo.value(v.ub)
                msg = 'Warning: Bounds tightening for lb for var {0} was unsuccessful. The lb was not changed.'.format(
                    v)
                warnings.warn(msg)
                logger.warning(msg)
        if lb_or_ub == 'lb':
            if new_bnd < pyo.value(v.lb) or new_bnd is None:
                new_bnd = pyo.value(v.lb)
        else:
            if new_bnd > pyo.value(v.ub) or new_bnd is None:
                new_bnd = pyo.value(v.ub)
        new_bounds.append(new_bnd)

        # remove the objective function
        del model.__obj_bounds_tightening

    return new_bounds


def _bt_prep(model, solver, objective_ub=None):
    """
    Prepare the model for bounds tightening.
    Gather the variable values to load back in after bounds tightening.
    Deactivate any active objectives.
    If objective_ub is not None, then add a constraint forcing the objective to be less than objective_ub

    Parameters
    ----------
    model : pyo.ConcreteModel or pyo.Block
        The model object that will be used for bounds tightening.
    objective_ub : float
        The objective value for the current best upper bound incumbent

    Returns
    -------
    initial_var_values: ComponentMap
    deactivated_objectives: list
    """
    if isinstance(solver, PersistentSolver):
        solver.set_instance(model)

    initial_var_values = ComponentMap()
    for v in model.component_data_objects(ctype=pyo.Var, active=None, sort=True, descend_into=True):
        initial_var_values[v] = v.value

    deactivated_objectives = list()
    for obj in model.component_data_objects(pyo.Objective, active=True, sort=True, descend_into=True):
        deactivated_objectives.append(obj)
        obj.deactivate()

    # add inequality bound on objective functions if required
    # obj.expr <= objective_ub
    if objective_ub is not None:
        if len(deactivated_objectives) != 1:
            e = 'BoundsTightener: When providing objective_ub,' + \
                ' the model must have one and only one objective function.'
            logger.error(e)
            raise ValueError(e)
        model.__objective_ineq = \
            pyo.Constraint(expr=deactivated_objectives[0].expr <= objective_ub)
        if isinstance(solver, PersistentSolver):
            solver.add_constraint(model.__objective_ineq)

    return initial_var_values, deactivated_objectives


def _build_vardatalist(model, varlist=None):
    """
    Convert a list of pyomo variables to a list of SimpleVar and _GeneralVarData. If varlist is none, builds a
    list of all variables in the model. The new list is stored in the vars_to_tighten attribute.

    Parameters
    ----------
    model: ConcreteModel
    varlist: None or list of pyo.Var
    """
    vardatalist = None

    # if the varlist is None, then assume we want all the active variables
    if varlist is None:
        raise NotImplementedError('Still need to do this.')
    elif isinstance(varlist, pyo.Var):
        # user provided a variable, not a list of variables. Let's work with it anyway
        varlist = [varlist]

    if vardatalist is None:
        # expand any indexed components in the list to their
        # component data objects
        vardatalist = list()
        for v in varlist:
            if v.is_indexed():
                vardatalist.extend(v.values())
            else:
                vardatalist.append(v)

    # remove from vardatalist if the variable is fixed (maybe there is a better way to do this)
    corrected_vardatalist = []
    for v in vardatalist:
        if not v.is_fixed():
            if v.has_lb() and v.has_ub():
                if v.ub - v.lb < 1e-6:
                    e = 'Warning: ub - lb is less than 1e-6: {0}, lb: {1}, ub: {2}'.format(v, v.lb, v.ub)
                    logger.warning(e)
                    warnings.warn(e)
            corrected_vardatalist.append(v)

    return corrected_vardatalist


def perform_obbt(model, solver, varlist=None, objective_ub=None, update_bounds=True):
    """
    Perform optimization-based bounds tighening on the variables in varlist subject to the constraints in model.

    Parameters
    ----------
    model: pyo.ConcreteModel or pyo.Block
        The model to be used for bounds tightening
    solver: pyomo solver object
        The solver to be used for bounds tightening.
    varlist: list of pyo.Var
        The variables for which OBBT should be performed. If varlist is None, then we attempt to automatically
        detect which variables need tightened.
    objective_ub: float
        An upper bound on the objective. If this is not None, then a constraint will be added to the
        bounds tightening problems constraining the objective to be less than objective_ub.
    update_bounds: bool
        If True, then the variable bounds will be updated

    Returns
    -------
    lower_bounds: list of float
    upper_bounds: list of float

    """
    initial_var_values, deactivated_objectives = _bt_prep(model=model, solver=solver, objective_ub=objective_ub)

    vardata_list = _build_vardatalist(model=model, varlist=varlist)
    if mpi_available:
        mpi_interface = mpiu.MPIInterface()
        alloc_map = mpiu.MPIAllocationMap(mpi_interface, len(vardata_list))
        local_vardata_list = alloc_map.local_list(vardata_list)
    else:
        local_vardata_list = vardata_list

    exc = None
    try:
        local_lower_bounds = _tighten_bnds(model=model, solver=solver, vardatalist=local_vardata_list, lb_or_ub='lb')
        local_upper_bounds = _tighten_bnds(model=model, solver=solver, vardatalist=local_vardata_list, lb_or_ub='ub')
        status = 1
        msg = None
    except Exception as err:
        exc = err
        tb = traceback.format_exc()
        status = 0
        msg = str(tb)

    if mpi_available:
        local_status = np.array([status], dtype='i')
        global_status = np.array([0 for i in range(mpiu.MPI.COMM_WORLD.Get_size())], dtype='i')
        mpiu.MPI.COMM_WORLD.Allgatherv([local_status, mpiu.MPI.INT], [global_status, mpiu.MPI.INT])
        if not np.all(global_status):
            messages = mpi_interface.comm.allgather(msg)
            msg = None
            for m in messages:
                if m is not None:
                    msg = m
            logger.error('An error was raised in one or more processes:\n' + msg)
            raise mpiu.MPISyncError('An error was raised in one or more processes:\n' + msg)
    else:
        if status != 1:
            logger.error('An error was raised during OBBT:\n' + msg)
            raise exc

    if mpi_available:
        global_lower = alloc_map.global_list_float64(local_lower_bounds)
        global_upper = alloc_map.global_list_float64(local_upper_bounds)
    else:
        global_lower = local_lower_bounds
        global_upper = local_upper_bounds

    _lower_bounds = None
    _upper_bounds = None
    if update_bounds:
        _lower_bounds = global_lower
        _upper_bounds = global_upper
    _bt_cleanup(model=model, solver=solver, vardatalist=vardata_list, initial_var_values=initial_var_values,
                deactivated_objectives=deactivated_objectives, lower_bounds=_lower_bounds, upper_bounds=_upper_bounds)
    return global_lower, global_upper


