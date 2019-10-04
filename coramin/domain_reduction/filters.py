from pyomo.core.kernel.component_set import ComponentSet


def filter_variables_from_solution(candidate_variables_at_relaxation_solution, tolerance=0):
    """
    This function takes a set of candidate variables for OBBT and filters out 
    the variables that are at their bounds in the provided solution to the 
    relaxation. See 

    Gleixner, Ambros M., et al. "Three enhancements for
    optimization-based bound tightening." Journal of Global
    Optimization 67.4 (2017): 731-757.

    for details on why this works. The basic idea is that if x = xl is
    feasible for the relaxation that will be used for OBBT, then
    minimizing x subject to that relaxation is guaranteed to result in
    an optimal solution of x* = xl.

    This function simply loops through
    candidate_variables_at_relaxation_solution and specifies which
    variables should be minimized and which variables should be
    maximized with OBBT.

    Parameters
    ----------
    candidate_variables_at_relaxation_solution: iterable of _GeneralVarData
        This should be an iterable of the variables which are candidates 
        for OBBT. The values of the variables should be feasible for the 
        relaxation that would be used to perform OBBT on the variables.
    tolerance: float
        A float greater than or equal to zero. If the value of the variable
        is within tolerance of its lower bound, then that variable is filtered
        from the set of variables that should be minimized for OBBT. The same
        is true for upper bounds and variables that should be maximized.

    Returns
    -------
    vars_to_minimize: list of _GeneralVarData
        variables that should be considered for minimization
    vars_to_maximize: list of _GeneralVarData
        variables that should be considered for maximization
    """
    candidate_vars = ComponentSet(candidate_variables_at_relaxation_solution)
    vars_to_minimize = list()
    vars_to_maximize = list()

    for v in candidate_vars:
        if (not v.has_lb()) or (v.value - v.lb > tolerance):
            vars_to_minimize.append(v)
        if (not v.has_ub()) or (v.ub - v.value > tolerance):
            vars_to_maximize.append(v)

    return vars_to_minimize, vars_to_maximize


def aggresive_filter(candidate_variables, relaxation, solver, tolerance=0):
    """
    This function takes a set of candidate variables for OBBT and filters out 
    the variables for which it does not make senese to perform OBBT on. See 

    Gleixner, Ambros M., et al. "Three enhancements for
    optimization-based bound tightening." Journal of Global
    Optimization 67.4 (2017): 731-757.

    for details. The basic idea is that if x = xl is
    feasible for the relaxation that will be used for OBBT, then
    minimizing x subject to that relaxation is guaranteed to result in
    an optimal solution of x* = xl.

    This function solves a series of optimization problems to try to 
    filter as many variables as possible.

    Parameters
    ----------
    candidate_variables_at_relaxation_solution: iterable of _GeneralVarData
        This should be an iterable of the variables which are candidates 
        for OBBT.
    relaxation: Block
        a convex relaxation
    solver: solver
    tolerance: float
        A float greater than or equal to zero. If the value of the variable
        is within tolerance of its lower bound, then that variable is filtered
        from the set of variables that should be minimized for OBBT. The same
        is true for upper bounds and variables that should be maximized.

    Returns
    -------
    vars_to_minimize: list of _GeneralVarData
        variables that should be considered for minimization
    vars_to_maximize: list of _GeneralVarData
        variables that should be considered for maximization
    """
    vars_to_minimize = ComponentSet(candidate_variables_at_relaxation_solution)
    vars_to_maximize = ComponentSet(candidate_variables_at_relaxation_solution)

    

    return vars_to_minimize, vars_to_maximize


