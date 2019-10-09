from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.kernel.component_set import ComponentSet
from coramin.relaxations.relaxations_base import BaseRelaxationData, BaseRelaxation
import pyomo.environ as pe
import time
from pyomo.opt.results.results_ import SolverResults


import logging
logger = logging.getLogger(__name__)


class ConvexOuterApproximationSolver(PersistentSolver):
    def __init__(self, subproblem_solver='gurobi_persistent'):
        self._subproblem_solver = pe.SolverFactory(subproblem_solver)
        if not isinstance(self._subproblem_solver, PersistentSolver):
            raise ValueError('subproblem_solver must be a persistent solver')
        self._relaxations = ComponentSet()
        self._relaxations_not_tracking_solver = ComponentSet()
        self.feasibility_tol = 1e-6
        self._pyomo_model = None
        self.max_iter = 30

    def set_instance(self, model, **kwargs):
        self._pyomo_model = model
        self._subproblem_solver.set_instance(model, **kwargs)

    def add_block(self, block):
        self._subproblem_solver.add_block(block)

    def add_constraint(self, con):
        self._subproblem_solver.add_constraint(con)

    def add_var(self, var):
        self._subproblem_solver.add_var(var)

    def remove_block(self, block):
        self._subproblem_solver.remove_block(block)

    def remove_constraint(self, con):
        self._subproblem_solver.remove_constraint(con)

    def remove_var(self, var):
        self._subproblem_solver.remove_var(var)

    def set_objective(self, obj):
        self._subproblem_solver.set_objective(obj)

    def update_var(self, var):
        self._subproblem_solver.update_var(var)

    def solve(self, *args, **kwargs):
        t0 = time.time()
        logger.info('{0:<10}{1:<12}{2:<12}{3:<12}'.format('Iter', 'objective', 'max_viol', 'time'))

        obj = None
        for _obj in self._pyomo_model.component_data_objects(pe.Objective, descend_into=True, active=True, sort=True):
            if obj is not None:
                raise ValueError('Found multiple active objectives')
            obj = _obj
        if obj is None:
            raise ValueError('Could not find any active objectives')

        final_res = SolverResults()

        self._relaxations = ComponentSet()
        self._relaxations_not_tracking_solver = ComponentSet()
        for b in self._pyomo_model.component_data_objects(pe.Block, descend_into=True, active=True, sort=True):
            if isinstance(b, (BaseRelaxationData, BaseRelaxation)):
                self._relaxations.add(b)
                if self not in b._persistent_solvers:
                    b.add_persistent_solver(self)
                    self._relaxations_not_tracking_solver.add(b)

        for _iter in range(self.max_iter):
            res = self._subproblem_solver.solve(save_results=False)
            if res.solver.termination_condition != pe.TerminationCondition.optimal:
                final_res.solver.termination_condition = pe.TerminationCondition.other
                final_res.solver.status = pe.SolverStatus.aborted
                break

            num_cuts_added = 0
            max_viol = 0
            for b in self._relaxations:
                viol = None
                if b.is_rhs_convex():
                    viol = pe.value(b.get_rhs_expr()) - b.get_aux_var().value
                elif b.is_rhs_concave():
                    viol = b.get_aux_var().value - pe.value(b.get_rhs_expr())
                if viol is not None:
                    if viol > max_viol:
                        max_viol = viol
                    if viol > self.feasibility_tol:
                        b.add_cut()
                        num_cuts_added += 1

            if obj.sense == pe.minimize:
                obj_val = res.problem.lower_bound
                final_res.problem.sense = pe.minimize
                final_res.problem.upper_bound = None
                final_res.problem.lower_bound = obj_val
            else:
                obj_val = res.problem.upper_bound
                final_res.problem.sense = pe.maximize
                final_res.problem.lower_bound = None
                final_res.problem.upper_bound = obj_val
            elapsed_time = time.time() - t0
            logger.info('{0:<10d}{1:<12.3e}{2:<12.3e}{3:<12.3e}'.format(_iter, obj_val, max_viol, elapsed_time))

            if num_cuts_added == 0:
                final_res.solver.termination_condition = pe.TerminationCondition.optimal
                final_res.solver.status = pe.SolverStatus.ok
                break

            if _iter == self.max_iter - 1:
                final_res.solver.termination_condition = pe.TerminationCondition.maxIterations
                final_res.solver.status = pe.SolverStatus.aborted

        for b in self._relaxations_not_tracking_solver:
            b.remove_persistent_solver(self)

        final_res.solver.wallclock_time = time.time() - t0
        return final_res

    def load_vars(self, vars_to_load=None):
        self._subproblem_solver.load_vars(vars_to_load=vars_to_load)

    def load_duals(self, cons_to_load=None):
        self._subproblem_solver.load_duals(cons_to_load=cons_to_load)
