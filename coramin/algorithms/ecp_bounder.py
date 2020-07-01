from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.kernel.component_set import ComponentSet
from coramin.relaxations.relaxations_base import BaseRelaxationData, BaseRelaxation
import pyomo.environ as pe
import time
from pyomo.opt.results.results_ import SolverResults
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeInt, NonNegativeFloat, In


import logging
logger = logging.getLogger(__name__)


class _ECPBounder(OptSolver):
    """
    This class is used for testing
    """
    def __init__(self, subproblem_solver):
        self._subproblem_solver = pe.SolverFactory(subproblem_solver)
        if isinstance(self._subproblem_solver, PersistentSolver):
            self._using_persistent_solver = True
        else:
            self._using_persistent_solver = False
        self._relaxations = ComponentSet()
        self._relaxations_not_tracking_solver = ComponentSet()
        self._relaxations_with_added_cuts = ComponentSet()
        self._pyomo_model = None

        self.options = ConfigBlock()
        self.options.declare('feasibility_tol', ConfigValue(default=1e-6, domain=NonNegativeFloat,
                                                            doc='Tolerance below which cuts will not be added'))
        self.options.declare('max_iter', ConfigValue(default=30, domain=NonNegativeInt,
                                                     doc='Maximum number of iterations'))
        self.options.declare('keep_cuts', ConfigValue(default=False, domain=In([True, False]),
                                                      doc='Whether or not to keep the cuts generated after the solve'))
        self.options.declare('time_limit', ConfigValue(default=float('inf'), domain=NonNegativeFloat,
                                                       doc='Time limit in seconds'))

        self.subproblem_solver_options = ConfigBlock(implicit=True)

    def solve(self, *args, **kwargs):
        t0 = time.time()
        logger.info('{0:<10}{1:<12}{2:<12}{3:<12}{4:<12}'.format('Iter', 'objective', 'max_viol', 'time', '# cuts'))

        if not self._using_persistent_solver:
            self._pyomo_model = args[0]

        options = self.options(kwargs.pop('options', dict()))
        subproblem_solver_options = dict()
        for k, v in self.subproblem_solver_options.items():
            subproblem_solver_options[k] = v
        for k, v in kwargs.pop('subproblem_solver_options', dict()).items():
            subproblem_solver_options[k] = v

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
        self._relaxations_with_added_cuts = ComponentSet()
        for b in self._pyomo_model.component_data_objects(pe.Block, descend_into=True, active=True, sort=True):
            if isinstance(b, (BaseRelaxationData, BaseRelaxation)):
                self._relaxations.add(b)
                if self._using_persistent_solver:
                    if self not in b._persistent_solvers:
                        b.add_persistent_solver(self)
                        self._relaxations_not_tracking_solver.add(b)

        for _iter in range(options.max_iter):
            if time.time() - t0 > options.time_limit:
                final_res.solver.termination_condition = pe.TerminationCondition.maxTimeLimit
                final_res.solver.status = pe.SolverStatus.aborted
                logger.warning('ECPBounder: time limit reached.')
                break
            if self._using_persistent_solver:
                res = self._subproblem_solver.solve(save_results=False, options=subproblem_solver_options)
            else:
                res = self._subproblem_solver.solve(self._pyomo_model, options=subproblem_solver_options)
            if res.solver.termination_condition != pe.TerminationCondition.optimal:
                final_res.solver.termination_condition = pe.TerminationCondition.other
                final_res.solver.status = pe.SolverStatus.aborted
                logger.warning('ECPBounder: subproblem did not terminate optimally')
                break

            num_cuts_added = 0
            max_viol = 0
            for b in self._relaxations:
                viol = None
                try:
                    if b.is_rhs_convex():
                        viol = pe.value(b.get_rhs_expr()) - b.get_aux_var().value
                    elif b.is_rhs_concave():
                        viol = b.get_aux_var().value - pe.value(b.get_rhs_expr())
                except (OverflowError, ZeroDivisionError, ValueError) as err:
                    logger.warning('could not generate ECP cut due to ' + str(err))
                if viol is not None:
                    if viol > max_viol:
                        max_viol = viol
                    if viol > options.feasibility_tol:
                        b.add_cut(keep_cut=options.keep_cuts)
                        self._relaxations_with_added_cuts.add(b)
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
            logger.info('{0:<10d}{1:<12.3e}{2:<12.3e}{3:<12.3e}{4:<12d}'.format(_iter, obj_val, max_viol, elapsed_time, num_cuts_added))

            if num_cuts_added == 0:
                final_res.solver.termination_condition = pe.TerminationCondition.optimal
                final_res.solver.status = pe.SolverStatus.ok
                logger.info('ECPBounder: converged!')
                break

            if _iter == options.max_iter - 1:
                final_res.solver.termination_condition = pe.TerminationCondition.maxIterations
                final_res.solver.status = pe.SolverStatus.aborted
                logger.warning('ECPBounder: reached maximum number of iterations')

        if not options.keep_cuts:
            for b in self._relaxations_with_added_cuts:
                b.rebuild()

        if self._using_persistent_solver:
            for b in self._relaxations_not_tracking_solver:
                b.remove_persistent_solver(self)

        final_res.solver.wallclock_time = time.time() - t0
        return final_res


class ECPBounder(_ECPBounder, PersistentSolver):
    """
    A solver designed for use inside of OBBT. This solver is a persistent solver for efficient changes to the
    objective. Additionally, it provides a mechanism for refining convex nonlinear constraints during OBBT.
    """
    def __init__(self, subproblem_solver):
        super(ECPBounder, self).__init__(subproblem_solver=subproblem_solver)
        if not self._using_persistent_solver:
            raise ValueError('subproblem solver must be a persistent solver')

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

    def load_vars(self, vars_to_load=None):
        self._subproblem_solver.load_vars(vars_to_load=vars_to_load)

    def load_duals(self, cons_to_load=None):
        self._subproblem_solver.load_duals(cons_to_load=cons_to_load)
