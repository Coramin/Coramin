import math
from coramin.relaxations.relaxations_base import BaseRelaxationData
import pyomo.environ as pe
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import _BlockData
from pyomo.contrib.appsi.base import (
    Results,
    PersistentSolver,
    Solver,
    MIPSolverConfig,
    TerminationCondition,
    SolutionLoaderBase,
    UpdateConfig,
)
from typing import Tuple, Optional, MutableMapping, Sequence
from pyomo.common.config import ConfigValue, NonNegativeInt, PositiveFloat
import logging
from coramin.relaxations.auto_relax import relax
from coramin.relaxations.iterators import (
    relaxation_data_objects,
    nonrelaxation_component_data_objects,
)
from coramin.utils.coramin_enums import RelaxationSide
from coramin.domain_reduction.dbt import push_integers, pop_integers
import time
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.objective import _GeneralObjectiveData
from coramin.utils.pyomo_utils import get_objective
from pyomo.common.collections.component_set import ComponentSet
from pyomo.common.modeling import unique_component_name


logger = logging.getLogger(__name__)


class MultiTreeConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(MultiTreeConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare("solver_output_logger", ConfigValue())
        self.declare("log_level", ConfigValue(domain=NonNegativeInt))
        self.declare("feasibility_tolerance", ConfigValue(domain=PositiveFloat))
        self.declare("abs_gap", ConfigValue(domain=PositiveFloat))
        self.declare("rel_gap", ConfigValue(domain=PositiveFloat))

        self.solver_output_logger = logger
        self.log_level = logging.INFO
        self.feasibility_tolerance = 1e-6
        self.time_limit = 600
        self.abs_gap = 1e-4
        self.rel_gap = 0.01


def _is_problem_definitely_convex(m: _BlockData) -> bool:
    res = True
    for r in relaxation_data_objects(m, descend_into=True, active=True):
        if r.relaxation_side == RelaxationSide.BOTH:
            res = False
            break
        elif r.relaxation_side == RelaxationSide.UNDER and not r.is_rhs_convex():
            res = False
            break
        elif r.relaxation_side == RelaxationSide.OVER and not r.is_rhs_concave():
            res = False
            break
    return res


class MultiTreeSolutionLoader(SolutionLoaderBase):
    def __init__(self, primals: MutableMapping):
        self._primals = primals

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> MutableMapping[_GeneralVarData, float]:
        if vars_to_load is None:
            return pe.ComponentMap(self._primals.items())
        else:
            primals = pe.ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[v]
        return primals


class MultiTree(Solver):
    def __init__(self, mip_solver: PersistentSolver, nlp_solver: PersistentSolver):
        super(MultiTree, self).__init__()
        self._config = MultiTreeConfig()
        self.mip_solver: PersistentSolver = mip_solver
        self.nlp_solver: PersistentSolver = nlp_solver
        self._original_model: Optional[_BlockData] = None
        self._relaxation: Optional[_BlockData] = None
        self._start_time: Optional[float] = None
        self._incumbent: Optional[pe.ComponentMap] = None
        self._best_feasible_objective: Optional[float] = None
        self._best_objective_bound: Optional[float] = None
        self._objective: Optional[_GeneralObjectiveData] = None
        self._relaxation_objects: Optional[Sequence[BaseRelaxationData]] = None
        self._stop: Optional[TerminationCondition] = None
        self._discrete_vars: Optional[Sequence[_GeneralVarData]] = None
        self._rel_to_nlp_map: Optional[MutableMapping] = None

    def available(self):
        if (
            self.mip_solver.available() == Solver.Availability.FullLicense
            and self.nlp_solver.available() == Solver.Availability.FullLicense
        ):
            return Solver.Availability.FullLicense
        elif self.mip_solver.available() == Solver.Availability.FullLicense:
            return self.nlp_solver.available()
        else:
            return self.mip_solver.available()

    def version(self) -> Tuple:
        return 0, 1, 0

    @property
    def config(self) -> MultiTreeConfig:
        return self._config

    @config.setter
    def config(self, val: MultiTreeConfig):
        self._config = val

    @property
    def symbol_map(self):
        raise NotImplementedError("This solver does not have a symbol map")

    def _should_terminate(self) -> Tuple[bool, Optional[TerminationCondition]]:
        if time.time() - self._start_time >= self.config.time_limit:
            return True, TerminationCondition.maxTimeLimit
        if self._stop is not None:
            return True, self._stop
        abs_gap, rel_gap = self._get_abs_and_rel_gap()
        if abs_gap <= self.config.abs_gap:
            return True, TerminationCondition.optimal
        if rel_gap <= self.config.rel_gap:
            return True, TerminationCondition.optimal
        return False, None

    def _get_results(self, termination_condition: TerminationCondition) -> Results:
        res = Results()
        res.termination_condition = termination_condition
        res.best_feasible_objective = self._get_primal_bound()
        res.best_objective_bound = self._get_dual_bound()
        if self._best_feasible_objective is not None:
            res.solution_loader = MultiTreeSolutionLoader(self._incumbent)
        return res

    def _get_primal_bound(self) -> float:
        if self._best_feasible_objective is None:
            if self._objective.sense == pe.minimize:
                primal_bound = math.inf
            else:
                primal_bound = -math.inf
        else:
            primal_bound = self._best_feasible_objective
        return primal_bound

    def _get_dual_bound(self) -> float:
        if self._best_objective_bound is None:
            if self._objective.sense == pe.minimize:
                dual_bound = -math.inf
            else:
                dual_bound = math.inf
        else:
            dual_bound = self._best_objective_bound
        return dual_bound

    def _get_abs_and_rel_gap(self):
        primal_bound = self._get_primal_bound()
        dual_bound = self._get_dual_bound()
        abs_gap = abs(primal_bound - dual_bound)
        if abs_gap == 0:
            rel_gap = 0
        elif primal_bound == 0:
            rel_gap = math.inf
        elif math.isinf(abs_gap):
            rel_gap = math.inf
        else:
            rel_gap = abs_gap / abs(primal_bound)
        return abs_gap, rel_gap

    def _log(self, header=False):
        logger = self.config.solver_output_logger
        log_level = self.config.log_level
        if header:
            logger.log(
                log_level,
                f"    {'Primal Bound':<15}{'Dual Bound':<15}{'Abs Gap':<15}"
                f"{'Rel Gap':<15}{'Time':<15}",
            )
        else:
            primal_bound = self._get_primal_bound()
            dual_bound = self._get_dual_bound()
            abs_gap, rel_gap = self._get_abs_and_rel_gap()
            elapsed_time = time.time() - self._start_time
            logger.log(
                log_level,
                f"    {primal_bound:<15.3e}{dual_bound:<15.3e}{abs_gap:<15.3e}"
                f"{rel_gap:<15.3f}{elapsed_time:<15.2f}",
            )

    def _update_dual_bound(self, res: Results):
        if res.best_objective_bound is not None:
            if self._objective.sense == pe.minimize:
                if (
                    self._best_objective_bound is None
                    or res.best_objective_bound > self._best_objective_bound
                ):
                    self._best_objective_bound = res.best_objective_bound
            else:
                if (
                    self._best_objective_bound is None
                    or res.best_objective_bound < self._best_objective_bound
                ):
                    self._best_objective_bound = res.best_objective_bound

    def _update_primal_bound(self, res: Results):
        should_update = False
        if res.best_feasible_objective is not None:
            if self._objective.sense == pe.minimize:
                if (
                    self._best_feasible_objective is None
                    or res.best_feasible_objective < self._best_feasible_objective
                ):
                    should_update = True
            else:
                if (
                    self._best_feasible_objective is None
                    or res.best_feasible_objective > self._best_feasible_objective
                ):
                    should_update = True

        if should_update:
            self._best_feasible_objective = res.best_feasible_objective
            self._incumbent = pe.ComponentMap()
            for v in self._rel_to_nlp_map.values():
                self._incumbent[v] = v.value

    def _solve_nlp_with_fixed_vars(
        self, integer_var_values: MutableMapping[_GeneralVarData, float]
    ) -> Results:
        fixed_vars = list()
        for v in self._discrete_vars:
            if v.fixed:
                continue
            val = integer_var_values[v]
            assert math.isclose(val, round(val))
            orig_v = self._rel_to_nlp_map[v]
            orig_v.fix(val)
            fixed_vars.append(orig_v)

        elapsed_time = time.time() - self._start_time
        sub_time_limit = self.config.time_limit - elapsed_time
        self.nlp_solver.config.time_limit = sub_time_limit
        self.nlp_solver.config.load_solution = False
        nlp_res = self.nlp_solver.solve(self._original_model)
        if nlp_res.best_feasible_objective is not None:
            nlp_res.solution_loader.load_vars()
        self._update_primal_bound(nlp_res)
        self._log(header=False)

    def _solve_relaxation(self) -> Results:
        elapsed_time = time.time() - self._start_time
        sub_time_limit = self.config.time_limit - elapsed_time
        self.mip_solver.config.time_limit = sub_time_limit
        self.mip_solver.config.load_solution = False
        rel_res = self.mip_solver.solve(self._relaxation)
        self._update_dual_bound(rel_res)
        self._log(header=False)
        if rel_res.best_feasible_objective is not None:
            rel_res.solution_loader.load_vars()
            all_cons_satisfied = True
            for r in self._relaxation_objects:
                if r.get_deviation() > self.config.feasibility_tolerance:
                    all_cons_satisfied = False
                    break
            if all_cons_satisfied:
                for v in self._discrete_vars:
                    if not math.isclose(v.value, round(v.value)):
                        all_cons_satisfied = False
                        break
            if all_cons_satisfied:
                self._update_primal_bound(rel_res)
                self._log(header=False)
        elif rel_res.termination_condition not in {
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.objectiveLimit,
            TerminationCondition.interrupted,
        }:
            self._stop = rel_res.termination_condition
        return rel_res

    def _add_oa_cuts(self, tol, max_iter) -> Results:
        original_update_config: UpdateConfig = self.mip_solver.update_config()

        self.mip_solver.update()

        self.mip_solver.update_config.update_params = False
        self.mip_solver.update_config.update_vars = False
        self.mip_solver.update_config.update_objective = False
        self.mip_solver.update_config.update_constraints = False
        self.mip_solver.update_config.check_for_new_objective = False
        self.mip_solver.update_config.check_for_new_or_removed_constraints = False
        self.mip_solver.update_config.check_for_new_or_removed_vars = False
        self.mip_solver.update_config.check_for_new_or_removed_params = True
        self.mip_solver.update_config.treat_fixed_vars_as_params = True
        self.mip_solver.update_config.update_named_expressions = False

        last_res = None

        for _iter in range(max_iter):
            if self._should_terminate()[0]:
                break

            rel_res = self._solve_relaxation()
            if rel_res.best_feasible_objective is not None:
                last_res = Results()
                last_res.best_feasible_objective = rel_res.best_feasible_objective
                last_res.best_objective_bound = rel_res.best_objective_bound
                last_res.termination_condition = rel_res.termination_condition
                last_res.solution_loader = MultiTreeSolutionLoader(
                    rel_res.solution_loader.get_primals(
                        vars_to_load=self._discrete_vars
                    )
                )

            if self._should_terminate()[0]:
                break

            new_con_list = list()
            for b in self._relaxation_objects:
                new_con = b.add_cut(
                    keep_cut=True, check_violation=True, feasibility_tol=tol
                )
                if new_con is not None:
                    new_con_list.append(new_con)
            if len(new_con_list) == 0:
                break
            self.mip_solver.add_constraints(new_con_list)

        self.mip_solver.update_config.update_params = (
            original_update_config.update_params
        )
        self.mip_solver.update_config.update_vars = original_update_config.update_vars
        self.mip_solver.update_config.update_objective = (
            original_update_config.update_objective
        )
        self.mip_solver.update_config.update_constraints = (
            original_update_config.update_constraints
        )
        self.mip_solver.update_config.check_for_new_objective = (
            original_update_config.check_for_new_objective
        )
        self.mip_solver.update_config.check_for_new_or_removed_constraints = (
            original_update_config.check_for_new_or_removed_constraints
        )
        self.mip_solver.update_config.check_for_new_or_removed_vars = (
            original_update_config.check_for_new_or_removed_vars
        )
        self.mip_solver.update_config.check_for_new_or_removed_params = (
            original_update_config.check_for_new_or_removed_params
        )
        self.mip_solver.update_config.treat_fixed_vars_as_params = (
            original_update_config.treat_fixed_vars_as_params
        )
        self.mip_solver.update_config.update_named_expressions = (
            original_update_config.update_named_expressions
        )

        if last_res is None:
            last_res = Results()

        return last_res

    def _construct_relaxation(self):
        all_vars = list(
            ComponentSet(
                self._original_model.component_data_objects(pe.Var, descend_into=True)
            )
        )
        tmp_name = unique_component_name(self._original_model, "all_vars")
        setattr(self._original_model, tmp_name, all_vars)
        self._relaxation = relax(
            model=self._original_model,
            in_place=False,
            use_fbbt=True,
            fbbt_options={"deactivate_satisfied_constraints": True, "max_iter": 2},
        )
        new_vars = getattr(self._relaxation, tmp_name)
        self._rel_to_nlp_map = pe.ComponentMap(zip(new_vars, all_vars))
        delattr(self._original_model, tmp_name)
        delattr(self._relaxation, tmp_name)

    def solve(self, model: _BlockData, timer: HierarchicalTimer = None) -> Results:
        self._start_time = time.time()
        if timer is None:
            timer = HierarchicalTimer()
        timer.start("solve")

        self._original_model = model

        self._log(header=True)

        timer.start("construct relaxation")
        self._construct_relaxation()
        timer.stop("construct relaxation")

        self._objective = get_objective(self._relaxation)
        self._relaxation_objects = list()
        for r in relaxation_data_objects(
            self._relaxation, descend_into=True, active=True
        ):
            self._relaxation_objects.append(r)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        self._log(header=False)

        self.mip_solver.set_instance(self._relaxation)

        relaxed_binaries, relaxed_integers = push_integers(self._relaxation)
        self._discrete_vars = list(relaxed_binaries) + list(relaxed_integers)
        oa_results = self._add_oa_cuts(self.config.feasibility_tolerance * 100, 100)
        pop_integers(relaxed_binaries, relaxed_integers)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        if _is_problem_definitely_convex(self._relaxation):
            oa_results = self._add_oa_cuts(self.config.feasibility_tolerance, 100)
        else:
            oa_results = self._add_oa_cuts(self.config.feasibility_tolerance * 1e3, 3)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        if oa_results.best_feasible_objective is not None:
            integer_var_values = pe.ComponentMap()
            for v in self._discrete_vars:
                integer_var_values[v] = v.value
            nlp_res = self._solve_nlp_with_fixed_vars(integer_var_values)

        timer.stop("solve")

        return self._get_results(TerminationCondition.unknown)
