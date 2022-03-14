import math

import coramin
from coramin.third_party.minlplib_tools import get_minlplib, get_minlplib_instancedata
import unittest
from pyomo.contrib import appsi
import os
import logging
from suspect.pyomo.osil_reader import read_osil
import math
from pyomo.common import download
import pyomo.environ as pe
from pyomo.core.base.block import _BlockData


def _get_sol(pname):
    current_dir = os.getcwd()
    target_fname = os.path.join(current_dir, f'{pname}.sol')
    downloader = download.FileDownloader()
    downloader.set_destination_filename(target_fname)
    downloader.get_binary_file(f'http://www.minlplib.org/sol/{pname}.p1.sol')
    res = dict()
    f = open(target_fname, 'r')
    for line in f.readlines():
        l = line.split()
        vname = l[0]
        vval = float(l[1])
        if vname != 'objvar':
            res[vname] = vval
    f.close()
    return res


class TestMultiTree(unittest.TestCase):
    def setUp(self) -> None:
        self.test_problems = {'batch': 285506.5082,
                              'ball_mk3_10': None,
                              'ball_mk2_10': 0,
                              'syn05m': 837.73240090,
                              'autocorr_bern20-03': -72,
                              'alkyl': -1.76499965}
        self.primal_sol = dict()
        self.primal_sol['batch'] = _get_sol('batch')
        self.primal_sol['alkyl'] = _get_sol('alkyl')
        self.primal_sol['ball_mk2_10'] = _get_sol('ball_mk2_10')
        self.primal_sol['syn05m'] = _get_sol('syn05m')
        self.primal_sol['autocorr_bern20-03'] = _get_sol('autocorr_bern20-03')
        for pname in self.test_problems.keys():
            get_minlplib(problem_name=pname)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        self.opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                                nlp_solver=nlp_solver)

    def tearDown(self) -> None:
        current_dir = os.getcwd()
        for pname in self.test_problems.keys():
            os.remove(os.path.join(current_dir, 'minlplib', 'osil', f'{pname}.osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib', 'osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))
        for pname in self.primal_sol.keys():
            os.remove(os.path.join(current_dir, f'{pname}.sol'))

    def get_model(self, pname):
        current_dir = os.getcwd()
        fname = os.path.join(current_dir, 'minlplib', 'osil', f'{pname}.osil')
        m = read_osil(fname, objective_prefix='obj_')
        return m

    def _check_relative_diff(self, expected, got, abs_tol=1e-3, rel_tol=1e-3):
        abs_diff = abs(expected - got)
        if expected == 0:
            rel_diff = math.inf
        else:
            rel_diff = abs_diff / abs(expected)
        success = abs_diff <= abs_tol or rel_diff <= rel_tol
        self.assertTrue(success, msg=f'\n    expected: {expected}\n    got: {got}\n    abs diff: {abs_diff}\n    rel diff: {rel_diff}')

    def _check_primal_sol(self, pname, m: _BlockData, res: appsi.base.Results):
        expected_by_str = self.primal_sol[pname]
        expected_by_var = pe.ComponentMap()
        for vname, vval in expected_by_str.items():
            v = getattr(m, vname)
            expected_by_var[v] = vval
        got = res.solution_loader.get_primals()
        for v, val in expected_by_var.items():
            self._check_relative_diff(val, got[v])
        got = res.solution_loader.get_primals(vars_to_load=list(expected_by_var.keys()))
        for v, val in expected_by_var.items():
            self._check_relative_diff(val, got[v])

    def optimal_helper(self, pname, check_primal_sol=True):
        m = self.get_model(pname)
        res = self.opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self._check_relative_diff(self.test_problems[pname], res.best_feasible_objective)
        self._check_relative_diff(self.test_problems[pname], res.best_objective_bound)
        if check_primal_sol:
            self._check_primal_sol(pname, m, res)

    def infeasible_helper(self, pname):
        m = self.get_model(pname)
        res = self.opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.infeasible)

    def time_limit_helper(self, pname):
        orig_time_limit = self.opt.config.time_limit
        for new_limit in [0.1, 0.2, 0.3]:
            self.opt.config.time_limit = new_limit
            m = self.get_model(pname)
            res = self.opt.solve(m)
            self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.maxTimeLimit)

    def test_batch(self):
        self.optimal_helper('batch')

    def test_ball_mk2_10(self):
        self.optimal_helper('ball_mk2_10')

    def test_alkyl(self):
        self.optimal_helper('alkyl')

    def test_syn05m(self):
        self.optimal_helper('syn05m')

    def test_autocorr_bern20_03(self):
        self.optimal_helper('autocorr_bern20-03', check_primal_sol=False)

    def test_time_limit(self):
        self.time_limit_helper('alkyl')

    def test_ball_mk3_10(self):
        self.infeasible_helper('ball_mk3_10')

    def test_available(self):
        avail = self.opt.available()
        assert avail in appsi.base.Solver.Availability
