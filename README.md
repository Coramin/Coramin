# Coramin

[![codecov](https://codecov.io/gh/Coramin/Coramin/branch/main_branch/graph/badge.svg)](https://codecov.io/gh/Coramin/Coramin)
[![Actions Status](https://github.com/Coramin/Coramin/workflows/main_ci/badge.svg?branch=main_branch)](https://github.com/Coramin/Coramin/actions)

Coramin is a Pyomo-based Python package that provides tools for
developing tailored algorithms for mixed-integer nonlinear programming
problems (MINLP's). This software includes classes for managing and
refining convex relaxations of nonconvex constraints. These classes
provide methods for updating the relaxation based on new variable
bounds, creating and managing piecewise relaxations (for multi-tree
based algorithms), and adding outer-approximation based cuts for
convex or concave constraints. These classes inherit from Pyomo
Blocks, so they can be easily integrated with Pyomo
models. Additionally, Coramin has functions for automatically
generating convex relaxations of general Pyomo models. Coramin also
has tools for domain reduction, including a parallel implementation
of optimization-based bounds tightening (OBBT) and various OBBT
filtering techniques.

Contributors
------------
Michael Bynum
- Relaxation classes
- OBBT
- OBBT Filtering
- Factorable programming approach to generating relaxations

Carl Laird
- Parallel OBBT
- McCormick and piecewise McCormick relaxations for bilinear terms
- Relaxations for univariate convex/concave fucntions

Anya Castillo
- Relaxation classes

Francesco Ceccon
- Alpha-BB relaxation