import pyomo.environ as pe
from pyomo.core.expr import numeric_expr
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import networkx


def split_expr(expr):
    if not isinstance(expr, numeric_expr.SumExpression):
        return [expr]

    vars_by_arg = pe.ComponentMap()
    args_by_var = pe.ComponentMap()
    all_vars = ComponentSet()
    for arg in expr.args:
        vlist = list(identify_variables(arg, include_fixed=False))
        all_vars.update(vlist)
        vars_by_arg[arg] = vlist
        for v in vlist:
            if v not in args_by_var:
                args_by_var[v] = list()
            args_by_var[v].append(arg)

    var_ids = {id(v): v for v in all_vars}
    graph = networkx.Graph()
    graph.add_nodes_from(var_ids.keys())

    for arg, vlist in vars_by_arg.items():
        for ndx1 in range(len(vlist)):
            for ndx2 in range(ndx1, len(vlist)):
                v1 = vlist[ndx1]
                v2 = vlist[ndx2]
                graph.add_edge(id(v1), id(v2))

    list_of_exprs = list()
    for cc in networkx.connected_components(graph):
        cc_args = ComponentSet()
        for vid in cc:
            v = var_ids[vid]
            for arg in args_by_var[v]:
                cc_args.add(arg)
        list_of_exprs.append(sum(cc_args))

    return list_of_exprs
