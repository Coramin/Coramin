import pyomo.environ as pe
import coramin
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr


nlp = pe.ConcreteModel()
nlp.x = pe.Var(bounds=(-2, 2))
nlp.obj = pe.Objective(expr=nlp.x**4 - 3*nlp.x**2 + nlp.x)
opt = pe.SolverFactory('ipopt')
res = opt.solve(nlp)
ub = pe.value(nlp.obj)

rel = pe.ConcreteModel()
rel.x = pe.Var(bounds=(-2, 2))
rel.x2 = pe.Var(bounds=compute_bounds_on_expr(rel.x**2))
rel.x4 = pe.Var(bounds=compute_bounds_on_expr(rel.x2**2))
rel.x2_con = coramin.relaxations.PWXSquaredRelaxation()
rel.x2_con.build(x=rel.x, w=rel.x2, use_linear_relaxation=True)
rel.x4_con = coramin.relaxations.PWXSquaredRelaxation()
rel.x4_con.build(x=rel.x2, w=rel.x4, use_linear_relaxation=True)
rel.obj = pe.Objective(expr=rel.x4 - 3*rel.x2 + rel.x)

print('*********************************')
print('OA Cut Generation')
print('*********************************')
opt = pe.SolverFactory('gurobi_direct')
res = opt.solve(rel)
lb = pe.value(rel.obj)
print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')

for _iter in range(5):
    for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
        if isinstance(b, coramin.relaxations.BasePWRelaxationData):
            b.add_cut()
    res = opt.solve(rel)
    lb = pe.value(rel.obj)
    print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')

for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
    if isinstance(b, coramin.relaxations.BasePWRelaxationData):
        b.rebuild()

print('\n*********************************')
print('OBBT')
print('*********************************')
res = opt.solve(rel)
lb = pe.value(rel.obj)
print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')
for _iter in range(5):
    coramin.domain_reduction.perform_obbt(rel, opt, [rel.x, rel.x2], objective_ub=ub)
    for b in rel.component_data_objects(pe.Block, active=True, sort=True, descend_into=True):
        if isinstance(b, coramin.relaxations.BasePWRelaxationData):
            b.rebuild()
    res = opt.solve(rel)
    lb = pe.value(rel.obj)
    print('gap: ' + str(100 * abs(ub - lb) / abs(ub)) + ' %')


