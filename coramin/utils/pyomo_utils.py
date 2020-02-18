import pyomo.environ as pe


def get_obj(m):
    """
    Assert that there is only one active objective in m and return it.

    Parameters
    ----------
    m: pyomo.core.base.block._BlockData

    Returns
    -------
    obj: pyomo.core.base.objective._ObjectiveData
    """
    obj = None
    for o in m.component_data_objects(pe.Objective, descend_into=True, active=True, sort=True):
        if obj is not None:
            raise ValueError('Found multiple active objectives')
        obj = o
    return obj
