import pyomo.environ as pe
from .relaxations_base import BaseRelaxationData


def relaxation_data_objects(block, descend_into=True, active=None, sort=False):
    for b in block.component_data_objects(pe.Block, descend_into=descend_into, active=active, sort=sort):
        if isinstance(b, BaseRelaxationData):
            yield b


def _nonrelaxation_block_objects(block, descend_into=True, active=None, sort=False):
    for b in block.component_data_objects(pe.Block, descend_into=False, active=active, sort=sort):
        if isinstance(b, BaseRelaxationData):
            continue
        else:
            yield b
            if descend_into:
                for _b in _nonrelaxation_block_objects(b, descend_into=True, active=active, sort=sort):
                    yield _b


def nonrelaxation_component_data_objects(block, ctype=None, active=None, sort=False, descend_into=True):
    if ctype is pe.Block:
        for b in _nonrelaxation_block_objects(block, descend_into=descend_into, active=active, sort=sort):
            yield b
    else:
        for comp in block.component_data_objects(ctype=ctype, descend_into=False, active=active, sort=sort):
            yield comp
        if descend_into:
            for b in _nonrelaxation_block_objects(block, descend_into=True, active=active, sort=sort):
                for comp in b.component_data_objects(ctype=ctype, descend_into=False, active=active, sort=sort):
                    yield comp
