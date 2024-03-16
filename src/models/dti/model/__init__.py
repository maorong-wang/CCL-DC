import torch

from .dtigmm import DTIGMM
from .dtikmeans import DTIKmeans
from .tools import safe_model_state_dict


def get_model(name):
    return {
        'dtikmeans': DTIKmeans,
        'dtigmm': DTIGMM,
    }[name]
