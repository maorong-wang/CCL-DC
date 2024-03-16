import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from .transformer import PrototypeTransformationNetwork
from .tools import copy_with_noise, create_gaussian_weights, generate_data, generate_data_from_batch
from src.models.dti.utils.logger import print_warning


NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2


class DTIKmeans(nn.Module):
    name = 'dtikmeans'

    def __init__(self, dataset=None, data_batch=None, n_prototypes=10, **kwargs):
        super().__init__()
        self.n_prototypes = n_prototypes
        # print(kwargs)
        init_type = kwargs.get('init_type', 'sample')
        if dataset is None:
            self.prototypes = nn.Parameter(torch.stack(generate_data_from_batch(data_batch, n_prototypes, init_type)))
        else:
            self.prototypes = nn.Parameter(torch.stack(generate_data(dataset, n_prototypes, init_type)))
        self.transformer = PrototypeTransformationNetwork(3, (32,32), n_prototypes, **kwargs)
        self.empty_cluster_threshold = kwargs.get('empty_cluster_threshold', EMPTY_CLUSTER_THRESHOLD / n_prototypes)
        self._reassign_cluster = kwargs.get('reassign_cluster', True)
        use_gaussian_weights = kwargs.get('gaussian_weights', False)
        if use_gaussian_weights:
            std = kwargs['gaussian_weights_std']
            self.register_buffer('loss_weights', create_gaussian_weights((32,32), 3, std))
        else:
            self.loss_weights = None

    def cluster_parameters(self):
        return [self.prototypes]

    def transformer_parameters(self):
        return self.transformer.parameters()

    def forward(self, x):
        prototypes = self.prototypes.unsqueeze(1).expand(-1, x.size(0), x.size(1), -1, -1)
        inp, target = self.transformer(x, prototypes)
        distances = (inp - target)**2
        if self.loss_weights is not None:
            distances = distances * self.loss_weights
        distances = distances.flatten(2).mean(2)
        dist_min = distances.min(1)[0]
        return dist_min.mean(), distances

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse: 
            return self.transformer.inverse_transform(x)
        else:
            prototypes = self.prototypes.unsqueeze(1).expand(-1, x.size(0), x.size(1), -1, -1)
            return self.transformer(x, prototypes)[1]

    def step(self):
        self.transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        self.transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster:
            return [], 0

        idx = np.argmax(proportions)
        reassigned = []
        for i in range(self.n_prototypes):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j):
        self.prototypes[i].data.copy_(copy_with_noise(self.prototypes[j], NOISE_SCALE))
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(self, 'optimizer'):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                param = self.prototypes
                opt.state[param]['exp_avg'][i] = opt.state[param]['exp_avg'][j]
                opt.state[param]['exp_avg_sq'][i] = opt.state[param]['exp_avg_sq'][j]
            else:
                raise NotImplementedError('unknown optimizer: you should define how to reinstanciate statistics if any')
