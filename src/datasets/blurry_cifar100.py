"""Custom CIFAR100 implementation to load one class at a time only.
"""
import torch
import numpy as np
import pandas as pd

from src.utils.utils import filter_labels
from torchvision.datasets.cifar import CIFAR100
from torchvision import transforms
from PIL import Image


class BlurryCIFAR100(CIFAR100):
    def __init__(self, root, train, transform, download=False, labels_order=None, n_tasks=5, scale=3000):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.labels_order = labels_order
        self.targets = torch.Tensor(self.targets)
        step = len(labels_order) // n_tasks
        indexes_all = []
        for t in range(n_tasks):
            labels_t = self.labels_order[t * step: (t+1)*step]
            indexes_t = torch.nonzero(filter_labels(self.targets, labels_t)).flatten()
            indexes_all.append(indexes_t[torch.randperm(len(indexes_t))])
        self.indexes = torch.cat(indexes_all)
        shuffled_indexes = []
        self.scale = scale
        
        noise = torch.distributions.half_normal.HalfNormal(self.scale)

        n_data = len(self.indexes)
        for _ in range(n_data):
            idx = noise.sample().long()
            counter = 0
            while idx >= len(self.indexes) and counter < 10:
                idx = noise.sample().long()
                counter += 1
            else:
                if counter >= 10:
                    idx = torch.Tensor([0]).long()
            selected_idx = self.indexes[idx]
            shuffled_indexes.append(selected_idx.item())
            self.indexes = self.indexes[self.indexes != self.indexes[idx]]
        self.indexes = torch.Tensor(shuffled_indexes).long()

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], self.targets[self.indexes[index]]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexes[index]

    def __len__(self):
        return len(self.indexes)
