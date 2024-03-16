"""Custom CIFAR100 implementation to load one class at a time only.
"""
import torch
import numpy as np
import pandas as pd

from torchvision import transforms

from src.utils.utils import filter_labels
from src.datasets.tinyImageNet import TinyImageNet
from skimage import io
from PIL import Image

class BlurryTiny(TinyImageNet):
    def __init__(self, root, train, transform, download=False, labels_order=None, n_tasks=5, scale=3000):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.labels_order = labels_order
        self.targets = self.get_targets()
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
        img, target = io.imread(self.samples[self.indexes[index]][0]), self.samples[self.indexes[index]][1]

        if self.transform is not None:
            img = self.transform(img)
        if img.size(0) == 1:
            img = torch.cat([img, img, img], dim=0)
            
        return img, target, self.indexes[index]

    def __len__(self):
        return len(self.indexes)
