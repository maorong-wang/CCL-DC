"""Custom ImageNet implementation to load a subset of classes only
"""
import torch

from src.utils.utils import filter_labels
from src.datasets.ImageNet100 import ImageNet100


class SplitImageNet100(ImageNet100):
    def __init__(self, root, train, transform=None, selected_labels=[0]):
        super().__init__(root=root, train=train, transform=transform)
        self.selected_labels = selected_labels
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

    def __getitem__(self, index):
        img, label = super().__getitem__(self.indexes[index])
        return img, label, self.indexes[index]

    def __len__(self):
        return len(self.indexes)
