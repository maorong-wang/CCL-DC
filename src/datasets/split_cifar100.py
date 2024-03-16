"""Custom CIFAR100 implementation to load one class at a time only.
"""
import torch
import numpy as np

from torchvision.datasets.cifar import CIFAR100
from torchvision import transforms
from PIL import Image

from src.utils.utils import filter_labels


class SplitCIFAR100(CIFAR100):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_labels = selected_labels
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

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
