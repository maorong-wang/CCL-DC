"""Custom FashionMNIST implementation to load one class at a time only.
"""
import torch
import numpy as np

from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from PIL import Image

from src.utils.utils import filter_labels


class SplitFashion(MNIST):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_labels = selected_labels
        print(selected_labels)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], int(self.targets[self.indexes[index]])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.indexes)
