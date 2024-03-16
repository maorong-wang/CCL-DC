"""Custom TinyIN implementation to load a subset of classes only.
"""
import torch
import numpy as np

from src.utils.utils import filter_labels
from src.datasets.tinyImageNet import TinyImageNet
from torchvision import transforms
from skimage import io
# import imageio.v2 as io


class SplitTiny(TinyImageNet):
    def __init__(self, root, train, transform, download=False, selected_labels=[0]):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_labels = selected_labels
        self.indexes = torch.nonzero(filter_labels(self.get_targets(), self.selected_labels)).flatten()

    def __getitem__(self, index):
        img, target = io.imread(self.samples[self.indexes[index]][0]), self.samples[self.indexes[index]][1]

        if self.transform is not None:
            img = self.transform(img)
        if img.size(0) == 1:
            img = torch.cat([img, img, img], dim=0)
            
        return img, target, self.indexes[index]

    def __len__(self):
        return len(self.indexes)
