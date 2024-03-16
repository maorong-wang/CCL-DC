from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class FashionMNIST(datasets.FashionMNIST):
    """FashionMNIST Instance Dataset.
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.train:
        img, target = self.data[index], self.targets[index]
        # else:
            # img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
