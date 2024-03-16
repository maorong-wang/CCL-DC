"""Custom MNIST implementation to load one class at a time only.
"""
import torch
import numpy as np

from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from PIL import Image


class Number(MNIST):
    def __init__(self, root, train, transform, download=False, selected_labels=[0], permute=False):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_labels = selected_labels
        print(selected_labels)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

        if permute:
            # Generate permute indices
            ind_permute = np.arange(0, 28)
            np.random.seed(0)
            np.random.shuffle(ind_permute)
            transform_permute = lambda img_batch: permute_image(img_batch, ind_permute)
            self.transform = transforms.Compose([
                transforms.Lambda(transform_permute),
                self.transform
            ])

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], int(self.targets[self.indexes[index]])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexes[index]

    def __len__(self):
        return len(self.indexes)


def permute_image(img, ind_permute):
        """ind_permute: numpy ndarray, ind_permute shape: (28,)"""
        img_new = np.array(img)[ind_permute, :]
        return Image.fromarray(img_new)


# @utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte()

    for label in labels:
        mapping = mapping | y.eq(label)

    return mapping