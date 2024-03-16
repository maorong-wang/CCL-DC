from .mnist import MNIST
from .number import Number
from .split_fashion import SplitFashion
from .fashion_mnist import FashionMNIST
from .cifar10 import CIFAR10
from .split_cifar10 import SplitCIFAR10
from .cifar100 import CIFAR100
from .split_cifar100 import SplitCIFAR100
from .ImageNet import ImageNet
from .split_ImageNet import SplitImageNet
from .blurry_cifar10 import BlurryCIFAR10
from .blurry_cifar100 import BlurryCIFAR100
from .blurry_tiny import BlurryTiny


__all__ = (
    'MNIST', 
    'Number', 
    'FashionMNIST', 
    'SplitFashion', 
    'SplitCIFAR10', 
    'SplitCIFAR100',
    'ImageNet',
    'SplitImageNet',
    'BlurryCIFAR10',
    'BlurryCIFAR100',
    'BlurryTiny'
    )
