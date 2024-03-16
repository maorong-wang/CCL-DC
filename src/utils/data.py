"""Utils function for data loading and data processing.
"""
import torch
import numpy as np
import logging as lg
import random as r

from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
# from kornia.augmentation import Resize
from torch import nn
from sklearn.cluster import KMeans

from src.utils.AdaIN.test import test_transform
from src.utils.utils import get_device
from src.datasets import MNIST, Number, FashionMNIST, SplitFashion, ImageNet
from src.datasets import CIFAR10, SplitCIFAR10, CIFAR100, SplitCIFAR100, SplitImageNet
from src.datasets import BlurryCIFAR10, BlurryCIFAR100, BlurryTiny
from src.datasets.tinyImageNet import TinyImageNet
from src.datasets.split_tiny import SplitTiny
from src.datasets.ImageNet100 import ImageNet100
from src.datasets.split_ImageNet100 import SplitImageNet100

device = get_device()

def get_loaders(args):
    tf = transforms.ToTensor()
    dataloaders = {}
    if args.labels_order is None:
        l = np.arange(args.n_classes)
        np.random.shuffle(l)
        args.labels_order = l.tolist()
    if args.dataset == 'mnist':
        dataset_train = MNIST(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = MNIST(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'fmnist':
        dataset_train = FashionMNIST(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = FashionMNIST(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'cifar10':
        if args.training_type == 'blurry':
            dataset_train = BlurryCIFAR10(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = CIFAR10(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = CIFAR10(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = CIFAR10(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'cifar100':
        if args.training_type == 'blurry':
            dataset_train = BlurryCIFAR100(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = CIFAR100(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = CIFAR100(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = CIFAR100(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'tiny':
        if args.training_type == 'blurry':
            dataset_train = BlurryTiny(root=args.data_root_dir, labels_order=args.labels_order,
                train=True, download=True, transform=tf, n_tasks=args.n_tasks, scale=args.blurry_scale)
            dataset_test = TinyImageNet(args.data_root_dir, train=False, download=True, transform=tf)
        else:
            dataset_train = TinyImageNet(args.data_root_dir, train=True, download=True, transform=tf)
            dataset_test = TinyImageNet(args.data_root_dir, train=False, download=True, transform=tf) 
    elif args.dataset == 'sub':
        # Loading only the necessary labels for SubImageNet
        dataset_train = SplitImageNet(root=args.data_root_dir, split='train',
                                        selected_labels=np.arange(args.n_classes), transform=tf)
        dataset_test = SplitImageNet(root=args.data_root_dir, split='val',
                                        selected_labels=np.arange(args.n_classes), transform=tf)
    elif args.dataset == 'yt':
        tf = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                Resize(size=(256,256)),
        ])
        dataset_train = ImageFolder('/storage8To/datasets/deepsponsorblock/images/old/train_old', transform=tf)
        dataset_test = ImageFolder('/storage8To/datasets/deepsponsorblock/images/old/test_old', transform=tf)
    
    elif args.dataset == 'imagenet100':
        dataset_train = ImageNet100(root=args.data_root_dir, train=True, transform=tf)
        dataset_test = ImageNet100(root=args.data_root_dir, train=False, transform=tf)
    
    if args.training_type == 'inc':
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="train")
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="test")
    
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=args.training_type != 'blurry', num_workers=args.num_workers)
    dataloaders['test'] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloaders


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def add_incremental_splits(args, dataloaders, tf, tag="train"):
    is_train = tag == "train"
    step_size = int(args.n_classes / args.n_tasks)
    lg.info("Loading incremental splits with labels :")
    for i in range(0, args.n_classes, step_size):
        lg.info([args.labels_order[j] for j in range(i, i+step_size)])
    for i in range(0, args.n_classes, step_size):
        if args.dataset == 'mnist':
            dataset = Number(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)],
                permute=False
                )
        elif args.dataset == 'fmnist':
            dataset = SplitFashion(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == 'cifar10':
            dataset = SplitCIFAR10(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == 'cifar100':
            dataset = SplitCIFAR100(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == 'tiny':
            dataset = SplitTiny(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == "sub":
            dataset = SplitImageNet(
                root=args.data_root_dir,
                split='train' if is_train else 'val',
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)],
                transform=tf
            )
        elif args.dataset == "imagenet100":
            dataset = SplitImageNet100(
                root=args.data_root_dir,
                train=is_train,
                selected_labels=[args.labels_order[j] for j in range(i, i+step_size)],
                transform=tf
            )
        else:
            raise NotImplementedError
        dataloaders[f"{tag}{int(i/step_size)}"] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
 
    return dataloaders
