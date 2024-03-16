"""
Simple Tiny ImageNet dataset utility class for pytorch.
Modified version of https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54
"""
from cgi import test
import os

import numpy as np
import os
import torch

from torchvision.datasets.utils import download_and_extract_archive
from collections import defaultdict
from torch.utils.data import Dataset
from skimage import io
# import imageio.v2 as io

from tqdm.autonotebook import tqdm

dir_structure_help = """
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(url, root_dir):
		if os.path.exists(root_dir):
				return
		download_and_extract_archive(
				url,
				root_dir,
				filename='tiny-imagenet-200.zip',
				remove_finished=True,
				md5='90528d7ca1a48142e341f4ef8d21d0de'
		)

def _add_channels(img, total_channels=3):
	while len(img.shape) < 3:  # third axis is the channels
		img = np.expand_dims(img, axis=-1)
	while(img.shape[-1]) < 3:
		img = np.concatenate([img, img[:, :, -1:]], axis=-1)
	return img

"""Creates a paths datastructure for the tiny imagenet.
Args:
	root_dir: Where the data is located
	download: Download if the data is not there
Members:
	label_id:
	ids:
	nit_to_words:
	data_dict:
"""
class TinyImageNetPaths:
	def __init__(self, root_dir, download=False):
		root_dir = os.path.join(root_dir, 'tiny-imagenet-200')
		if download:
			download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
												 root_dir)
		train_path = os.path.join(root_dir, 'train')
		val_path = os.path.join(root_dir, 'val')
		test_path = os.path.join(root_dir, 'test')

		wnids_path = os.path.join(root_dir, 'wnids.txt')
		words_path = os.path.join(root_dir, 'words.txt')

		self._make_paths(train_path, val_path, test_path,
										 wnids_path, words_path)

	def _make_paths(self, train_path, val_path, test_path,
									wnids_path, words_path):
		self.ids = []
		with open(wnids_path, 'r') as idf:
			for nid in idf:
				nid = nid.strip()
				self.ids.append(nid)
		self.nid_to_words = defaultdict(list)
		with open(words_path, 'r') as wf:
			for line in wf:
				nid, labels = line.split('\t')
				labels = list(map(lambda x: x.strip(), labels.split(',')))
				self.nid_to_words[nid].extend(labels)

		self.paths = {
			'train': [],  # [img_path, id, nid, box]
			'val': [],  # [img_path, id, nid, box]
			'test': []  # img_path
		}

		# Get the test paths
		# print(test_path)
		self.paths['test'] = list(map(lambda x: os.path.join(test_path, x), os.listdir(os.path.join(test_path, 'images'))))
		# print(self.paths['test'])
		# Get the validation paths and labels
		with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
			for line in valf:
				fname, nid, x0, y0, x1, y1 = line.split()
				fname = os.path.join(val_path, 'images', fname)
				bbox = int(x0), int(y0), int(x1), int(y1)
				label_id = self.ids.index(nid)
				self.paths['val'].append((fname, label_id, nid, bbox))

		# Get the training paths
		train_nids = os.listdir(train_path)
		for nid in train_nids:
			anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
			imgs_path = os.path.join(train_path, nid, 'images')
			label_id = self.ids.index(nid)
			with open(anno_path, 'r') as annof:
				for line in annof:
					fname, x0, y0, x1, y1 = line.split()
					fname = os.path.join(imgs_path, fname)
					bbox = int(x0), int(y0), int(x1), int(y1)
					self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.
Args:
	root_dir: Root directory for the data
	train: One of "train", "test", or "val"
	preload: Preload into memory
	load_transform: Transformation to use at the preload time
	transform: Transformation to use at the retrieval time
	download: Download the dataset
Members:
	tinp: Instance of the TinyImageNetPaths
	data: Image data
	targets: Label data
"""
class TinyImageNet(Dataset):
	def __init__(self, root:str, train:bool, transform=None, download:bool=True):
		tinp = TinyImageNetPaths(root, download)
		self.train = train
		self.transform = transform
		self.IMAGE_SHAPE = (64, 64, 3)
		if self.train:
			self.samples = list(tinp.paths['train'])
		else:
			self.samples = list(tinp.paths['val'])

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img, target = io.imread(self.samples[idx][0]), self.samples[idx][1]

		if self.transform:
			img = self.transform(img)
		if img.size(0) == 1:
			img = torch.cat([img, img, img], dim=0)

		return img, target
	
	def __gettarget__(self, idx):
		return self.samples[idx][1]

	def get_targets(self):
		targets = []
		for i in range(len(self.samples)):
			targets.append(self.__gettarget__(i))
		return torch.Tensor(targets)