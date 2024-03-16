import torch
import numpy as np
import random as r
import logging as lg

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()

class GreedySampler(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, drop_method='random', n_classes=10, **kwargs):
        super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.drop_method = drop_method
        self.img_size = img_size
        self.nb_ch = nb_ch

    def update(self, imgs, labels, **kwargs):
        for stream_data, stream_label in zip(imgs, labels):
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                # Do nothing if class has reached maximum number of images
                if len(class_indexes) < max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)
                    idx = int(np.random.choice(class_indexes.numpy().squeeze(), 1))
                    self.replace_data(idx, stream_data, stream_label)
            self.n_seen_so_far += 1