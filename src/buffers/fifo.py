import random as r
import numpy as np
import logging as lg
import torch

from src.buffers.buffer import Buffer


class QueueMemory(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10):
        super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.max_size = max_size

    def update(self, imgs, labels=None):
        for im, lab in zip(imgs, labels):
            self.put(im, lab)
            self.n_seen_so_far += 1
        # TODO : add dropped images and labels
        return [], [], [], []

    def put(self, im, lab):
        # Index of latest memory element
        memory_index = self.n_added_so_far % self.max_size
        self.replace_data(memory_index, im, lab)
        