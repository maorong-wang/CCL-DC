import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class CReservoir(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, drop_method='random', shape=None):
        if shape is not None:
            super().__init__(max_size, shape=shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch

    def update(self, imgs, labels, model):
        ignored_images, ignored_labels, dropped_imgs, dropped_labels = [], [], [], []
        for stream_img, stream_label in zip(imgs, labels):
            if self.n_seen_so_far < self.max_size:
                self.stack_data(stream_img, stream_label)
            else:
                reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                
                if reservoir_idx < self.max_size:
                    self.model.eval()
                    representations, _ = model(self.buffer_imgs.to(device))
                    cons_val, cons_indices = (representations @ representations.T).sum(1).sort(descending=True)
                    drop_idx = cons_indices[0]
                    d_img, d_label = self.replace_data(drop_idx, stream_img, stream_label)
                    dropped_imgs.append(d_img)
                    dropped_labels.append(d_label)
                else:
                    ignored_images.append(stream_img)
                    ignored_labels.append(stream_label)

            self.n_seen_so_far += 1
        return ignored_images, ignored_labels, dropped_imgs, dropped_labels
    
    def replace_data(self, idx, img, label):
        old_img = self.buffer_imgs[idx]
        old_label = self.buffer_labels[idx]
        
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.n_added_so_far += 1

        return old_img, old_label
