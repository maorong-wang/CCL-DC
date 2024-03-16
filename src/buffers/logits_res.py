import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class LogitsRes(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, shape=None, **kwargs):
        """Reservoir sampling with images + logits for derpp.
        """
        if shape is not None:
            super().__init__(max_size, shape=shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.drop_method = kwargs.get('drop_method', 'random')
        self.register_buffer('buffer_logits', torch.FloatTensor(self.max_size, n_classes).fill_(0))

    def update(self, imgs, labels, logits, **kwargs):
        """Update buffer with the given list of images and labels.
            Note that labels are not used update selection, only when storing the image in memory.
        Args:
            imgs (torch.tensor): stream images seen by the buffer
            labels (torch.tensor): stream labels seen by the buffer
            logits (torch.tensor): stream logits seen by the buffer
        """
        for stream_img, stream_label, stream_logit in zip(imgs, labels, logits):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
            if reservoir_idx < self.max_size:
                self.replace_data(reservoir_idx, stream_img, stream_label, stream_logit)
            self.n_seen_so_far += 1
    
    def replace_data(self, idx, img, label, logit):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.buffer_logits[idx] = logit
        self.n_added_so_far += 1

    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far], self.buffer_logits[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        ret_logits = self.buffer_logits[ret_indexes]

        return ret_imgs, ret_labels, ret_logits