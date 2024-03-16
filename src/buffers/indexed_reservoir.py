import torch
import random as r
import numpy as np
import logging as lg

from src.buffers.reservoir import Reservoir
from src.utils.utils import get_device

device = get_device()


class IndexedReservoir(Reservoir):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        """Reservoir sampling for memory update
        Args:
            max_size (int, optional): maximum buffer size. Defaults to 200.
            img_size (int, optional): Image width/height. Images are considered square. Defaults to 32.
            nb_ch (int, optional): Number of image channels. Defaults to 3.
            n_classes (int, optional): Number of classes expected total. For print purposes only. Defaults to 10.
        """
        super().__init__(
            max_size=max_size,
            img_size=img_size,
            nb_ch=nb_ch,
            n_classes=n_classes,
            shape=kwargs.get('shape', None)
            )
        self.drop_method = kwargs.get('drop_method', 'random')
        self.register_buffer('buffer_data_idx', torch.FloatTensor(self.max_size).fill_(-1))

    def update(self, imgs, labels, idx_data, **kwargs):
        """Update buffer with the given list of images and labels.
            Note that labels are not used update selection, only when storing the image in memory.
        Args:
            imgs (torch.tensor): stream images seen by the buffer
            labels (torch.tensor): stream labels seen by the buffer
        Raises:
            NotImplementedError: NotImplementedError
        """
        for stream_img, stream_label, id_data in zip(imgs, labels, idx_data):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
            if reservoir_idx < self.max_size:
                if self.drop_method == 'random':
                    self.replace_data(reservoir_idx, stream_img, stream_label, id_data)
                else:
                    raise NotImplementedError("Only random update is implemented here.")
            self.n_seen_so_far += 1

    def replace_data(self, idx_buf, img, label, idx_data):
        self.buffer_imgs[idx_buf] = img
        self.buffer_labels[idx_buf] = label
        self.buffer_data_idx[idx_buf] = idx_data
        self.n_added_so_far += 1
    
    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], \
                self.buffer_labels[:self.n_added_so_far], \
                self.buffer_data_idx[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)

        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        ret_idx = self.buffer_data_idx[ret_indexes]
        
        return ret_imgs, ret_labels, ret_idx