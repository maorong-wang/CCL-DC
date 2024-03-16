import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class Reservoir(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        """Reservoir sampling for memory update
        Args:
            max_size (int, optional): maximum buffer size. Defaults to 200.
            img_size (int, optional): Image width/height. Images are considered square. Defaults to 32.
            nb_ch (int, optional): Number of image channels. Defaults to 3.
            n_classes (int, optional): Number of classes expected total. For print purposes only. Defaults to 10.
        """
        self.shape = kwargs.get('shape', None)
        super().__init__(
            max_size,
            shape=self.shape if self.shape is not None else (nb_ch, img_size, img_size),
            n_classes=n_classes,
            )
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.drop_method = kwargs.get('drop_method', 'random')

    def reset(self):
        """Resets n_seen_so_far counter to that the reservoir starts storing all incming data
        """
        self.n_seen_so_far = 0
        
    def update(self, imgs, labels, **kwargs):
        """Update buffer with the given list of images and labels.
            Note that labels are not used update selection, only when storing the image in memory.
        Args:
            imgs (torch.tensor): stream images seen by the buffer
            labels (torch.tensor): stream labels seen by the buffer
        Raises:
            NotImplementedError: NotImplementedError
        """
        for stream_img, stream_label in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_added_so_far
            if reservoir_idx < self.max_size:
                if self.drop_method == 'random':
                    self.replace_data(reservoir_idx, stream_img, stream_label)
                else:
                    raise NotImplementedError("Only random update is implemented here.")
            self.n_seen_so_far += 1
