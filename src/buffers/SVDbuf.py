import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()

class SVDbuf(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        super().__init__(max_size, shape=None, n_classes=n_classes)
        self.n_components = kwargs.get('n_components', 8)
        self.register_buffer("U", torch.zeros(max_size, nb_ch, img_size, self.n_components))
        self.register_buffer("V", torch.zeros(max_size, nb_ch, self.n_components, img_size))
        self.img_size = img_size
        self.nb_ch = nb_ch

    def update(self, imgs, labels, **kwargs):
        for im, lab in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < self.max_size:
                reservoir_idx = self.n_seen_so_far
            if reservoir_idx < self.max_size:
                u1, s1, v1 = np.linalg.svd(im[0, :, :], full_matrices=False)
                u2, s2, v2 = np.linalg.svd(im[1, :, :], full_matrices=False)
                u3, s3, v3 = np.linalg.svd(im[2, :, :], full_matrices=False)
                us1 = u1[:, :self.n_components] @ np.diag(s1[:self.n_components])
                us2 = u2[:, :self.n_components] @ np.diag(s2[:self.n_components])
                us3 = u3[:, :self.n_components] @ np.diag(s3[:self.n_components])
                self.U[reservoir_idx, 0, :, :] = torch.from_numpy(us1).float()
                self.U[reservoir_idx, 1, :, :] = torch.from_numpy(us2).float()
                self.U[reservoir_idx, 2, :, :] = torch.from_numpy(us3).float()
                self.V[reservoir_idx, 0, :, :] = torch.from_numpy(v1[:self.n_components, :]).float()
                self.V[reservoir_idx, 1, :, :] = torch.from_numpy(v2[:self.n_components, :]).float()
                self.V[reservoir_idx, 2, :, :] = torch.from_numpy(v3[:self.n_components, :]).float()

                self.buffer_labels[reservoir_idx] = lab

                self.n_added_so_far +=1
            self.n_seen_so_far += 1

    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            u1 = self.U[:self.n_added_so_far, 0, :, :]
            u2 = self.U[:self.n_added_so_far, 1, :, :]
            u3 = self.U[:self.n_added_so_far, 2, :, :]
            v1 = self.V[:self.n_added_so_far, 0, :, :]
            v2 = self.V[:self.n_added_so_far, 1, :, :]
            v3 = self.V[:self.n_added_so_far, 2, :, :]
            
            im1 = torch.matmul(u1, v1)
            im2 = torch.matmul(u2, v2)
            im3 = torch.matmul(u3, v3)
            im = torch.stack([im1, im2, im3], dim=1)
            labels = self.buffer_labels[:self.n_added_so_far]
        else:
            ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)

            u1 = self.U[ret_indexes, 0, :, :]
            u2 = self.U[ret_indexes, 1, :, :]
            u3 = self.U[ret_indexes, 2, :, :]
            v1 = self.V[ret_indexes, 0, :, :]
            v2 = self.V[ret_indexes, 1, :, :]
            v3 = self.V[ret_indexes, 2, :, :]
            
            im1 = torch.matmul(u1, v1)
            im2 = torch.matmul(u2, v2)
            im3 = torch.matmul(u3, v3)
            im = torch.stack([im1, im2, im3], dim=1)
            labels = self.buffer_labels[ret_indexes]

        return im, labels