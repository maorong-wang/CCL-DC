import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class BoostedBuffer(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        self.shape = kwargs.get('shape', None)
        if self.shape is not None:
            super().__init__(max_size, shape=self.shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.drop_method = kwargs.get('drop_method', 'random')
        self.register_buffer('buffer_coefs', torch.FloatTensor(self.max_size).fill_(1/self.max_size))

    def update(self, imgs, labels, **kwargs):
        crit = kwargs.get('crit', None)
        model = kwargs.get('model', None)
        stream_coefs = kwargs.get('stream_coefs', torch.ones(imgs.size(0)))
        tf = kwargs.get('transform', None)
        # reupdate every coefficients
        if model is not None and crit is not None and tf is not None:
            model.eval()
            with torch.no_grad():
                # combined_imgs = torch.cat([self.buffer_imgs, imgs]).to(device)
                _, projections1 = model(tf(self.buffer_imgs.to(device)))  # (batch_size, projectio   n_dim)
                _, projections2 = model(tf(self.buffer_imgs.to(device)))
                projections = torch.cat([projections1.unsqueeze(1), projections2.unsqueeze(1)], dim=1)

                # get updated coefs
                loss, coefs = crit(features=projections, labels=None)
                coefs = coefs[0,:]
                ema = 0.9
                self.buffer_coefs = (1-ema) * self.buffer_coefs + ema * coefs.cpu()
        
        idx_mem_coefs_sorted = self.buffer_coefs.sort().indices
        idx_stream_coefs_sorted = stream_coefs.sort(descending=True).indices
        n_data_replaced = 0
        for _, _ in zip(imgs, labels):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if reservoir_idx < self.max_size:
                idx_stream = idx_stream_coefs_sorted[n_data_replaced]
                if self.n_seen_so_far < self.max_size:
                    self.buffer_imgs[self.n_seen_so_far] = imgs[idx_stream]
                    self.buffer_labels[self.n_seen_so_far] = labels[idx_stream]
                    self.buffer_coefs[self.n_seen_so_far] = stream_coefs[idx_stream]
                else:
                    # print(self.buffer_coefs.sort())
                    idx_mem = idx_mem_coefs_sorted[n_data_replaced]
                    # print(idx, stream_coef)
                    self.buffer_imgs[idx_mem] = imgs[idx_stream]
                    self.buffer_labels[idx_mem] = labels[idx_stream]
                    self.buffer_coefs[idx_mem] = stream_coefs[idx_stream]
                n_data_replaced += 1
                self.n_added_so_far += 1
            self.n_seen_so_far += 1
    
    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far], self.buffer_coefs[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)

        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        ret_coefs = self.buffer_coefs[ret_indexes]
        
        return ret_imgs, ret_labels, ret_coefs
    