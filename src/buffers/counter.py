import random as r
import numpy as np
import logging as lg

from src.buffers.buffer import Buffer


class ShortMemory(Buffer):
    def __init__(self, max_size=200, max_count=10, img_size=32, nb_ch=3, n_classes=10):
        super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.max_count = max_count
        # use -2 if the slot is empty, -1 if the slot is taken by an image that should be dropped
        self.img_counts = [-2] * self.max_size

    def update(self, imgs, labels=None):
        free_slots = self.free_slots()
        if len(free_slots) < len(imgs):
            print(f"Number of images to add {len(imgs)} is superior to memory space {len(free_slots)}.")
            imgs = imgs[:len(free_slots)]
            labels = labels[:len(free_slots)]
        else:
            free_slots = free_slots[:len(imgs)]

        for slot_idx, stream_img, stream_label in zip(free_slots, imgs, labels):
            self.replace_data(slot_idx, stream_img, stream_label)

    def replace_data(self, idx, img, label):
        super().replace_data(idx, img, label)
        self.img_counts[idx] = 0
    
    def is_empty(self):
        return not len(self.taken_slots())
    
    def free_slots(self):
        return [i for i in range(len(self.img_counts)) if self.img_counts[i] < 0]

    def taken_slots(self):
        return [i for i in range(len(self.img_counts)) if self.img_counts[i] >= 0]

    def drop_data(self):
        # To optim
        to_drop_idx = [idx for idx, count in enumerate(self.img_counts) if count == -1]
        to_drop_images = self.buffer_imgs[to_drop_idx]
        to_drop_labels = self.buffer_labels[to_drop_idx]

        for idx in to_drop_idx:
            self.img_counts[idx] = -2

        return to_drop_images, to_drop_labels

    def update_counts(self):
        for idx, count in enumerate(self.img_counts):
            if count >= self.max_count:
                self.img_counts[idx] = -1

    def random_retrieve(self, n_imgs=100):
        if len(self.taken_slots()) < n_imgs:
            lg.debug(f"""
                Cannot retrieve the number of requested images from short memory 
                {len(self.free_slots())}/{n_imgs}
                """)
            ret_indexes = self.taken_slots()
        else:
            ret_indexes = r.sample(self.taken_slots(), n_imgs)
            
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        for idx in ret_indexes:
            self.img_counts[idx] += 1
        
        self.update_counts()

        return ret_imgs, ret_labels

    def get_labels_distribution(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0], minlength=self.n_classes)

        return counts / len(self.buffer_labels[self.buffer_labels >= 0])