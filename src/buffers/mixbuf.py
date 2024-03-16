import torch
import random as r
import numpy as np
import pandas as pd

from torchvision import transforms
from src.buffers.buffer import Buffer
from src.utils.data import get_color_distortion
from src.utils.utils import get_device

device = get_device()

class MixBuf(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, shape=None):
        """Reservoir sampling with mixup update and storing old reps
        """
        if shape is not None:
            super().__init__(max_size, shape=shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.register_buffer('buffer_projs1', torch.FloatTensor(self.max_size, 128).fill_(0))
        self.register_buffer('buffer_projs1_coefs', torch.FloatTensor(self.max_size, 1).fill_(0))
        self.register_buffer('buffer_projs2', torch.FloatTensor(self.max_size, 128).fill_(0))
        self.register_buffer('buffer_projs2_coefs', torch.FloatTensor(self.max_size, 1).fill_(0))

    def update(self, imgs, labels, model, method="mixup"):
        for stream_img, stream_label in zip(imgs, labels):
            if self.n_seen_so_far < self.max_size:
                self.stack_data(stream_img, stream_label, model)
            else:
                reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                if reservoir_idx < self.max_size:
                    # If the image is not mixed up
                    if self.buffer_projs2_coefs[reservoir_idx] == 0:
                        self.mix_data(reservoir_idx, stream_img, stream_label, model, method)
                    else:
                        self.replace_data(reservoir_idx, stream_img, stream_label, model)
            self.n_seen_so_far += 1
    
    def mix_data(self, idx, img, label, model, method):
        with torch.no_grad():
            old_img = self.buffer_imgs[idx]
            # Get images projections
            model.eval()
            model.to(device)
            _, proj1 = model(img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
            _, proj2 = model(img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
            # Keep old proj ?
            self.buffer_projs1[idx] = proj1
            self.buffer_projs2[idx] = proj2
            
            if method == "mixup":
                # Mixup ratio
                mix_ratio = r.random()

                # New mixuped image
                self.buffer_imgs[idx] = old_img * mix_ratio + (1 - mix_ratio) * img
            elif method == 'cutmix':
                lam = np.random.beta(1, 1)
                image_h, image_w = img.size(1), img.size(2)
                cx = np.random.uniform(0, image_w)
                cy = np.random.uniform(0, image_h)
                w = image_w * np.sqrt(1 - lam)
                h = image_h * np.sqrt(1 - lam)
                x0 = int(np.round(max(cx - w / 2, 0)))
                x1 = int(np.round(min(cx + w / 2, image_w)))
                y0 = int(np.round(max(cy - h / 2, 0)))
                y1 = int(np.round(min(cy + h / 2, image_h)))
                self.buffer_imgs[idx][:, x0:x1, y0:y1] = img[:, x0:x1, y0:y1]
                mix_ratio = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))
                self.buffer_projs1_coefs[idx] = (x1 - x0) * (y1 - y0) / (image_h * image_w)
            else:
                raise NotImplementedError

            self.buffer_projs1_coefs[idx] = mix_ratio
            self.buffer_projs2_coefs[idx] = 1 - mix_ratio
            self.n_added_so_far += 1

    def replace_data(self, idx, img, label, model):
        with torch.no_grad():
            model.eval()
            model.to(device)
            _, proj = model(img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
            self.buffer_imgs[idx] = img
            self.buffer_labels[idx] = label
            self.buffer_projs1[idx] = proj
            # reset placeholder
            self.buffer_projs1_coefs[idx] = 1
            self.buffer_projs2_coefs[idx] = 0
            self.n_added_so_far += 1
    
    def stack_data(self, stream_img, stream_label, model):
        with torch.no_grad():
            model.eval()
            model.to(device)
            _, proj = model(stream_img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
            self.buffer_imgs[self.n_seen_so_far] = stream_img
            self.buffer_labels[self.n_added_so_far] = stream_label
            self.buffer_projs1[self.n_added_so_far] = proj
            self.n_added_so_far += 1

    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far],\
                self.buffer_projs1[:self.n_added_so_far], self.buffer_projs2[:self.n_added_so_far],\
                self.buffer_projs1_coefs[:self.n_added_so_far], self.buffer_projs2_coefs[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        ret_projs1 = self.buffer_projs1[ret_indexes]
        ret_projs2 = self.buffer_projs2[ret_indexes]
        ret_projs1_coef = self.buffer_projs1_coefs[ret_indexes]
        ret_projs2_coef = self.buffer_projs2_coefs[ret_indexes]

        return ret_imgs, ret_labels, ret_projs1, ret_projs2, ret_projs1_coef, ret_projs2_coef


class DualCutBuf(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, shape=None):
        """CutMix with pre-defined regions and loss between different different image regions and all.

        Args:
            max_size (int, optional): _description_. Defaults to 200.
            img_size (int, optional): _description_. Defaults to 32.
            nb_ch (int, optional): _description_. Defaults to 3.
            n_classes (int, optional): _description_. Defaults to 10.
            shape (_type_, optional): _description_. Defaults to None.
        """
        if shape is not None:
            super().__init__(max_size, shape=shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.register_buffer('buffer_pos', torch.FloatTensor(self.max_size, 1).fill_(-2))
        tf = transforms.Compose([
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
        self.transform_train = transforms.Compose([
                    transforms.RandomResizedCrop((32, 32), (0.2, 1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    get_color_distortion(s=0.5),
                    tf
                ])

    def update(self, imgs, labels, model, method="mixup"):
        for stream_img, stream_label in zip(imgs, labels):
            if self.n_seen_so_far < self.max_size:
                self.stack_data(stream_img, stream_label, model)
            else:
                reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                if reservoir_idx < self.max_size:
                    # If the image is not mixed up
                    if self.buffer_pos[reservoir_idx] == -1:
                        self.mix_data(reservoir_idx, stream_img, stream_label, model, method)
                    else:
                        self.replace_data(reservoir_idx, stream_img, stream_label, model)
            self.n_seen_so_far += 1
    
    def mix_data(self, idx, img, label, model, method):
        dists = []
        with torch.no_grad():
            # Get images projections
            model.eval()
            model.to(device)

            # _, proj1 = model(img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
            # _, proj2 = model(img.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))

            # select zone to cutmix
            # the random way
            # zone_in = r.randint(0, 3)  # zone to select from new img
            # zone_out = r.randint(0, 3)  # Zone to replace from old image
            # self.buffer_pos[idx] = zone_out
            # self.cutmix(idx, zone_in, zone_out, img)

            # The optimal way ?
            zones = []
            old_img = self.buffer_imgs[idx].clone()
            for zone_in in range(4):
                for zone_out in range(4):
                    self.cutmix(idx, zone_in, zone_out, img)
                    imgs = self.get_cuts([idx])
                    im1, im2 = imgs[0], imgs[1]
                    _, p1 = model(im1.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
                    _, p2 = model(im2.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
                    mix = self.buffer_imgs[idx]
                    _, pmix = model(mix.view(-1, self.nb_ch, self.img_size, self.img_size).to(device))
                    
                    d = torch.square(p2 @ pmix.T) + torch.square(p2 @ pmix.T)
                    dists.append(d.item())
                    zones.append((zone_in, zone_out))
                    df = pd.DataFrame(dists)
            
            idx_min = df.idxmax().values[0]
            zone_in = zones[idx_min][0]
            zone_out = zones[idx_min][1]

            self.buffer_pos[idx] = zone_out
            self.buffer_imgs[idx] = old_img
            self.cutmix(idx, zone_in, zone_out, img)
            self.n_added_so_far += 1

    def cutmix(self, idx, zone_in, zone_out, img):
        img_w = self.buffer_imgs[idx].size(1)
        img_h = self.buffer_imgs[idx].size(2)
        x0 = 0
        x1 = int(img_w/2)
        x2 = img_w
        y0 = 0
        y1 = int(img_h/2)
        y2 = img_h
        
        # Select patch in
        if zone_in == 0:
            patch_in = img[:, x0:x1, y0:y1]
        elif zone_in == 1:
            patch_in = img[:, x1:x2, y0:y1]
        elif zone_in == 2:
            patch_in = img[:, x0:x1, y1:y2]
        elif zone_in == 3:
            patch_in = img[:, x1:x2, y1:y2]

        # Replace zone_out by patch in
        if zone_out == 0:
            self.buffer_imgs[idx][:, x0:x1, y0:y1] = patch_in
        elif zone_out == 1:
            self.buffer_imgs[idx][:, x1:x2, y0:y1] = patch_in
        elif zone_out == 2:
            self.buffer_imgs[idx][:, x0:x1, y1:y2] = patch_in
        elif zone_out == 2:
            self.buffer_imgs[idx][:, x1:x2, y1:y2] = patch_in

    def replace_data(self, idx, img, label, model):
        with torch.no_grad():
            model.eval()
            model.to(device)
            self.buffer_imgs[idx] = img
            self.buffer_labels[idx] = label

            # reset placeholder
            self.buffer_pos[idx] = -1

            self.n_added_so_far += 1
    
    def stack_data(self, stream_img, stream_label, model):
        with torch.no_grad():
            model.eval()
            model.to(device)
            self.buffer_imgs[self.n_seen_so_far] = stream_img
            self.buffer_labels[self.n_added_so_far] = stream_label
            self.buffer_pos[self.n_added_so_far] = -1
            self.n_added_so_far += 1

    def get_cuts(self, indexes):
        patches1 = None
        patches2 = None
        for i in indexes:
            im_full = self.buffer_imgs[i]
            zone = self.buffer_pos[i]
            if zone >=0:
                img_w = im_full.size(1)
                img_h = im_full.size(2)
                x0 = 0
                x1 = int(img_w/2)
                x2 = img_w
                y0 = 0
                y1 = int(img_h/2)
                y2 = img_h

                patch1 = im_full.clone()
                patch2 = im_full.clone()

                if zone == 0:
                    patch1[:, :, :] = 0
                    patch1[:, x0:x1, y0:y1] = im_full[:, x0:x1, y0:y1]
                    patch2[:, x0:x1, y0:y1] = 0
                elif zone == 1:
                    patch1[:, :, :] = 0
                    patch1[:, x1:x2, y0:y1] = im_full[:, x1:x2, y0:y1]
                    patch2[:, x1:x2, y0:y1] = 0
                elif zone == 2:
                    patch1[:, :, :] = 0
                    patch1[:, x0:x1, y1:y2] = im_full[:, x0:x1, y1:y2]
                    patch2[:, x0:x1, y1:y2] = 0
                elif zone == 2:
                    patch1[:, :, :] = 0
                    patch1[:, x1:x2, y1:y2] = im_full[:, x1:x2, y1:y2]
                    patch2[:, x1:x2, y1:y2] = 0
            else:
                patch1 = self.transform_train(im_full)
                patch2 = self.transform_train(im_full)

            if patches1 is None:
                patches1 = patch1.unsqueeze(0)
            else:
                patches1 = torch.cat([patches1, patch1.unsqueeze(0)], dim=0)

            if patches2 is None:
                patches2 = patch2.unsqueeze(0)
            else:
                patches2 = torch.cat([patches2, patch2.unsqueeze(0)], dim=0)

        images = torch.cat([patches2, patches2], dim=0)

        return images

    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            ret_indexes = np.arange(0, self.n_added_so_far).tolist()
        else:
            ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        if len(ret_indexes) == 0:
            return torch.Tensor((0)), torch.Tensor((0))
        im1 = self.get_cuts(ret_indexes)
        im2 = torch.cat([self.buffer_imgs[ret_indexes], self.buffer_imgs[ret_indexes]])

        return im1, im2