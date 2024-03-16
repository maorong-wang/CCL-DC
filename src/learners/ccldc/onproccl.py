
"""Code adapted from https://github.com/weilllllls/OnPro
"""
import torch
import wandb
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import math
import pickle
import itertools
import torch.cuda.amp as amp
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale, RandomMixUpV2
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.distributions import Categorical
from sync_batchnorm import patch_replication_callback

from src.misc import onpro_transforms as TL
from src.misc.onpro_transforms import Rotation
from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.utils import name_match
from src.models.resnet import OCMResnet
from src.models.onproresnet import resnet18, imagenet_resnet18
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.buffers.onprobuf import Buffer

device = get_device()

pdist = torch.nn.PairwiseDistance(p=2).cuda()

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

class OPELoss(nn.Module):
    def __init__(self, class_per_task, temperature=0.5, only_old_proto=False):
        super(OPELoss, self).__init__()
        self.class_per_task = class_per_task
        self.temperature = temperature
        self.only_old_proto = only_old_proto
        

    def cal_prototype(self, z1, z2, y, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((end_i, dim), device=z1.device)
        for i in range(start_i, end_i):
            indices = (y == i)
            if not any(indices):
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]

            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)

            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        return current_classes_mean_z1, current_classes_mean_z2

    def forward(self, z1, z2, labels, task_id, is_new=False):
        prototype_z1, prototype_z2 = self.cal_prototype(z1, z2, labels, task_id)

        if not self.only_old_proto or is_new:
            nonZeroRows = torch.abs(prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = prototype_z2[nonZeroRows]
        else:
            old_prototype_z1 = prototype_z1[:task_id * self.class_per_task]
            old_prototype_z2 = prototype_z2[:task_id * self.class_per_task]
            nonZeroRows = torch.abs(old_prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = old_prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = old_prototype_z2[nonZeroRows]

        nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
        nonZero_prototype_z2 = F.normalize(nonZero_prototype_z2)

        device = nonZero_prototype_z1.device

        class_num = nonZero_prototype_z1.size(0)
        z = torch.cat((nonZero_prototype_z1, nonZero_prototype_z2), dim=0)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = torch.zeros((2 * class_num, 2 * class_num), dtype=torch.bool, device=device)
        pos_mask[:, class_num:].fill_diagonal_(True)
        pos_mask[class_num:, :].fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss, prototype_z1, prototype_z2

class AdaptivePrototypicalFeedback(nn.Module):
    def __init__(self, buffer, mixup_base_rate, mixup_p, mixup_lower, mixup_upper, mixup_alpha,
                 class_per_task):
        super(AdaptivePrototypicalFeedback, self).__init__()
        self.buffer = buffer
        self.class_per_task = class_per_task
        self.mixup_base_rate = mixup_base_rate
        self.mixup_p = mixup_p
        self.mixup_lower = mixup_lower
        self.mixup_upper = mixup_upper
        self.mixup_alpha = mixup_alpha
        self.mixup = RandomMixUpV2(p=mixup_p, lambda_val=(mixup_lower, mixup_upper),
                                   data_keys=["input", "class"]).cuda()

    def forward(self, mem_x, mem_y, buffer_batch_size, classes_mean, task_id):
        base_rate = self.mixup_base_rate
        base_sample_num = int(buffer_batch_size * base_rate)

        indices = torch.from_numpy(np.random.choice(mem_x.size(0), base_sample_num, replace=False)).cuda()
        mem_x_base = mem_x[indices]
        mem_y_base = mem_y[indices]

        mem_x_base_mix, mem_y_base_mix = self.mixup(mem_x_base, mem_y_base)

        prob_sample_num = buffer_batch_size - base_sample_num
        if prob_sample_num != 0:
            nonZeroRows = torch.abs(classes_mean).sum(dim=1) > 0
            ZeroRows = torch.abs(classes_mean).sum(dim=1) == 0
            class_num = classes_mean.shape[0]
            nonZero_class = torch.arange(class_num)[nonZeroRows.cpu()]
            Zero_class = torch.arange(class_num)[ZeroRows.cpu()]

            classes_mean = classes_mean[nonZeroRows]

            dis = torch.pdist(classes_mean)  # K*(K-1)/2

            sample_p = F.softmax(1 / dis, dim=0)

            mix_x_by_prob, mix_y_by_prob = self.make_mix_pair(sample_p, prob_sample_num, nonZero_class, Zero_class,
                                                              task_id)

            mem_x = torch.cat([mem_x_base_mix, mix_x_by_prob])
            mem_y_mix = torch.cat([mem_y_base_mix, mix_y_by_prob])

            origin_mem_y, mix_mem_y, mix_lam = mem_y_mix[:, 0], mem_y_mix[:, 1], mem_y_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
        else:
            mem_x = mem_x_base_mix
            origin_mem_y, mix_mem_y, mix_lam = mem_y_base_mix[:, 0], mem_y_base_mix[:, 1], mem_y_base_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
            mem_y_mix = mem_y_base_mix

        return mem_x, mem_y, mem_y_mix

    def make_mix_pair(self, sample_prob, prob_sample_num, nonZero_class, Zero_class, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        sample_num_per_class_pair = (sample_prob * prob_sample_num).round()
        diff_num = int((prob_sample_num - sample_num_per_class_pair.sum()).item())
        if diff_num > 0:
            add_idx = torch.randperm(sample_num_per_class_pair.shape[0])[:diff_num]
            sample_num_per_class_pair[add_idx] += 1
        elif diff_num < 0:
            reduce_idx = torch.nonzero(sample_num_per_class_pair, as_tuple=True)[0]
            reduce_idx_ = torch.randperm(reduce_idx.shape[0])[:-diff_num]
            reduce_idx = reduce_idx[reduce_idx_]
            sample_num_per_class_pair[reduce_idx] -= 1

        assert sample_num_per_class_pair.sum() == prob_sample_num

        x_indices = torch.arange(self.buffer.x.shape[0])
        y_indices = torch.arange(self.buffer.y.shape[0])
        y = self.buffer.y.cuda()
        _, y = torch.max(y, dim=1)

        class_x_list = []
        class_y_list = []
        class_id_map = {}
        for task_id in range(start_i, end_i):
            if task_id in Zero_class:
                continue
            indices = (y == task_id)
            if not any(indices):
                continue

            class_x_list.append(x_indices[indices.cpu()])
            class_y_list.append(y_indices[indices.cpu()])
            class_id_map[task_id] = len(class_y_list) - 1

        mix_images = []
        mix_labels = []

        for idx, class_pair in enumerate(itertools.combinations(nonZero_class.tolist(), 2)):
            n = int(sample_num_per_class_pair[idx].item())
            if n == 0:
                continue
            first_class_y = class_pair[0]
            second_class_y = class_pair[1]

            if first_class_y not in class_id_map:
                first_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                first_class_y = int(first_class_y)
            if second_class_y not in class_id_map:
                second_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                second_class_y = int(second_class_y)

            first_class_idx = class_id_map[first_class_y]
            second_class_idx = class_id_map[second_class_y]

            first_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[first_class_idx].tolist(), n)).long()
            second_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[second_class_idx].tolist(), n)).long()

            first_class_x = self.buffer.x[first_class_sample_idx].cuda()
            second_class_x = self.buffer.x[second_class_sample_idx].cuda()

            mix_pair, mix_lam = self.mixup_by_input_pair(first_class_x, second_class_x, n)
            mix_y = torch.zeros(n, 3)
            mix_y[:, 0] = first_class_y
            mix_y[:, 1] = second_class_y
            mix_y[:, 2] = mix_lam

            mix_images.append(mix_pair)
            mix_labels.append(mix_y)

        mix_images_by_prob = torch.cat(mix_images).cuda()
        mix_labels_by_prob = torch.cat(mix_labels).cuda()

        return mix_images_by_prob, mix_labels_by_prob

    def mixup_by_input_pair(self, x1, x2, n):
        if torch.rand([]) <= self.mixup_p:
            lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, n)).cuda()
            lam_ = lam.unsqueeze(0).unsqueeze(0).unsqueeze(0).view(-1, 1, 1, 1)
        else:
            lam = 0
            lam_ = 0
        lam = torch.tensor(lam, dtype=x1.dtype)
        lam_ = torch.tensor(lam_, dtype=x1.dtype)
        image = (1 - lam_) * x1 + lam_ * x2
        return image, lam

def Supervised_NT_xent_n(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM : https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """
    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    eye = torch.zeros((B * chunk, B * chunk), dtype=torch.bool, device=device)
    eye[:, :].fill_diagonal_(True)
    sim_matrix = torch.exp(sim_matrix / temperature) * (~eye)

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    loss2 = 2 * torch.sum(Mask1 * sim_matrix) / (2 * B)
    loss1 = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss1 + loss2


def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM: https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """

    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    sim_matrix = torch.exp(sim_matrix / temperature)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    return torch.sum(Mask1 * sim_matrix) / (2 * B)

class OnProCCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.oop_base = self.params.n_classes
        self.oop = 16
        self.n_classes_num = self.params.n_classes
        self.fea_dim = self.params.proj_dim
        self.classes_mean = torch.zeros((self.n_classes_num, self.fea_dim), requires_grad=False).cuda()
        self.class_per_task = int(self.params.n_classes / self.params.n_tasks)
        self.class_holder = []
        self.ins_t = 0.07
        self.proto_t = 0.5
        self.previous_model = None
        self.kd_lambda = self.params.kd_lambda
        self.mem = None

        self.iter = 0

        self.buffer = Buffer(
                mem_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
                device=device
            )
        self.buffer_batch_size = 64
        self.buffer_per_class = 7

        self.OPELoss = OPELoss(self.class_per_task, temperature=self.proto_t)

        if self.n_classes_num == 10:
            self.sim_lambda = 0.5
            self.total_samples = 10000
            self.mixup_base_rate = 0.75
            self.mixup_p = 0.6
        elif self.n_classes_num == 100:
            self.sim_lambda = 1.0
            self.total_samples = 5000
            self.mixup_base_rate = 0.9
            self.mixup_p = 0.2
        elif self.n_classes_num == 200:
            self.sim_lambda = 1.0
            self.total_samples = 10000
            self.mixup_base_rate = 0.9
            self.mixup_p = 0.2
        self.print_num = self.total_samples // 10

        hflip = TL.HorizontalFlipLayer().cuda()
        with torch.no_grad():
            input_size = [3, self.params.img_size, self.params.img_size]
            resize_scale = (0.3, 1.0)
            color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.transform = torch.nn.Sequential(
                hflip,
                color_gray,
                resize_crop)

        self.APF = AdaptivePrototypicalFeedback(self.buffer, self.mixup_base_rate, self.mixup_p, 0, 0.6,
                                  0.4, self.class_per_task)

        self.scaler1 = amp.GradScaler()
        self.scaler2 = amp.GradScaler()

    def load_criterion(self):
        pass
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = resnet18(nclasses=self.params.n_classes)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            # for imagenet experiments, the 80 gig memory is not enough, so do it in a data parallel way
            model = MyDataParallel(imagenet_resnet18(nclasses=self.params.n_classes))
            patch_replication_callback(model)
            return model.to(device)

    def train(self, dataloader, **kwargs):
        self.model1.train()
        self.model2.train()
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)


    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        dataloaders = kwargs.get('dataloaders', None)
        task_name = kwargs.get('task_name', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch[0], batch[1]
                ybuf = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y])
                y = ybuf.to(device)
                batch_x = x
                batch_y = y
                num_d += x.shape[0]

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder:
                        self.class_holder.append(Y[j].detach())

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    # x = x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_aug = self.transform(rot_x)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)

                    rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)

                    ### model1
                    features, projections = self.model1(images_pair, is_simclr=True)
                    projections = F.normalize(projections)

                    # instance-wise contrastive loss in OCM
                    features = F.normalize(features)
                    dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_len = projections.shape[1]

                    sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix += torch.mm(projections, projections.t())

                    ins_loss1 = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                    
                    if batch_idx != 0:
                        buffer_x, buffer_y = self.sample_from_buffer_for_prototypes()
                        # buffer_x.requires_grad = True
                        buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                        buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)

                        proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss_m1(buffer_x, buffer_y, buffer_x_pair, task_id)
                    else:
                        proto_seen_loss = 0

                    z = projections[:rot_x.shape[0]]
                    zt = projections[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss1 = proto_new_loss + proto_seen_loss

                    y_pred = self.model1.logits(self.transform(x))
                    ce1 = F.cross_entropy(y_pred, y)

                    loss1 = ce1 + ins_loss1 + OPE_loss1

                    ### model2
                    features, projections = self.model2(images_pair, is_simclr=True)
                    projections = F.normalize(projections)

                    # instance-wise contrastive loss in OCM
                    features = F.normalize(features)
                    dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_len = projections.shape[1]

                    sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix += torch.mm(projections, projections.t())

                    ins_loss2 = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                    
                    if batch_idx != 0:
                        buffer_x, buffer_y = self.sample_from_buffer_for_prototypes()
                        # buffer_x.requires_grad = True
                        buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                        buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)

                        proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss_m2(buffer_x, buffer_y, buffer_x_pair, task_id)
                    else:
                        proto_seen_loss = 0

                    z = projections[:rot_x.shape[0]]
                    zt = projections[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss2 = proto_new_loss + proto_seen_loss

                    y_pred = self.model2.logits(self.transform(x))
                    ce2 = F.cross_entropy(y_pred, y)

                    loss2 = ce2 + ins_loss2 + OPE_loss2

                    # Distillation loss

                    # Combined batch
                    combined_x = torch.cat([batch_x.to(device)], dim=0)
                    combined_y = torch.cat([batch_y.to(device)], dim=0)
                    
                    # Augment
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug = self.transform_3(combined_aug2)
                    # Inference

                    logits1 = self.model1.logits(combined_aug)
                    logits2 = self.model2.logits(combined_aug)

                    logits1_vanilla = self.model1.logits(combined_x)
                    logits2_vanilla = self.model2.logits(combined_x)

                    logits1_step1 = self.model1.logits(combined_aug1)
                    logits2_step1 = self.model2.logits(combined_aug1)

                    logits1_step2 = self.model1.logits(combined_aug2)
                    logits2_step2 = self.model2.logits(combined_aug2)
                    # Cls Loss
                    loss_ce = nn.CrossEntropyLoss()(logits1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step2, combined_y.long())
                    loss_ce2 = nn.CrossEntropyLoss()(logits2, combined_y.long()) + nn.CrossEntropyLoss()(logits2_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step2, combined_y.long())

                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # loss
                    loss1_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss1  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                self.scaler1.scale(loss1_sum).backward()
                self.scaler1.step(self.optim1)
                self.scaler1.update()
                self.optim1.zero_grad()

                self.scaler2.scale(loss2_sum).backward()
                self.scaler2.step(self.optim2)
                self.scaler2.update()
                self.optim2.zero_grad()
                print(f"Loss (Peer1) : {loss1_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {batch_idx}", end="\r")

                self.iter += 1

                self.buffer.add_reservoir(x=batch[0].detach(), y=ybuf.detach(), logits=None, t=task_id)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss1_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                    )


        else:        
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch[0], batch[1]
                ybuf = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y])
                y = ybuf.to(device)
                batch_x = x
                batch_y = y
                num_d += x.shape[0]

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder:
                        self.class_holder.append(Y[j].detach())

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    # x = x.requires_grad_()
                    buffer_batch_size = min(self.buffer_batch_size, self.buffer_per_class * len(self.class_holder))

                    ori_mem_x, ori_mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)
                    ori_mem_x, ori_mem_y = ori_mem_x.to(device), ori_mem_y.to(device)
                    if batch_idx != 0:
                        mem_x, mem_y, mem_y_mix = self.APF(ori_mem_x, ori_mem_y, buffer_batch_size, self.classes_mean, task_id)
                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix = torch.zeros(rot_sim_labels_r.shape[0], 3).cuda()
                        rot_mem_y_mix[:, 0] = torch.cat([mem_y_mix[:, 0] + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix[:, 1] = torch.cat([mem_y_mix[:, 1] + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix[:, 2] = mem_y_mix[:, 2].repeat(self.oop)
                    else:
                        mem_x = ori_mem_x
                        mem_y = ori_mem_y

                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)

                    # mem_x = mem_x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_r = Rotation(mem_x)
                    rot_x_aug = self.transform(rot_x)
                    rot_x_r_aug = self.transform(rot_x_r)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)
                    images_pair_r = torch.cat([rot_x_r, rot_x_r_aug], dim=0)

                    all_images = torch.cat((images_pair, images_pair_r), dim=0)

                    ### model1
                    features, projections = self.model1(all_images, is_simclr=True)

                    projections_x = projections[:images_pair.shape[0]]
                    projections_x_r = projections[images_pair.shape[0]:]

                    projections_x = F.normalize(projections_x)
                    projections_x_r = F.normalize(projections_x_r)

                    # instance-wise contrastive loss in OCM
                    features_x = F.normalize(features[:images_pair.shape[0]])
                    features_x_r = F.normalize(features[images_pair.shape[0]:])

                    dim_diff = features_x.shape[1] - projections_x.shape[1]
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_begin_r = torch.randperm(dim_diff)[0]
                    dim_len = projections_x.shape[1]

                    sim_matrix = self.sim_lambda * torch.matmul(projections_x, features_x[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r, features_x_r[:, dim_begin_r:dim_begin_r + dim_len].t())

                    sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
                    sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

                    loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.ins_t)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                    
                    ins_loss1 = loss_sim_r + loss_sim

                    y_pred = self.model1.logits(self.transform(mem_x))

                    buffer_x = ori_mem_x
                    buffer_y = ori_mem_y
                    buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)
                    proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss_m1(buffer_x, buffer_y, buffer_x_pair, task_id)

                    z = projections_x[:rot_x.shape[0]]
                    zt = projections_x[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss1 = proto_new_loss + proto_seen_loss

                    if batch_idx != 0:
                        ce1 = self.loss_mixup(y_pred, mem_y_mix)
                    else:
                        ce1 = F.cross_entropy(y_pred, mem_y)

                    loss1 = ce1 + ins_loss1 + OPE_loss1


                    ### model2
                    features, projections = self.model2(all_images, is_simclr=True)

                    projections_x = projections[:images_pair.shape[0]]
                    projections_x_r = projections[images_pair.shape[0]:]

                    projections_x = F.normalize(projections_x)
                    projections_x_r = F.normalize(projections_x_r)

                    # instance-wise contrastive loss in OCM
                    features_x = F.normalize(features[:images_pair.shape[0]])
                    features_x_r = F.normalize(features[images_pair.shape[0]:])

                    dim_diff = features_x.shape[1] - projections_x.shape[1]
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_begin_r = torch.randperm(dim_diff)[0]
                    dim_len = projections_x.shape[1]

                    sim_matrix = self.sim_lambda * torch.matmul(projections_x, features_x[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r, features_x_r[:, dim_begin_r:dim_begin_r + dim_len].t())

                    sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
                    sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

                    loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.ins_t)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                    
                    ins_loss2 = loss_sim_r + loss_sim

                    y_pred = self.model2.logits(self.transform(mem_x))

                    buffer_x = ori_mem_x
                    buffer_y = ori_mem_y
                    buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)
                    proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss_m2(buffer_x, buffer_y, buffer_x_pair, task_id)

                    z = projections_x[:rot_x.shape[0]]
                    zt = projections_x[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss2 = proto_new_loss + proto_seen_loss

                    if batch_idx != 0:
                        ce2 = self.loss_mixup(y_pred, mem_y_mix)
                    else:
                        ce2 = F.cross_entropy(y_pred, mem_y)

                    loss2 = ce2 + ins_loss2 + OPE_loss2

                    torch.cuda.empty_cache()

                    # Distillation loss
                    # Combined batch
                    combined_x = torch.cat([batch_x.to(device), mem_x.to(device)], dim=0)
                    combined_y = torch.cat([batch_y.to(device), mem_y.to(device)], dim=0)
                    # combined_x = mem_x
                    # combined_y = mem_y
                    
                    # Augment
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug = self.transform_3(combined_aug2)
                    # Inference

                    logits1 = self.model1.logits(combined_aug)
                    logits2 = self.model2.logits(combined_aug)

                    logits1_vanilla = self.model1.logits(combined_x)
                    logits2_vanilla = self.model2.logits(combined_x)

                    logits1_step1 = self.model1.logits(combined_aug1)
                    logits2_step1 = self.model2.logits(combined_aug1)

                    logits1_step2 = self.model1.logits(combined_aug2)
                    logits2_step2 = self.model2.logits(combined_aug2)
                    # Cls Loss
                    loss_ce = nn.CrossEntropyLoss()(logits1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step2, combined_y.long())
                    loss_ce2 = nn.CrossEntropyLoss()(logits2, combined_y.long()) + nn.CrossEntropyLoss()(logits2_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step2, combined_y.long())

                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # Total Loss
                    loss1_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss1  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                self.scaler1.scale(loss1_sum).backward()
                self.scaler1.step(self.optim1)
                self.scaler1.update()
                self.optim1.zero_grad()

                self.scaler2.scale(loss2_sum).backward()
                self.scaler2.step(self.optim2)
                self.scaler2.update()
                self.optim2.zero_grad()
                print(f"Loss (Peer1) : {loss1_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {batch_idx}", end="\r")

                self.iter += 1

                self.buffer.add_reservoir(x=batch[0].detach(), y=ybuf.detach(), logits=None, t=task_id)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss1_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                    )


    def cal_buffer_proto_loss_m1(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea, buffer_z = self.model1(buffer_x_pair, is_simclr=True)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm

    def cal_buffer_proto_loss_m2(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea, buffer_z = self.model2(buffer_x_pair, is_simclr=True)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm

    def sample_from_buffer_for_prototypes(self):
        b_num = self.buffer.x.shape[0]
        if b_num <= self.buffer_batch_size:
            buffer_x = self.buffer.x
            buffer_y = self.buffer.y
            _, buffer_y = torch.max(buffer_y, dim=1)
        else:
            buffer_x, buffer_y, _ = self.buffer.sample(self.buffer_batch_size, exclude_task=None)

        return buffer_x, buffer_y

    def loss_mixup(self, logits, y):
        criterion = F.cross_entropy
        loss_a = criterion(logits, y[:, 0].long(), reduction='none')
        loss_b = criterion(logits, y[:, 1].long(), reduction='none')
        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

    def encode_logits(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                feat1 = self.model1.logits(self.transform_test(inputs))
                feat2 = self.model2.logits(self.transform_test(inputs))
                feat_ens = (feat1 + feat2) / 2.0

                preds_ens = feat_ens.argmax(dim=1)
                preds_1 = feat1.argmax(dim=1)
                preds_2 = feat2.argmax(dim=1)
                
                preds_ens = torch.tensor([self.params.labels_order[i] for i in preds_ens])
                preds_1 = torch.tensor([self.params.labels_order[i] for i in preds_1])
                preds_2 = torch.tensor([self.params.labels_order[i] for i in preds_2])

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat1 = preds_1.cpu().numpy()
                    all_feat2 = preds_2.cpu().numpy()
                    all_feat_ens = preds_ens.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat_ens = np.hstack([all_feat_ens, preds_ens.cpu().numpy()])
                    all_feat1 = np.hstack([all_feat1, preds_1.cpu().numpy()])
                    all_feat2 = np.hstack([all_feat2, preds_2.cpu().numpy()])
        return all_feat_ens, all_feat1, all_feat2, all_labels

    def evaluate_clustering(self, dataloaders, task_id, **kwargs):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            mem_representations_ens, mem_representations_n1, mem_representations_n2, mem_labels = self.get_mem_rep_labels_ens(use_proj=self.params.eval_proj)

            mem_labels = torch.tensor([self.params.labels_order[i] for i in mem_labels])
            # UMAP visualization
            # reduction = self.umap_reduction(mem_representations_ens.cpu().numpy())
            # plt.figure()
            # figure = plt.scatter(reduction[:, 0], reduction[:, 1], c=mem_labels, cmap='Spectral', s=1)
            # if not self.params.no_wandb:
            #     wandb.log({
            #         "ens_umap": wandb.Image(figure),
            #         "task_id": task_id
            #     })

            classifiers_ens = self.init_classifiers()
            classifiers_n1 = self.init_classifiers()
            classifiers_n2 = self.init_classifiers()
            classifiers_ens = self.fit_classifiers(classifiers=classifiers_ens, representations=mem_representations_ens, labels=mem_labels)
            classifiers_n1 = self.fit_classifiers(classifiers=classifiers_n1, representations=mem_representations_n1, labels=mem_labels)
            classifiers_n2 = self.fit_classifiers(classifiers=classifiers_n2, representations=mem_representations_n2, labels=mem_labels)
            
            accs = []
            accs1 = []
            accs2 = []
            representations_ens = {}
            representations_n1 = {}
            representations_n2 = {}
            targets_ens = {}
            targets_n1 = {}
            targets_n2 = {}
            preds_ens = []
            preds_1 = []
            preds_2 = []
            all_targets = []
            tag = 'stu'

            for j in range(task_id + 1):
                test_representation, test_representation_n1, test_representation_n2, test_targets = self.encode_fea(dataloaders[f"test{j}"])
                representations_ens[f"test{j}"] = test_representation
                targets_ens[f"test{j}"] = test_targets
                representations_n1[f"test{j}"] = test_representation_n1
                targets_n1[f"test{j}"] = test_targets
                representations_n2[f"test{j}"] = test_representation_n2
                targets_n2[f"test{j}"] = test_targets

                test_preds_ens = classifiers_ens[0].predict(representations_ens[f'test{j}'])
                test_preds_1 = classifiers_n1[0].predict(representations_n1[f'test{j}'])
                test_preds_2 = classifiers_n2[0].predict(representations_n2[f'test{j}'])

                acc_ens = accuracy_score(targets_ens[f"test{j}"], test_preds_ens) 
                acc_1 = accuracy_score(targets_n1[f"test{j}"], test_preds_1) 
                acc_2 = accuracy_score(targets_n2[f"test{j}"], test_preds_2) 

                accs.append(acc_ens)
                accs1.append(acc_1)
                accs2.append(acc_2)
                # Wandb logs
                if not self.params.no_wandb:
                    preds_ens = np.concatenate([preds_ens, test_preds_ens])
                    preds_1 = np.concatenate([preds_1, test_preds_1])
                    preds_2 = np.concatenate([preds_2, test_preds_2])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ncm_ens_acc_{j}": acc_ens,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"ncm_net1_acc_{j}": acc_1,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"ncm_net2_acc_{j}": acc_2,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds_ens = [self.params.labels_order.index(int(i)) for i in preds_ens]
                preds_1 = [self.params.labels_order.index(int(i)) for i in preds_1]
                preds_2 = [self.params.labels_order.index(int(i)) for i in preds_2]
                cm_ens = np.log(1 + confusion_matrix(all_targets, preds_ens))
                cm_1 = np.log(1 + confusion_matrix(all_targets, preds_1))
                cm_2 = np.log(1 + confusion_matrix(all_targets, preds_2))
                fig = plt.matshow(cm_ens)
                wandb.log({
                        tag + f"ncm_ens_cm": fig,
                        "task_id": task_id
                    })
                fig = plt.matshow(cm_1)
                wandb.log({
                        tag + f"ncm_net1_cm": fig,
                        "task_id": task_id
                    })
                fig = plt.matshow(cm_2)
                wandb.log({
                        tag + f"ncm_net2_cm": fig,
                        "task_id": task_id
                    })
                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)
                accs1.append(np.nan)
                accs2.append(np.nan)

            self.results_clustering.append(accs)
            self.results_1.append(accs1)
            self.results_2.append(accs2)
            
            line = forgetting_line(pd.DataFrame(self.results_clustering), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_clustering_forgetting.append(line)

            line = forgetting_line(pd.DataFrame(self.results_1), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_1.append(line)

            line = forgetting_line(pd.DataFrame(self.results_2), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_2.append(line)

            return np.nanmean(self.results_clustering[-1]), np.nanmean(self.results_clustering_forgetting[-1]), np.nanmean(self.results_1[-1]), np.nanmean(self.results_forgetting_1[-1]), np.nanmean(self.results_2[-1]), np.nanmean(self.results_forgetting_2[-1])

    def save(self, path):
        lg.debug("Saving checkpoint...")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
        torch.save(self.model1.state_dict(), os.path.join(path, 'model1.pth'))
        torch.save(self.model2.state_dict(), os.path.join(path, 'model2.pth'))

        self.mem = self.buffer.get_all()
        
        torch.save(self.mem, os.path.join(path, 'memory.pth'))

    def resume(self, path):
        self.model1.load_state_dict(torch.load(os.path.join(path, 'model1.pth')))
        self.model2.load_state_dict(torch.load(os.path.join(path, 'model2.pth')))
        self.mem = torch.load(os.path.join(path, 'memory.pth'))
        torch.cuda.empty_cache()

    def get_mem_rep_labels_ens_withmem(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: 
            self.model1.eval()
            self.model2.eval()
        mem_imgs, mem_labels = self.mem
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        n1_reps = []
        n2_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b1 = self.model1(mem_imgs_b)
            mem_representations_b2 = self.model2(mem_imgs_b)
            mem_representations_b = (mem_representations_b1 + mem_representations_b2) / 2.0
            n1_reps.append(mem_representations_b1)
            n2_reps.append(mem_representations_b2)
            all_reps.append(mem_representations_b)
        mem_representations_ens = torch.cat(all_reps, dim=0)
        mem_representations_n1 = torch.cat(n1_reps, dim=0)
        mem_representations_n2 = torch.cat(n2_reps, dim=0)
        return mem_representations_ens, mem_representations_n1, mem_representations_n2, mem_labels

    def get_entropy(self, dataloaders, task_id):
        trainloader = dataloaders[f"train{task_id}"]
        testloader = dataloaders[f"test{task_id}"]

        train_ce = 0
        train_en = 0
        test_ce = 0
        test_en = 0
        samples = 0

        self.model1.eval()
        self.model2.eval()

        for i, batch in enumerate(trainloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            labels = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in labels]).to(device)
            samples += inputs.shape[0]
            outputs = self.model1.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            train_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            train_en += Categorical(probs=prob).entropy().sum().item()

        train_ce /= samples
        train_en /= samples

        samples = 0

        for i, batch in enumerate(testloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            labels = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in labels]).to(device)
            samples += inputs.shape[0]
            outputs = self.model1.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            test_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            test_en += Categorical(probs=prob).entropy().sum().item()

        test_ce /= samples
        test_en /= samples

        self.model1.train()
        self.model2.train()
        return train_ce, train_en, test_ce, test_en