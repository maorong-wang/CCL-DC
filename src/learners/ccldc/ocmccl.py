import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import math
import torch.cuda.amp as amp
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale
from sync_batchnorm import patch_replication_callback

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import OCMResnet, ImageNet_OCMResnet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device

device = get_device()
scaler = amp.GradScaler()
scaler2 = amp.GradScaler()

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

class OCMCCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        # When task id need to be infered
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        self.old_classes = torch.LongTensor(size=(0,)).to(device)
        self.lag_task_change = 100

        self.oop = 16

        self.kd_lambda = self.params.kd_lambda

        self.results = []
        self.results_forgetting = []
        self.results_1 = []
        self.results_forgetting_1 = []
        self.results_2 = []
        self.results_forgetting_2 = []

        self.iter = 0

    def rotation(self, x):
        X = self.rot_inner_all(x)#, 1, 0)
        return torch.cat((X,torch.rot90(X,2,(2,3)),torch.rot90(X,1,(2,3)),torch.rot90(X,3,(2,3))),dim=0)

    def rot_inner_all(self, x):
        num=x.shape[0]
        R=x.repeat(4,1,1,1)
        a=x.permute(0,1,3,2)
        a = a.view(num,3, 2, self.params.img_size // 2 , self.params.img_size)
        a = a.permute(2,0, 1, 3, 4)
        s1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2=a[1]#.permute(1,0, 2, 3)
        a= torch.rot90(a, 2, (3, 4))
        s1_1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2_2=a[1]#.permute(1,0, 2, 3)R[3*num:]

        R[num:2*num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[3*num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        return R

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            return OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            # for imagenet experiments, the 80 gig memory is not enough, so do it in a data parallel way
            model = MyDataParallel(ImageNet_OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ))
            patch_replication_callback(model)
            return model.to(device)

    def load_criterion(self):
        return SupConLoss(self.params.temperature) 
    
    def normalize(self, x, dim=1, eps=1e-8):
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def get_similarity_matrix(self, outputs, chunk=2, multi_gpu=False):
        '''
            Compute similarity matrix
            - outputs: (B', d) tensor for B' = B * chunk
            - sim_matrix: (B', B') tensor
        '''
        sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')#这里是sim(z(x),z(x'))
        return sim_matrix
    
    def Supervised_NT_xent_n(self, sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        eye = torch.eye(B * chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)
        return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) +  loss1#+1*loss2
    
    def Supervised_NT_xent_uni(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature)# * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

    def Supervised_NT_xent_pre(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels#.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature) #* (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        dataloaders = kwargs.get('dataloaders', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            for j, batch in enumerate(dataloader):
                # Stream data
                self.model1.train()
                self.model2.train()

                self.optim1.zero_grad()
                self.optim2.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    self.stream_idx += len(batch_x)

                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    # Augment
                    aug1 = self.rotation(batch_x)
                    aug2 = self.transform_train(aug1)
                    images_pair = torch.cat([aug1, aug2], dim=0)

                    # labels rotations or something
                    rot_sim_labels = torch.cat([batch_y.to(device) + 1000 * i for i in range(self.oop)], dim=0)

                    # Inference - model1
                    feature_map, output_aux = self.model1(images_pair, is_simclr=True)
                    simclr = self.normalize(output_aux)
                    feature_map_out = self.normalize(feature_map[:images_pair.shape[0]])
                    
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim1 = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                    temperature=0.07)
                    lo1 = 1 * loss_sim1
                    y_pred = self.model1.logits(self.transform_train(batch_x))

                    loss = 1*F.cross_entropy(y_pred, batch_y.long())+1*lo1

                    # Inference - model2
                    feature_map2, output_aux2 = self.model2(images_pair, is_simclr=True)
                    simclr2 = self.normalize(output_aux2)
                    feature_map_out2 = self.normalize(feature_map2[:images_pair.shape[0]])

                    num2 = feature_map_out2.shape[1] - simclr2.shape[1]
                    id2 = torch.randperm(num2)[0]
                    size = simclr2.shape[1]
                    sim_matrix2 = 1*torch.matmul(simclr2, feature_map_out2[:, id2 :id2+ 1 * size].t())

                    sim_matrix2 += 1 * self.get_similarity_matrix(simclr2)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim2 = self.Supervised_NT_xent_n(sim_matrix2, labels=rot_sim_labels,
                                                    temperature=0.07)
                    lo2 = 1 * loss_sim2
                    y_pred2 = self.model2.logits(self.transform_train(batch_x))

                    loss2 = 1*F.cross_entropy(y_pred2, batch_y.long())+1*lo2

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

                    # Total Loss
                    loss_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                scaler.scale(loss_sum).backward()
                scaler.step(self.optim1)
                scaler.update()

                scaler2.scale(loss2_sum).backward()
                scaler2.step(self.optim2)
                scaler2.update()

                self.iter += 1

                self.loss = loss_sum.item()
                print(f"Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {j}", end="\r")
                # Update buffer
                self.buffer.update(imgs=batch_x.detach().cpu(), labels=batch_y.detach().cpu())

                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                    )
        else:
            for j, batch in enumerate(dataloader):
                self.model1.train()
                self.model2.train()

                self.optim1.zero_grad()
                self.optim2.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # print(scaler.get_scale())
                    # Stream data
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    self.stream_idx += len(batch_x)
                    
                    # Augment
                    aug1_batch = self.rotation(batch_x)
                    aug2_batch = self.transform_train(aug1_batch)
                    aug1_mem = self.rotation(mem_x)
                    aug2_mem = self.transform_train(aug1_mem)

                    images_pair_batch = torch.cat((aug1_batch, aug2_batch), dim=0)
                    images_pair_mem = torch.cat([aug1_mem, aug2_mem], dim=0)

                    # Inference- model1
                    t = torch.cat((images_pair_batch, images_pair_mem),dim=0)
                    feature_map, u = self.model1(t, is_simclr=True)
                    pre_u = self.previous_model(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch = self.normalize(feature_map[:images_pair_batch.shape[0]])
                    feature_map_out_mem = self.normalize(feature_map[images_pair_batch.shape[0]:])
                    
                    images_out = u[:images_pair_batch.shape[0]]
                    images_out_r = u[images_pair_batch.shape[0]:]
                    pre_u = self.normalize(pre_u)
                    simclr = self.normalize(images_out)
                    simclr_r = self.normalize(images_out_r)

                    rot_sim_labels = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)

                    num1 = feature_map_out_batch.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id2=torch.randperm(num1)[0]

                    size = simclr.shape[1]

                    sim_matrix = 0.5*torch.matmul(simclr, feature_map_out_batch[:, id1:id1 + size].t())
                    sim_matrix_r = 0.5*torch.matmul(simclr_r,
                                                    feature_map_out_mem[:, id2:id2 + size].t())


                    sim_matrix += 0.5 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r += 0.5 * self.get_similarity_matrix(simclr_r)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):    
                        sim_matrix_r_pre = torch.matmul(simclr_r[:aug1_mem.shape[0]],pre_u.t())

                    loss_sim_r = self.Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
                    loss_sim_pre = self.Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                    y_label = self.model1.logits(self.transform_train(mem_x))

                    y_label_pre = self.previous_model(self.transform_train(mem_x))
                    loss = 1 * F.cross_entropy(y_label, mem_y) + lo1  + \
                        1 * F.mse_loss(y_label_pre[:, self.old_classes.long()], y_label[:, self.old_classes.long()])
                    
                    #Inference - model2
                    feature_map2, u2 = self.model2(t, is_simclr=True)
                    pre_u2 = self.previous_model2(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch2 = self.normalize(feature_map2[:images_pair_batch.shape[0]])
                    feature_map_out_mem2 = self.normalize(feature_map2[images_pair_batch.shape[0]:])

                    images_out2 = u2[:images_pair_batch.shape[0]]
                    images_out_r2 = u2[images_pair_batch.shape[0]:]
                    pre_u2 = self.normalize(pre_u2)
                    simclr2 = self.normalize(images_out2)
                    simclr_r2 = self.normalize(images_out_r2)

                    rot_sim_labels2 = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r2 = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)

                    num12 = feature_map_out_batch2.shape[1] - simclr2.shape[1]
                    id12 = torch.randperm(num12)[0]
                    id22=torch.randperm(num12)[0]

                    size2 = simclr2.shape[1]

                    sim_matrix2 = 0.5*torch.matmul(simclr2, feature_map_out_batch2[:, id12:id12 + size2].t())
                    sim_matrix_r2 = 0.5*torch.matmul(simclr_r2,
                                                    feature_map_out_mem2[:, id22:id22 + size2].t())
                    
                    sim_matrix2 += 0.5 * self.get_similarity_matrix(simclr2)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r2 += 0.5 * self.get_similarity_matrix(simclr_r2)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                        sim_matrix_r_pre2 = torch.matmul(simclr_r2[:aug1_mem.shape[0]],pre_u2.t())

                    loss_sim_r2 = self.Supervised_NT_xent_uni(sim_matrix_r2,labels=rot_sim_labels_r2,temperature=0.07)
                    loss_sim_pre2 = self.Supervised_NT_xent_pre(sim_matrix_r_pre2, labels=rot_sim_labels_r2, temperature=0.07)
                    loss_sim2 = self.Supervised_NT_xent_n(sim_matrix2, labels=rot_sim_labels2, temperature=0.07)

                    lo12 =1* loss_sim_r2+1*loss_sim2+loss_sim_pre2#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                    y_label2 = self.model2.logits(self.transform_train(mem_x))

                    y_label_pre2 = self.previous_model2(self.transform_train(mem_x))
                    loss2 = 1 * F.cross_entropy(y_label2, mem_y) + lo12  + \
                        1 * F.mse_loss(y_label_pre2[:, self.old_classes.long()], y_label2[:, self.old_classes.long()])
                    
                    # Distillation loss
                    # Combined batch
                    combined_x = torch.cat([batch_x.to(device), mem_x.to(device)], dim=0)
                    combined_y = torch.cat([batch_y.to(device), mem_y.to(device)], dim=0)
                    
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
                    loss_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                # Loss
                scaler.scale(loss_sum).backward()
                scaler.step(self.optim1)
                scaler.update()

                scaler2.scale(loss2_sum).backward()
                scaler2.step(self.optim2)
                scaler2.update()

                self.iter += 1
                
                self.loss = loss_sum.item()
                print(f"Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {j}", end="\r")

                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                    )

        self.previous_model = deepcopy(self.model1)
        self.previous_model2 = deepcopy(self.model2)
        self.old_classes = torch.cat([self.old_classes, present]).unique()

    
    def train_blurry(self, dataloader, **kwargs):
        raise NotImplementedError

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")