import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

from copy import deepcopy

device = get_device()

class ERCCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.previous_model = None
        self.kd_lambda = self.params.kd_lambda
        self.iter = 0

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            return ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    
    def load_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:

                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    # Augment
                    bs = combined_x.size(0)
                    combined_aug = self.transform_train(combined_x)
                    # Inference

                    feat, logits = self.model1.full(combined_aug)
                    feat2, logits2 = self.model2.full(combined_aug)


                    loss_ce = self.criterion(logits, combined_y.long())
                    loss_ce2 = self.criterion(logits2, combined_y.long())

                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
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
                    loss_ce += self.criterion(logits1, combined_y.long()) + self.criterion(logits1_vanilla, combined_y.long()) + self.criterion(logits1_step1, combined_y.long()) + self.criterion(logits1_step2, combined_y.long())
                    loss_ce2 += self.criterion(logits2, combined_y.long()) + self.criterion(logits2_vanilla, combined_y.long()) + self.criterion(logits2_step1, combined_y.long()) + self.criterion(logits2_step2, combined_y.long())

                    # Distillation Loss
                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # Total Loss
                    loss = 0.5 * loss_ce  + self.kd_lambda * loss_dist 
                    loss2 = 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

                    self.loss = loss.item()
                    self.loss2 = loss2.item()
                    print(f"Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   batch {j}", end="\r")
                    self.optim1.zero_grad()
                    loss.backward()
                    self.optim1.step()

                    self.optim2.zero_grad()
                    loss2.backward()
                    self.optim2.step()

                    self.iter += 1
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   time : {time.time() - self.start:.4f}s"
                )


    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        
    