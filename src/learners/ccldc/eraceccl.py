import torch
import wandb
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from copy import deepcopy

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class ER_ACECCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)

        self.previous_model = None
        self.kd_lambda = self.params.kd_lambda

        self.results = []
        self.results_forgetting = []
        self.results_1 = []
        self.results_forgetting_1 = []
        self.results_2 = []
        self.results_forgetting_2 = []
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
        return F.cross_entropy

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'Unknown task name')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()
        present = torch.LongTensor(size=(0,)).to(device)

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1].long()
            self.stream_idx += len(batch_x)

            # update classes seen
            present = batch_y.unique().to(device)
            self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique()
            
            for _ in range(self.params.mem_iters):
                
                # process stream
                aug_xs = self.transform_train(batch_x).to(device)
                batch_x = batch_x.to(device)

                logits = self.model1.logits(aug_xs)
                logits2 = self.model2.logits(aug_xs)
                mask = torch.zeros_like(logits).to(device)

                # unmask curent classes
                mask[:, present] = 1
                
                # unmask unseen classes
                unseen = torch.arange(len(logits)).to(device)
                for c in self.classes_seen_so_far:
                    unseen = unseen[unseen != c]
                mask[:, unseen] = 1    

                logits_stream = logits.masked_fill(mask == 0, -1e9)   
                logits_stream2 = logits2.masked_fill(mask == 0, -1e9)


                loss = self.criterion(logits_stream, batch_y.to(device))
                loss2 = self.criterion(logits_stream2, batch_y.to(device))
                
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Augment
                    aug_xm = self.transform_train(mem_x).to(device)
                    mem_x = mem_x.to(device)

                    # Inference
                    logits_mem = self.model1.logits(aug_xm)
                    logits_mem2 = self.model2.logits(aug_xm)
                    

                    loss += self.criterion(logits_mem, mem_y.to(device)) 
                    loss2 += self.criterion(logits_mem2, mem_y.to(device))

                # distillation
                batch_aug1 = self.transform_1(batch_x)
                batch_aug2 = self.transform_2(batch_aug1)
                batch_aug = self.transform_3(batch_aug2)

                # batch
                logits1 = self.model1.logits(batch_aug)
                logits2 = self.model2.logits(batch_aug)

                logits1_vanilla = self.model1.logits(batch_x)
                logits2_vanilla = self.model2.logits(batch_x)

                logits1_step1 = self.model1.logits(batch_aug1)
                logits2_step1 = self.model2.logits(batch_aug1)

                logits1_step2 = self.model1.logits(batch_aug2)
                logits2_step2 = self.model2.logits(batch_aug2)

                # mask
                logits1_ce = logits1.masked_fill(mask == 0, -1e9)   
                logits2_ce = logits2.masked_fill(mask == 0, -1e9)

                logits1_vanilla_ce = logits1_vanilla.masked_fill(mask == 0, -1e9)
                logits2_vanilla_ce = logits2_vanilla.masked_fill(mask == 0, -1e9)

                logits1_step1_ce = logits1_step1.masked_fill(mask == 0, -1e9)
                logits2_step1_ce = logits2_step1.masked_fill(mask == 0, -1e9)

                logits1_step2_ce = logits1_step2.masked_fill(mask == 0, -1e9)
                logits2_step2_ce = logits2_step2.masked_fill(mask == 0, -1e9)

                # Cls Loss
                loss_ce = self.criterion(logits1_ce, batch_y.to(device)) + self.criterion(logits1_vanilla_ce, batch_y.to(device)) + self.criterion(logits1_step1_ce, batch_y.to(device)) + self.criterion(logits1_step2_ce, batch_y.to(device))
                
                loss_ce2 = self.criterion(logits2_ce, batch_y.to(device)) + self.criterion(logits2_vanilla_ce, batch_y.to(device)) + self.criterion(logits2_step1_ce, batch_y.to(device)) + self.criterion(logits2_step2_ce, batch_y.to(device))

                # Distillation Loss
                loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                
                loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                if mem_x.size(0) > 0:
                    mem_aug1 = self.transform_1(mem_x)
                    mem_aug2 = self.transform_2(mem_aug1)
                    mem_aug = self.transform_3(mem_aug2)
                    
                    # mem
                    logits1_mem = self.model1.logits(mem_aug)
                    logits2_mem = self.model2.logits(mem_aug)

                    logits1_mem_vanilla = self.model1.logits(mem_x)
                    logits2_mem_vanilla = self.model2.logits(mem_x)

                    logits1_mem_step1 = self.model1.logits(mem_aug1)
                    logits2_mem_step1 = self.model2.logits(mem_aug1)

                    logits1_mem_step2 = self.model1.logits(mem_aug2)
                    logits2_mem_step2 = self.model2.logits(mem_aug2)

                    # Cls Loss
                    loss_ce += self.criterion(logits1_mem, mem_y.to(device)) + self.criterion(logits1_mem_vanilla, mem_y.to(device)) + self.criterion(logits1_mem_step1, mem_y.to(device)) + self.criterion(logits1_mem_step2, mem_y.to(device))
                    
                    loss_ce2 += self.criterion(logits2_mem, mem_y.to(device)) + self.criterion(logits2_mem_vanilla, mem_y.to(device)) + self.criterion(logits2_mem_step1, mem_y.to(device)) + self.criterion(logits2_mem_step2, mem_y.to(device))

                    # Distillation Loss
                    loss_dist += self.kl_loss(logits1_mem, logits2_mem.detach()) + self.kl_loss(logits1_mem_vanilla, logits2_mem_step1.detach()) + self.kl_loss(logits1_mem_step1, logits2_mem_step2.detach()) + self.kl_loss(logits1_mem_step2, logits2_mem.detach())
                    
                    loss_dist2 += self.kl_loss(logits2_mem, logits1_mem.detach()) + self.kl_loss(logits2_mem_vanilla, logits1_mem_step1.detach()) + self.kl_loss(logits2_mem_step1, logits1_mem_step2.detach()) + self.kl_loss(logits2_mem_step2, logits1_mem.detach())


                # Total Loss
                loss += 0.5 * loss_ce + self.kd_lambda * loss_dist 
                loss2 += 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

                # Loss
                self.loss = loss.item()
                print(f"Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}  batch {j}", end="\r")
                self.optim1.zero_grad()
                loss.backward()
                self.optim1.step()

                self.loss2 = loss2.item()
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