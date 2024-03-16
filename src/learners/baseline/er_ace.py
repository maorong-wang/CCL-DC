"""Adapted from https://github.com/pclucas14/AML/blob/paper_open_source/methods/er_ace.py
and https://github.com/pclucas14/AML/blob/7c929363d9c687e0aa4539c1ab91c812330d421f/methods/er.py#L10
"""
import torch
import wandb
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.learners.baseline.base import BaseLearner 
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class ER_ACELearner(BaseLearner):
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
        self.model = self.model.train()
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
                aug_xs = self.transform_train(batch_x.to(device))
                logits = self.model.logits(aug_xs)
                mask = torch.zeros_like(logits).to(device)

                # unmask curent classes
                mask[:, present] = 1
                
                # unmask unseen classes
                unseen = torch.arange(len(logits)).to(device)
                for c in self.classes_seen_so_far:
                    unseen = unseen[unseen != c]
                mask[:, unseen] = 1    

                logits_stream = logits.masked_fill(mask == 0, -1e9)   
                loss = self.criterion(logits_stream, batch_y.to(device))

                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Augment
                    aug_xm = self.transform_train(mem_x).to(device)

                    # Inference
                    logits_mem = self.model.logits(aug_xm)
                    loss += self.criterion(logits_mem, mem_y.to(device))

                # Loss
                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.iter += 1
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )



    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)

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
        