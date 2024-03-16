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
import torchvision

from sklearn.metrics import accuracy_score

from src.learners.baseline.base import BaseLearner
from src.buffers.logits_res import LogitsRes
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device

device = get_device()

class DERppLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []
        self.buffer = LogitsRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        self.iter = 0

    def load_model(self):
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
        self.model = self.model.train()
        if self.params.training_type == 'inc':
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
        
    def train_inc(self, dataloader, task_name, **kwargs):
        """Adapted from https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
        """
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        for j, batch in enumerate(dataloader):
            self.model = self.model.train()
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                outputs = self.model.logits(self.transform_train(batch_x.to(device)))
                loss = self.criterion(outputs, batch_y.long().to(device))
                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                if mem_x.size(0) > 0:
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    
                    # Loss
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_beta * self.criterion(mem_outputs, mem_y.long().to(device))

                self.loss = loss.mean().item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                loss.backward()
                self.optim.step()

                self.iter += 1

            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=outputs.detach())

            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )

    def train_blurry(self, dataloader, **kwargs):
        raise NotImplementedError
    