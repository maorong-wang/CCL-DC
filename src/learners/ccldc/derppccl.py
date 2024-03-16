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
import wandb
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.buffers.logits_res import LogitsRes
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device

device = get_device()

class DERppCCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_1 = []
        self.results_2 = []
        self.results_forgetting = []
        self.results_forgetting_1 = []
        self.results_forgetting_2 = []
        self.buffer = LogitsRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        self.model2 = self.load_model()
        self.optim2 = self.load_optim2()

        self.kd_lambda = self.params.kd_lambda
        self.previous_model = None

        self.iter = 0

    def load_optim2(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model2.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model2.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model2.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                )
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer

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
        
    def kl_loss(self, logits_stu, logits_tea, temperature=3.0):
        """
        Args:
            logits_stu: student logits
            logits_tea: teacher logits
            temperature: temperature
        Returns:
            distillation loss
        """
        pred_teacher = F.softmax(logits_tea / temperature, dim=1)
        log_pred_student = F.log_softmax(logits_stu / temperature, dim=1)
        loss_kd = F.kl_div(
            log_pred_student,
            pred_teacher,
            reduction='none'
        ).sum(1).mean(0) * (temperature ** 2)
        return loss_kd
    
    def train(self, dataloader, **kwargs):
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()
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
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()
        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                batch_x_tr = self.transform_train(batch_x.to(device))
                outputs = self.model1.logits(batch_x_tr)
                outputs2 = self.model2.logits(batch_x_tr)

                loss = self.criterion(outputs, batch_y.long().to(device)) 
                loss2 = self.criterion(outputs2, batch_y.long().to(device))
                mem_x, mem_y, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                mem_x_old, mem_y_old = mem_x, mem_y
                if mem_x.size(0) > 0:
                    batch_s_tr = self.transform_train(mem_x.to(device))
                    mem_outputs = self.model1.logits(batch_s_tr)
                    mem_outputs2 = self.model2.logits(batch_s_tr)

                    # Loss
                    loss += self.params.derpp_alpha * (F.mse_loss(mem_outputs, mem_logits.to(device)))
                    loss2 += self.params.derpp_alpha * (F.mse_loss(mem_outputs2, mem_logits.to(device)))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    batch_s_tr = self.transform_train(mem_x.to(device))
                    mem_outputs = self.model1.logits(batch_s_tr)
                    mem_outputs2 = self.model2.logits(batch_s_tr)

                    loss += self.params.derpp_beta * (self.criterion(mem_outputs, mem_y.long().to(device)))
                    loss2 += self.params.derpp_beta * (self.criterion(mem_outputs2, mem_y.long().to(device)))

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
                loss_ce = self.criterion(logits1, combined_y.long()) + self.criterion(logits1_vanilla, combined_y.long()) + self.criterion(logits1_step1, combined_y.long()) + self.criterion(logits1_step2, combined_y.long())
                loss_ce2 = self.criterion(logits2, combined_y.long()) + self.criterion(logits2_vanilla, combined_y.long()) + self.criterion(logits2_step1, combined_y.long()) + self.criterion(logits2_step2, combined_y.long())

                # Distillation Loss
                loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                # Total Loss
                loss += 0.5 * loss_ce + self.kd_lambda * loss_dist 
                loss2 += 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

                self.loss = loss.mean().item()
                print(f"Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}  batch {j}", end="\r")
                self.optim1.zero_grad()
                loss.backward()
                self.optim1.step()
                self.optim2.zero_grad()
                loss2.backward()
                self.optim2.step()

                self.iter += 1

            # Update buffer
            logits = (outputs + outputs2) / 2.0
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=logits.detach())

            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}  time : {time.time() - self.start:.4f}s"
                )


    def train_blurry(self, dataloader, **kwargs):
        pass
                    

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")

    def save_results(self):
        pass

    def save_results_offline(self):
        pass