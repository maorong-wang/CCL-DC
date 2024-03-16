"""Code adapted from https://github.com/gydpku/GSA
Fair warning : the original code is one of the worst I've seen.
Sensitive developpers are advised to not click on the above link.
"""
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

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from torch.distributions import Categorical

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import GSAResnet, ImageNet_GSAResnet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device

device = get_device()
scaler = amp.GradScaler()
scaler2 = amp.GradScaler()

class GSACCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.negative_logits_SUM1 = None
        self.positive_logits_SUM1 = None
        self.negative_logits_SUM2 = None
        self.positive_logits_SUM2 = None
        self.classes_per_task = self.params.n_classes // self.params.n_tasks
        # I know the variable naming is terrible. Please dont judge me it all comes from the authors terrible code
        # One day I will make it look better but its the best I can do rn
        self.Category_sum2 = None
        self.Category_sum2 = None
        self.class_holder = []
        self.tf_gsa = nn.Sequential(
                        RandomGrayscale(p=0.2),
                        RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.))
                    ).to(device)
        self.flip_num=2

        self.previous_model = None
        self.kd_lambda = self.params.kd_lambda
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        self.iter = 0

    def RandomFlip(self, x, num=0):
        if not num:
            num=self.flip_num
        x=self.tf_gsa(x)
        X = []

        X.append(x)
        X.append(self.flip_inner(x, 1, 1))

        X.append(self.flip_inner(x, 0, 1))

        X.append(self.flip_inner(x, 1, 0))

        return torch.cat([X[i] for i in range(num)], dim=0)
    
    def flip_inner(self, x, flip1, flip2):
        num = x.shape[0]
        if x.shape[-1] == 32:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 16, 32)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
            if flip2:
                s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 32, 32)
        elif x.shape[-1] == 64:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 32, 64)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))
            if flip2:
                s2 = torch.flip(s2, (3,))
            
            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 64, 64)
        elif x.shape[-1] == 224:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 112, 224)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))
            if flip2:
                s2 = torch.flip(s2, (3,))
            
            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 224, 224)
        return S
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = GSAResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            )
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            model = ImageNet_GSAResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            )
        return model.to(device)

    def load_criterion(self):
        return F.cross_entropy
    
    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        dataloaders = kwargs.get('dataloaders', None)
        new_class_holder = []
        
        if task_id > 0:
            self.Category_sum1 = torch.cat((self.Category_sum1, torch.zeros(self.classes_per_task)))
            self.negative_logits_SUM1 = torch.cat((self.negative_logits_SUM1, torch.zeros(self.classes_per_task).to(device)))
            self.positive_logits_SUM1 = torch.cat((self.positive_logits_SUM1, torch.zeros(self.classes_per_task).to(device)))
            self.Category_sum2 = torch.cat((self.Category_sum2, torch.zeros(self.classes_per_task)))
            self.negative_logits_SUM2 = torch.cat((self.negative_logits_SUM2, torch.zeros(self.classes_per_task).to(device)))
            self.positive_logits_SUM2 = torch.cat((self.positive_logits_SUM2, torch.zeros(self.classes_per_task).to(device)))
            
        negative_logits_sum1=None
        positive_logits_sum1=None
        sum_num1=0
        negative_logits_sum2=None
        positive_logits_sum2=None
        sum_num2=0
        category_sum1 = None  
        category_sum2 = None  
         
        self.model1.train()
        self.model2.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            self.optim1.zero_grad()
            self.optim2.zero_grad()
            
            x, y = batch[0].to(device), batch[1].to(device)

            # re-order to adapt GSA code more easily
            y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y]).to(device)
            self.stream_idx += len(x)
            
            if not self.buffer.is_empty():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    Y = deepcopy(y)
                    for c in range(len(Y)):
                        if Y[c] not in self.class_holder:
                            self.class_holder.append(Y[c].detach())
                            new_class_holder.append(Y[c].detach())
                    
                    ori_x = x.detach()
                    ori_y = y.detach()
                    
                    curr_labels = self.params.labels_order[task_id*self.classes_per_task:(task_id+1)*self.classes_per_task]
                    
                    cur_x, cur_y = self.buffer.only_retrieve(n_imgs=22, desired_labels=curr_labels)
                    cur_x = cur_x.to(device)
                    cur_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in cur_y]).to(device) if len(cur_y) > 0 else cur_y.to(device)
                    
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))
                    
                    # transform
                    # x = self.transform_train(x)
                    # x = self.tf_gsa(x)
                    x = self.RandomFlip(x, num=2)
                    y = y.repeat(2)
                    
                    pred_y1 = self.model1.logits(x)[:, :(task_id+1)*self.classes_per_task]  # Inference 1
                    pred_y2 = self.model2.logits(x)[:, :(task_id+1)*self.classes_per_task] 
                    
                    if task_id>0:
                        pred_y_new1 = pred_y1[:, -self.classes_per_task:]
                        pred_y_new2 = pred_y2[:, -self.classes_per_task:]
                    else:
                        pred_y_new1=pred_y1
                        pred_y_new2=pred_y2

                    y_new = y - self.classes_per_task*task_id
                    rate = len(new_class_holder)/len(self.class_holder)

                    mem_x, mem_y = self.buffer.except_retrieve(int(self.params.mem_batch_size*(1-rate)), undesired_labels=curr_labels)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]) if len(mem_y) > 0 else mem_y

                    
                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    mem_x = torch.cat((mem_x[:int(self.params.mem_batch_size*(1-rate))].to(device),index_x[:int(self.params.mem_batch_size*rate)].to(device)),dim=0)
                    mem_y = torch.cat((mem_y[:int(self.params.mem_batch_size*(1-rate))].to(device),index_y[:int(self.params.mem_batch_size*rate)].to(device)))

                    index = torch.randperm(mem_y.size()[0])
                    mem_x = mem_x[index][:]
                    mem_y = mem_y[index][:]

                    mem_y = mem_y.reshape(-1).long()
                    
                    # mem_x = self.transform_train(mem_x)
                    # mem_x = self.tf_gsa(mem_x)
                    mem_x = self.RandomFlip(mem_x, num=2)
                    mem_y = mem_y.repeat(2)
                    
                    y_pred1 = self.model1.logits(mem_x)[:, :(task_id+1)*self.classes_per_task]  # Inference 2
                    y_pred2 = self.model2.logits(mem_x)[:, :(task_id+1)*self.classes_per_task]  # Inference 2
                    y_pred_new1 = y_pred1
                    y_pred_new2 = y_pred2

                    exp_new1 = torch.exp(y_pred_new1)
                    exp_new1 = exp_new1
                    exp_new_sum1 = torch.sum(exp_new1, dim=1)
                    logits_new1 = (exp_new1 / exp_new_sum1.unsqueeze(1))
                    category_matrix_new1 = torch.zeros(logits_new1.shape)
                    for i_v in range(int(logits_new1.shape[0])):
                        category_matrix_new1[i_v][mem_y[i_v]] = 1
                    positive_prob1 = torch.zeros(logits_new1.shape)
                    false_prob1 = deepcopy(logits_new1.detach())
                    for i_t in range(int(logits_new1.shape[0])):
                        false_prob1[i_t][mem_y[i_t]] = 0
                        positive_prob1[i_t][mem_y[i_t]] = logits_new1[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum1 is None:
                        negative_logits_sum1 = torch.sum(false_prob1, dim=0)
                        positive_logits_sum1 = torch.sum(positive_prob1, dim=0)
                        if task_id == 0:
                            self.Category_sum1 = torch.sum(category_matrix_new1, dim=0)
                        else:
                            self.Category_sum1 += torch.sum(category_matrix_new1, dim=0)

                        category_sum1 = torch.sum(category_matrix_new1, dim=0)
                    else:
                        self.Category_sum1 += torch.sum(category_matrix_new1, dim=0)
                        negative_logits_sum1 += torch.sum(false_prob1, dim=0)
                        positive_logits_sum1 += torch.sum(positive_prob1, dim=0)
                        category_sum1 += torch.sum(category_matrix_new1, dim=0)
                    if self.negative_logits_SUM1 is None:
                        self.negative_logits_SUM1 = torch.sum(false_prob1, dim=0).to(device)
                        self.positive_logits_SUM1 = torch.sum(positive_prob1, dim=0).to(device)
                    else:
                        self.negative_logits_SUM1 += torch.sum(false_prob1, dim=0).to(device)
                        self.positive_logits_SUM1 += torch.sum(positive_prob1, dim=0).to(device)

                    sum_num1 += int(logits_new1.shape[0])
                    
                    if j < 5:
                        ANT = torch.ones(len(self.class_holder))
                    else:
                        ANT = (self.Category_sum1.to(device) - self.positive_logits_SUM1).to(device)/self.negative_logits_SUM1.to(device)

                    ttt = torch.zeros(logits_new1.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT):
                            ttt[qqq][mem_y[qqq]] = 1
                        else:
                            ttt[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT[mem_y[qqq]])))

                    loss_n=-torch.sum(torch.log(logits_new1)*ttt.to(device))/mem_y.shape[0]
                    loss1 =2* loss_n + 1 * F.cross_entropy(pred_y_new1, y_new.long())

                    exp_new2 = torch.exp(y_pred_new2)
                    exp_new2 = exp_new2
                    exp_new_sum2 = torch.sum(exp_new2, dim=1)
                    logits_new2 = (exp_new2 / exp_new_sum2.unsqueeze(1))
                    category_matrix_new2 = torch.zeros(logits_new2.shape)
                    for i_v in range(int(logits_new2.shape[0])):
                        category_matrix_new2[i_v][mem_y[i_v]] = 1
                    positive_prob2 = torch.zeros(logits_new2.shape)
                    false_prob2 = deepcopy(logits_new2.detach())
                    for i_t in range(int(logits_new2.shape[0])):
                        false_prob2[i_t][mem_y[i_t]] = 0
                        positive_prob2[i_t][mem_y[i_t]] = logits_new2[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum2 is None:
                        negative_logits_sum2 = torch.sum(false_prob2, dim=0)
                        positive_logits_sum2 = torch.sum(positive_prob2, dim=0)
                        if task_id == 0:
                            self.Category_sum2 = torch.sum(category_matrix_new2, dim=0)
                        else:
                            self.Category_sum2 += torch.sum(category_matrix_new2, dim=0)

                        category_sum2 = torch.sum(category_matrix_new2, dim=0)
                    else:
                        self.Category_sum2 += torch.sum(category_matrix_new2, dim=0)
                        negative_logits_sum2 += torch.sum(false_prob2, dim=0)
                        positive_logits_sum2 += torch.sum(positive_prob2, dim=0)
                        category_sum2 += torch.sum(category_matrix_new2, dim=0)
                    if self.negative_logits_SUM2 is None:
                        self.negative_logits_SUM2 = torch.sum(false_prob2, dim=0).to(device)
                        self.positive_logits_SUM2 = torch.sum(positive_prob2, dim=0).to(device)
                    else:
                        self.negative_logits_SUM2 += torch.sum(false_prob2, dim=0).to(device)
                        self.positive_logits_SUM2 += torch.sum(positive_prob2, dim=0).to(device)

                    sum_num2 += int(logits_new2.shape[0])
                    
                    if j < 5:
                        ANT2 = torch.ones(len(self.class_holder))
                    else:
                        ANT2 = (self.Category_sum2.to(device) - self.positive_logits_SUM2).to(device)/self.negative_logits_SUM2.to(device)

                    ttt2 = torch.zeros(logits_new2.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT2):
                            ttt2[qqq][mem_y[qqq]] = 1
                        else:
                            ttt2[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT2[mem_y[qqq]])))

                    loss_n2 = -torch.sum(torch.log(logits_new2)*ttt2.to(device))/mem_y.shape[0]
                    loss2 =2* loss_n2  + 1 * F.cross_entropy(pred_y_new2, y_new.long())

                    # distillation stuff
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x = mem_x.to(device)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]).to(device)
                    batch_x = batch[0].to(device)
                    batch_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in batch[1]]).to(device)
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)

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
                    loss1 += 0.5 * loss_ce + self.kd_lambda * loss_dist
                    loss2 += 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

                scaler.scale(loss1).backward()
                scaler.step(self.optim1)
                scaler.update()

                scaler2.scale(loss2).backward()
                scaler2.step(self.optim2)
                scaler2.update()

                self.iter += 1

                self.loss = loss1.item()
                print(f"Loss (Peer1) : {loss1.item():.4f}  Loss (Peer2) : {loss2.item():.4f}  batch {j}", end="\r")

                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}  Loss (Peer1) : {loss1.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   time : {time.time() - self.start:.4f}s"
                    )

            # Update buffer
            self.buffer.update(imgs=batch[0].to(device).detach(), labels=batch[1].to(device).detach())
    
    def train_blurry(self, dataloader, **kwargs):
        raise NotImplemented

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    

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
    

    def get_entropy(self, dataloaders, task_id):
        #special adaption for GSA label orders
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
            # re-order to adapt GSA code more easily
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
            # re-order to adapt GSA code more easily
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