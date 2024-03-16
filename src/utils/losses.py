from __future__ import print_function
import torch
import math
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random as r
import time
import wandb

from torch.autograd import Function
from torch import nn
from sklearn.cluster import KMeans, DBSCAN
from .alias_multinomial import AliasMethod
from copy import deepcopy

from src.utils.utils import get_device, AG_SawSeriesPT, make_orthogonal

device = get_device()
eps = 1e-7

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    # @profile
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'proxy':
            anchor_feature = features[:, 0]
            contrast_feature = features[:, 1]
            anchor_count = 1
            contrast_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if self.contrast_mode == 'proxy':
            exp_logits = torch.exp(logits)
        else:
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        return loss


class MixCo(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, proj1, proj2, posmask, mix_idx):
        posmask = posmask.to(device)
        # compute logits
        dot_contrast = (proj1 @ proj2.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.ones_like(posmask).to(device)
        for i in range(posmask.size(0) - mix_idx):
            logits_mask[mix_idx + i, mix_idx*2 + i] = 0
        posmask = posmask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (posmask * log_prob).sum(1) / posmask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss


class SemiSupConLoss(nn.Module):
    """Adapted version of Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', stream_size=10, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.stream_size = stream_size

    def forward(self, features, input_weights, labels=None, wposmask=None):
        """Compute loss for model. If both `labels` and `wposmask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [memory_batch_size + stream_size, n_views, ...].
            labels: ground truth of shape [memory_batch_size].
            wposmask: contrastive weighted positive mask of shape [mbs, mbs], wposmask_{i,j}=1 if sample j
                        has the same class as sample i and is taken from memory.
                        wposmask_{i,i}=alpha if taken from the stream.
                        # TODO cite paper section.
                        Memory samples MUST be first in batch for this code to work as expected
                
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and wposmask is not None:
            raise ValueError('Cannot define both `labels` and `wposmask`')
        elif wposmask is None:
            if labels is not None:
                labels = labels.contiguous().view(-1, 1)
                # Complete missing label for strema data. Add arbitrary non-existing labels.
                # This is just a trick to obtain the wposmask matrix
                max_label = labels.max().item() if labels.shape[0] > 0 else 0
                n_missing_labels = batch_size - labels.shape[0]
                stream_labels = torch.tensor([max_label + i + 1 for i in range(n_missing_labels)]).view(-1, 1).to(device)
                labels = torch.cat([labels, stream_labels])
                posmask = torch.eq(labels, labels.T).float().to(device)
                wposmask = torch.eq(labels, labels.T).float().to(device)
            else:
                wposmask = torch.eye(batch_size, dtype=torch.float32).to(device)
                posmask = torch.eye(batch_size, dtype=torch.float32).to(device)

            # Weight mask matrix
            for j, valj in enumerate(input_weights):
                wposmask[j, :] = valj

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        wposmask = wposmask.repeat(anchor_count, contrast_count)
        posmask = posmask.repeat(anchor_count, contrast_count)

        mask_out = torch.ones_like(wposmask).to(device) - torch.eye(wposmask.size(0)).to(device)
        posmask = posmask * mask_out
        wposmask = wposmask * posmask

        # compute log_prob
        exp_logits = torch.exp(logits) * mask_out
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (wposmask * log_prob).sum(1) / posmask.sum(1)
        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        
        return loss


class BoostedLoss(nn.Module):
    """Modified Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Only difference is that we return the coefs for boosting here.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None, **kwargs):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        return loss, torch.exp(1e-1 * loss.detach())


class MetricLoss(nn.Module):
    """
    Loss based on data pairs distances for boosted metric learning
    """
    def __init__(self):
        super().__init__()

    def forward(self, projections, pairs, **kwargs):
        projections = torch.cat(torch.unbind(projections, dim=1), dim=0)
        pos_pairs, neg_pairs = pairs

        pos_delta = projections[pos_pairs[:, 0],:] - projections[pos_pairs[:, 1],:]
        neg_delta = projections[neg_pairs[:, 0],:] - projections[neg_pairs[:, 1],:]

        # print(f"{pos_delta.max()}, {neg_delta.max()}")


        pos_dist = torch.pow(pos_delta, 2).sum(dim=1)
        neg_dist = torch.pow(neg_delta, 2).sum(dim=1)
        
        # Numerical stability
        s_pos = pos_dist.max()
        s_neg = neg_dist.min()
        # print(f"{s_pos}, {s_neg}")

        pos_dist = pos_dist - s_pos
        neg_dist = neg_dist - s_neg
        loss = torch.log(torch.mean(torch.exp(pos_dist))) + torch.log(torch.mean(torch.exp(-neg_dist))) - s_neg + s_pos
        # loss = torch.log(torch.mean(torch.exp(pos_dist))) + torch.log(torch.mean(torch.exp(-neg_dist)))
        # print(loss.item())
        # loss = torch.mean(torch.exp(pos_dist))*torch.mean(torch.exp(-neg_dist))
        return loss


class DetachContrastive(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, mem_size, stream_size, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        mask = mask.repeat(2, 2)

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
                    
        dot_contrast = torch.div(
            features @ features.T,
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()

        dot_contrast_detach = torch.div(
            features.detach() @ features.T,
            self.temperature
        )

        # for numerical stability
        logits_max_detach, _ = torch.max(dot_contrast_detach, dim=1, keepdim=True)
        logits_detach = dot_contrast_detach - logits_max_detach.detach()

        # mask for partially detached logits
        mask_detach = torch.zeros((logits.shape[0], logits.shape[0]))
        mask_detach[:mem_size, mem_size:mem_size+ stream_size] = 1
        mask_detach[:mem_size, -stream_size:] = 1
        mask_detach[mem_size + stream_size:mem_size + stream_size + mem_size, mem_size:mem_size+ stream_size] = 1
        mask_detach[mem_size + stream_size:mem_size + stream_size + mem_size, -stream_size:] = 1
        mask_detach += mask_detach.clone().T
        mask_detach = torch.tensor(mask_detach, dtype=torch.bool)
        mask_out = torch.ones_like(logits).to(device) - torch.eye(mask_detach.size(0)).to(device)

        logits_norm = torch.zeros_like(logits)
        logits_norm[mask_detach] = logits_detach[mask_detach]
        logits_norm[torch.logical_not(mask_detach)] = logits[torch.logical_not(mask_detach)]
        exp_logits_norm = torch.exp(logits_norm) * mask_out

        log_prob = logits - torch.log(exp_logits_norm.sum(1, keepdim=True))

        mask = mask * mask_out

        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -1 * mean_log_prob

        return loss


class MultiAugLoss(nn.Module):
    """Contrastive Learning with many different types of augmentations
    """
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.data_count = {}

    def forward(self, features, labels=None, mask=None, **kwargs):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # Need to know which images are mixed-up together for masking
        mixup_idx = kwargs.get('mixup_idx', None) # Ids of the views mixed up
        mixup_coef = kwargs.get('mixup_coef', None)
        mixup_view_id = kwargs.get('mixup_view_id', None)

        cutmix_idx = kwargs.get('cutmix_idx', None) # Ids of the views mixed up
        cutmix_coef = kwargs.get('cutmix_coef', None)
        cutmix_view_id = kwargs.get('cutmix_view_id', None)

        # Indexes for weighting
        proj_idx = kwargs.get('proj_idx', None)
        for i in proj_idx:
            i = int(i.item())
            if i in self.data_count.keys():
                if self.data_count[i] < 1000:
                    self.data_count[i] += 1
            else:
                self.data_count[i] = 0

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        temperatures = []
        for i in proj_idx:
            temperatures.append(self.temperature / (self.data_count[int(i.item())]*0.01+ 1))
        temperatures = torch.Tensor(temperatures).to(device) 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperatures.repeat(contrast_count))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Update mask values for mixup
        if mixup_idx is not None and mixup_coef is not None:
            for i in range(len(mixup_idx)):
                view_ids = list(range(features.size(1)))
                view_ids.remove(mixup_view_id)
                for view_id in view_ids:
                    mask[batch_size * (features.size(1)-1) + i, view_id * batch_size + i] = mixup_coef[i]
                    mask[batch_size * (features.size(1)-1) + i, view_id * batch_size + mixup_idx[i]] = 1 - mixup_coef[i]
        
        # Update mask values for cutmix
        if cutmix_idx is not None and cutmix_coef is not None:
            for i in range(len(cutmix_idx)):
                view_ids = list(range(features.size(1)))
                view_ids.remove(cutmix_view_id)
                for view_id in view_ids:
                    mask[batch_size * (features.size(1)-1) + i, view_id * batch_size + i] = cutmix_coef[i]
                    mask[batch_size * (features.size(1)-1) + i, view_id * batch_size + cutmix_idx[i]] = 1 - cutmix_coef[i]

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )   
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask > 0).to(torch.int8).sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        return loss


class CaSSLELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:

        device = z1.device

        b = z1.size(0)

        p = F.normalize(torch.cat([p1, p2]), dim=-1)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)

        logits = torch.einsum("if, jf -> ij", p, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask.fill_diagonal_(True)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device)
        logit_mask.fill_diagonal_(True)
        logit_mask[:, b:].fill_diagonal_(True)
        logit_mask[b:, :].fill_diagonal_(True)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward_gen(self, features, labels, **kwargs):
        feat_mem = kwargs.get('feat_mem', None)
        feat_stream = kwargs.get('feat_stream', None)

        nviews = features.shape[1]

        crit_supcon = SupConLoss()

        labels = labels.contiguous().view(-1)
        
        mc = self.get_all_means(feat_dim=features.shape[2]).detach()
        
        uniq = [i for i in range(self.tot_classes)]
        
        to_cat = [(labels==label).unsqueeze(0) for label in uniq]
        mask = torch.cat(to_cat,dim=0)

        mask = mask.repeat(1, nviews).long()
        mask = mask.to(device)
        features_flat = torch.cat(torch.unbind(features, dim=1), dim=0)
        mask_mean = labels.unique().long()

        # Negatives generation
        if self.gen_strat == 'mem':
            labels_mem = labels[:len(feat_mem)]
            l_mem_unique = labels_mem.unique()
            n_im_pc = int(len(labels) / len(mask_mean.unique()))
            # n_im_pc = 10
            all_classes = torch.arange(self.tot_classes)
            for c in all_classes:
                if c not in l_mem_unique:
                    m = mc[c]
                    negs_c = torch.cat(
                        [
                            F.normalize(torch.normal(
                                mean=m,
                                std=torch.full_like(m, np.sqrt(self.var))
                                ), dim=0).unsqueeze(0) for _ in range(nviews * n_im_pc)
                        ], dim=0
                    )
                    feat_mem = torch.cat([feat_mem, negs_c.view(n_im_pc, nviews, -1)], dim=0)
                    labels_mem = torch.cat([labels_mem, (torch.ones(n_im_pc)*c).to(device)])
            # supcon_loss = crit_supcon(feat_mem, labels_mem)
            supcon_loss = crit_supcon(feat_mem)
        else:            
            n_im_pc = int(len(labels) / len(mask_mean.unique()))
            # n_im_pc = 10
            all_classes = torch.arange(self.tot_classes)
            for c in all_classes:
                if c in mask_mean:
                    if self.gen_strat == 'pos' or self.gen_strat == 'all':
                        m = mc[c]
                        negs_c = torch.cat(
                            [
                                F.normalize(torch.normal(
                                    mean=m,
                                    std=torch.full_like(m, np.sqrt(self.var))
                                    ), dim=0).unsqueeze(0) for _ in range(nviews * n_im_pc)
                            ], dim=0
                        )
                        features = torch.cat([features, negs_c.view(n_im_pc, nviews, -1)], dim=0)
                        labels = torch.cat([labels, (torch.ones(n_im_pc)*c).to(device)])
                else:
                    m = mc[c]
                    negs_c = torch.cat(
                        [
                            F.normalize(torch.normal(
                                mean=m,
                                std=torch.full_like(m, np.sqrt(self.var))
                                ), dim=0).unsqueeze(0) for _ in range(nviews * n_im_pc)
                        ], dim=0
                    )
                    features = torch.cat([features, negs_c.view(n_im_pc, nviews, -1)], dim=0)
                    labels = torch.cat([labels, (torch.ones(n_im_pc)*c).to(device)])

            supcon_loss = crit_supcon(features, labels)

        zi_mc = (features_flat.unsqueeze(0) - mc.unsqueeze(1)) # z_i - mc
        norm_zi_mc = - (zi_mc ** 2).sum(dim=2) / (2*self.var)    # -||z_i - m_c||^2 / 2sigma

        wmask = torch.ones_like(mask).float()
        
        wmask[mask_mean, :len(feat_mem)] *= 1
        wmask[mask_mean, len(feat_mem):] *= 1
        wmask = wmask / wmask.sum(dim=0, keepdim=True)
        
        logits_exps = torch.exp(norm_zi_mc) * wmask
        norms_exp = logits_exps.sum(0)

        # Compute final loss
        loss_gml = - ((norm_zi_mc - torch.log(norms_exp))  * mask)[mask_mean, :]
        loss_gml = (loss_gml / mask[mask_mean, :].sum(1, keepdim=True)).sum(1).mean()

        # loss = ((self.C * loss_gml + (1 - self.C) * supcon_loss.mean())) * 2
        loss = loss_gml + self.C * supcon_loss.mean()

        return loss
        # return supcon_loss

    def forward_double(self, features, feat_mem, feat_stream, labels, **kwargs):
        labels = labels.contiguous().view(-1)
        labels_mem = labels[:len(feat_mem)]
        labels_stream = labels[len(feat_mem):]
        nviews = features.shape[1]

        mc_fixed = self.get_all_means(feat_dim=features.shape[2])

        uniq = [i for i in range(self.tot_classes)]
        to_cat = [(labels==label).unsqueeze(0) for label in uniq]

        # masks
        mask = torch.cat(to_cat,dim=0)

        mask_mem = torch.ones_like(mask)
        mask_mem[:, len(feat_mem):] = 0
        mask_mem = (mask * mask_mem).to(device)

        mask_stream = torch.ones_like(mask)
        mask_stream[:, :len(feat_mem)] = 0
        mask_stream = (mask * mask_stream).to(device)

        mask = mask.repeat(1, nviews).long()
        mask = mask.to(device)
        mask_mem = mask_mem.repeat(1, nviews).long()
        mask_mem = mask_mem.to(device)
        # mask_stream = mask_stream.repeat(1, nviews).long()
        # mask_stream = mask_stream.to(device)
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        mc = mc_fixed

        mask_mean_all = labels.unique().long()
        mask_mean_mem = labels_mem.unique().long()
        # mask_mean_stream = labels_stream.unique().long()

        zi_mc = (features.unsqueeze(0) - mc.unsqueeze(1)) # z_i - mc
        norm_zi_mc = - (zi_mc ** 2).sum(dim=2) / (2*self.var)    # -||z_i - m_c||^2 / 2sigma
        norms_exp = torch.exp(norm_zi_mc).sum(0)

        loss_all = - ((norm_zi_mc - torch.log(norms_exp))  * mask)[mask_mean_all, :]
        loss_all = (loss_all / mask[mask_mean_all, :].sum(1, keepdim=True)).sum(1).mean()

        loss_mem = (torch.log((1 - (torch.exp(norm_zi_mc) / norms_exp))) * mask_mem)[mask_mean_mem, :]
        loss_mem = (loss_mem / mask[mask_mean_mem, :].sum(1, keepdim=True)).sum(1).mean()

        loss = loss_all + self.C * loss_mem
        return loss.sum()


class LocalConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log((mask * exp_logits).sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).mean()

        # loss
        loss = -1 * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size)

        return loss


class GMLLoss(nn.Module):
    def __init__(self, var=1, **kwargs):
        super().__init__()
        self.var = var
        self.C = kwargs.get('C', 0)
        self.dim = kwargs.get('proj_dim', 128)
        self.tot_classes = kwargs.get('tot_classes', 0)
        self.mu = kwargs.get('mu', 1)
        self.loss_type = kwargs.get('gml_loss', 'map')
        self.init_class_seen()
        self.init_means()

    def init_class_seen(self):
        self.class_seen = torch.LongTensor(size=(0,)).to(device)
    
    def init_means(self):
        if self.tot_classes > 0:
            self.means = torch.eye(self.tot_classes, self.dim).to(device) * self.mu
        else:
            self.means = None

    def update_class_seen(self, labels):
        new_classes = labels.unique()
        self.class_seen = torch.cat([self.class_seen, new_classes]).unique()

    def update_means(self, labels):
        curr = deepcopy(self.class_seen)
        self.update_class_seen(labels)
        if len(self.class_seen) > len(curr):
            self.means = torch.eye(max(self.class_seen) + 1, self.dim).to(device) 
                
    def get_means(self):
        return self.means
    
    # @profile
    def forward(self, features, labels=None, **kwargs):
        feat_mem = kwargs.get('feat_mem', None)
        supervised = kwargs.get('supervised', True)
        all_means = kwargs.get('all_means', False)
        if self.tot_classes == 0:
            all_means = False
        fixed_means = kwargs.get('fixed_means', False)
        nviews = features.shape[1]
        
        if labels is None:
            supervised = False
            labels = torch.Tensor(np.arange(features.size(0)))
        labels = labels.contiguous().view(-1).short()

        # fixing the means manually
        if fixed_means and supervised and not all_means:
            self.update_means(labels)
        
        if all_means:
            uniq = [i for i in range(self.tot_classes)]
        else:            
            if fixed_means and len(self.class_seen) > 0 and supervised:
                uniq = [i for i in range(int(max(self.class_seen) + 1))]
            else:
                uniq = [i for i in range(int(labels.max().item()) + 1)]

        mask = torch.eq(torch.tensor(uniq).unsqueeze(1).to(device),labels.unsqueeze(0))
        mask = mask.repeat(1, nviews).float().to(device)

        features_flat = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if not fixed_means:
            mc_moving = (features_flat.unsqueeze(0) * mask.unsqueeze(2)).mean(1)
            # mc_moving = F.normalize(mc_moving, dim=1)
            mc = mc_moving
        else:
            mc = self.get_means().to(device)
        
        mask_mean = labels.unique().long()

        zi_mc = (features_flat.unsqueeze(0) - mc.unsqueeze(1)) # z_i - mc
        if self.loss_type == 'L1':
            norm_zi_mc = - torch.abs(zi_mc).sum(dim=2) / (self.var)    # -||z_i - m_c|| / sigma Laplace distribution
        else:
            norm_zi_mc = - (zi_mc ** 2).sum(dim=2) / (2*self.var)    # -||z_i - m_c||^2 / 2sigma
        logits_exps = torch.exp(norm_zi_mc).to(device)
        
        if all_means:
            norms_exp = logits_exps.sum(0)
        else:
            norms_exp = logits_exps[mask_mean, :].sum(0)

        # Compute final loss
        if self.loss_type == 'likelihood':
            loss = - ((norm_zi_mc) * mask)[mask_mean, :]
            return loss.sum()
        else:
            loss = - ((norm_zi_mc - torch.log(norms_exp + 1e-8)) * mask)[mask_mean, :]
        
        if self.loss_type == 'LIc':
            loss = (loss / mask[mask_mean, :].sum(1, keepdim=True)).sum(1).mean()
        elif self.loss_type == 'Ic':
            loss = (loss / mask[mask_mean, :].sum(1, keepdim=True)).sum(1).sum()
        elif self.loss_type == 'L':
            loss = loss.sum(1).mean()
        elif self.loss_type == 'map' or self.loss_type == 'L1':
            loss = loss.sum()
        else:
            raise Warning("Unknown loss")

        return loss


class CO2LLoss(nn.Module):
    """Code from https://github.com/chaht01/Co2L/blob/main/losses_negative_only.py
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss


class AGDLoss(nn.Module):
    def __init__(self, var=1, **kwargs):
        super().__init__()
        self.var = var
        self.dim = kwargs.get('proj_dim', 128)
        self.mu = kwargs.get('mu', 1)
        self.init_class_seen()

    def init_class_seen(self):
        self.class_seen = torch.LongTensor(size=(0,)).to(device)
    
    def update_class_seen(self, labels):
        new_classes = labels.unique()
        self.class_seen = torch.cat([self.class_seen, new_classes]).unique()

    # @profile
    def forward(self, features, labels=None, **kwargs):
        nviews = features.shape[1]
        labels = labels.contiguous().view(-1).short()

        self.update_class_seen(labels)
        
        uniq = [i for i in range(int(max(self.class_seen) + 1))]

        mask = torch.eq(torch.tensor(uniq).unsqueeze(1).to(device),labels.unsqueeze(0))
        mask = mask.repeat(1, nviews).long().to(device)

        features_expand = (torch.cat(torch.unbind(features, dim=1), dim=0)).expand(mask.shape[0], mask.shape[1], features.shape[-1])
        maskoh = F.one_hot(torch.ones_like(mask) * torch.arange(0, mask.shape[0]).to(device).view(-1, 1), features_expand.shape[-1])
        features_p = (features_expand * maskoh).sum(-1)

        densities = AG_SawSeriesPT(
            y=features_p.double(),
            sigma2=torch.tensor([self.var], dtype=torch.float64).to(device),
            d=torch.tensor([self.dim], dtype=torch.float64).to(device),
            N=torch.arange(0,40)
            ).to(device)

        mask_mean = labels.unique().long()
        norms_densities = densities[mask_mean, :].sum(0, keepdim=True)

        # Compute final loss
        loss = - (torch.log(densities / norms_densities) * mask)[mask_mean, :].sum()
        
        return loss


class MCDKDLoss(nn.Module):
    """
    Implementatin of Multi-Class Decoupled Knowledge Distillation.
    Code adapted from DKD implementation: 
    """
    
    def __init__(self, alpha=1, beta=1, gamma=1, temperature=4, kappa=5, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.kappa = kappa
        self.beta_scheduler = kwargs.get('beta_scheduler', False)
        self.use_wandb = kwargs.get('use_wandb', False)

    def forward(self, logits_teacher, logits_student, old_classes, new_classes, **kwargs):
        tri_group = kwargs.get('tri_group', False)
        focal_dist = kwargs.get('focal_dist', False)
        self.task_id = kwargs.get('task_id', None)
        if tri_group:
            return self.tri_forward(logits_teacher, logits_student, old_classes, new_classes, **kwargs)
        if focal_dist:
            return self.focal_forward(logits_teacher, logits_student, old_classes, new_classes, **kwargs)
            
        seen_classes = old_classes + new_classes
        unseen_classes = [i for i in range(logits_teacher.size(1)) if i not in seen_classes]
        
        logits_teacher[:, unseen_classes] -= (1000*self.temperature)
        logits_student[:, unseen_classes] -= (1000*self.temperature)
        
        # Binary KL part
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        
        pred_teacher = self.cat_mask(pred_teacher, groups=[old_classes, new_classes])
        pred_student = self.cat_mask(pred_student, groups=[old_classes, new_classes])
        
        log_pred_student = torch.log(pred_student)
        
        binary_loss = F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1).mean(0)* (self.temperature**2)
        
        # Old classes KL
        pred_teacher_o = F.softmax(
            logits_teacher[:, old_classes] / self.temperature, dim=1
        )
        log_pred_student_o = F.log_softmax(
            logits_student[:, old_classes] / self.temperature, dim=1
        )
        old_kd_loss = F.kl_div(log_pred_student_o, pred_teacher_o, reduction='none').sum(1).mean(0) * (self.temperature**2)
        
        # New classes KL
        pred_teacher_n = F.softmax(
            logits_teacher[:, new_classes] / self.temperature, dim=1
        )
        log_pred_student_n = F.log_softmax(
            logits_student[:, new_classes] / self.temperature, dim=1
        )
        new_kd_loss = F.kl_div(log_pred_student_n, pred_teacher_n, reduction='none').sum(1).mean(0) * (self.temperature**2)
        
        if self.use_wandb:
            wandb.log({
                "kd_binary": binary_loss.item(),
                "kd_old": old_kd_loss.item(),
                "kd_new": new_kd_loss.item(),
                "task_id": self.task_id
            })
        if self.beta_scheduler:
            self.update_params(old_classes, new_classes)
        
        return self.alpha * binary_loss + self.beta * old_kd_loss + self.gamma * new_kd_loss

    def tri_forward(self, logits_teacher, logits_student, old_classes, new_classes, **kwargs):
        seen_classes = old_classes + new_classes
        unseen_classes = [i for i in range(logits_teacher.size(1)) if i not in seen_classes]
        
        loss = 0
        
        # Binary KL part
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        
        groups = [old_classes, new_classes, unseen_classes]
        
        pred_teacher = self.cat_mask(pred_teacher, groups=groups)
        pred_student = self.cat_mask(pred_student, groups=groups)
        
        log_pred_student = torch.log(pred_student)
        
        ternary_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1).mean(0)
            * (self.temperature**2)
        )

        loss += self.alpha * ternary_loss
        
        for i, g in enumerate(groups):
            pred_teacher_g = F.softmax(
                logits_teacher[:, g] / self.temperature, dim=1
            )
            log_pred_student_g = F.log_softmax(
                logits_student[:, g] / self.temperature, dim=1
            )
            kd_loss_g = (
                F.kl_div(log_pred_student_g, pred_teacher_g, reduction='none').sum(1).mean(0)
                * (self.temperature**2)
            )
            if i == 0:
                loss += self.beta * kd_loss_g
                if self.use_wandb:
                    wandb.log({
                        "kd_g0": kd_loss_g.item(),
                        "task_id": self.task_id
                    })
            elif i == 2:
                loss += self.gamma * kd_loss_g
                if self.use_wandb:
                    wandb.log({
                        "kd_g2": kd_loss_g.item(),
                        "task_id": self.task_id
                    })
        if self.use_wandb:
            wandb.log({
                "kd_binary": ternary_loss.item(),
                "task_id": self.task_id
            })
        if self.beta_scheduler:
            self.update_params(old_classes, new_classes)
        
        return loss

    def focal_forward(self, logits_teacher, logits_student, old_classes, new_classes, **kwargs):
        # seen_classes = old_classes + new_classes
        # unseen_classes = [i for i in range(logits_teacher.size(1)) if i not in seen_classes]
        
        # logits_teacher[:, unseen_classes] -= (1000*self.temperature)
        # logits_student[:, unseen_classes] -= (1000*self.temperature)
        
        # Binary KL part
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        
        pred_teacher = self.cat_mask(pred_teacher, groups=[old_classes, new_classes])
        pred_student = self.cat_mask(pred_student, groups=[old_classes, new_classes])
        
        log_pred_student = torch.log(pred_student)
        
        binary_loss = F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1).mean(0) * (self.temperature ** 2)
        
        # p_old computation
        p_old = (pred_teacher[:,0])**self.kappa
        
        # Old classes KL
        pred_teacher_o = F.softmax(logits_teacher[:, old_classes] / self.temperature, dim=1)
        log_pred_student_o = F.log_softmax(logits_student[:, old_classes] / self.temperature, dim=1)
        
        old_kd_loss = F.kl_div(log_pred_student_o, pred_teacher_o, reduction='none').sum(1) * (self.temperature ** 2)
        old_kd_loss = (p_old * old_kd_loss).mean(0)
        
        # New classes KL
        pred_teacher_n = F.softmax(logits_teacher[:, new_classes] / self.temperature, dim=1)
        log_pred_student_n = F.log_softmax(logits_student[:, new_classes] / self.temperature, dim=1)
        new_kd_loss = F.kl_div(log_pred_student_n, pred_teacher_n, reduction='none').sum(1) * (self.temperature ** 2)
        
        p_new = pred_teacher[:,1]
        new_kd_loss = (p_new * new_kd_loss).mean(0)
        
        if self.use_wandb:
            wandb.log({
                "kd_binary": binary_loss.item(),
                "kd_old": old_kd_loss.mean(0).item(),
                "kd_new": new_kd_loss.item(),
                "p_old_mean": p_old.mean(0).item(),
                "p_old_min": p_old.min(0).values.item(),
                "p_old_max": p_old.max(0).values.item(),
                "p_old_std": p_old.std(0).item(),
                "task_id": self.task_id
            })
        if self.beta_scheduler:
            self.update_params(old_classes, new_classes)
        
        return self.alpha * binary_loss + old_kd_loss
    
    def cat_mask(self, t, groups):
        to_cat = []
        for g in groups:
            to_cat.append(t[:, g].sum(dim=1, keepdims=True))
        rt = torch.cat(to_cat, dim=1)
        return rt

    def update_params(self, old_classes, new_classes):
        self.alpha = 1 - len(new_classes) / (len(new_classes) + len(old_classes))
        self.beta =  len(new_classes) / (len(new_classes) + len(old_classes))
        self.gamma = 0


class SepLoss(nn.Module):
    def __init__(
        self,
        temperature=4,
        alpha_ce=1,
        alpha_kd=1,
        beta_ce=1,
        beta_kd=1,
        gamma_ce=1,
        gamma_kd=1,
        kappa=1,
        **kwargs
        ):
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.beta_ce = beta_ce
        self.beta_kd = beta_kd
        self.gamma_ce = gamma_ce
        self.gamma_kd = gamma_kd
        self.kappa = kappa
        self.vanilla_beta = kwargs.get('vanilla_beta', False)
        self.beta_scheduler = kwargs.get('beta_scheduler', False)
        self.use_wandb = kwargs.get('use_wandb', False)
        self.inter_focal_ce = kwargs.get("inter_focal_ce", False)
        self.intra_focal_ce = kwargs.get("intra_focal_ce", False)
        self.inter_focal_kd = kwargs.get("inter_focal_kd", False)
        self.intra_focal_kd = kwargs.get("intra_focal_kd", False)
        self.kd_new = kwargs.get("kd_new", True)
        self.beta_min_value = kwargs.get('beta_min_value', "adaptative")

    def forward(self, logits_teacher, logits_student, gt, groups, **kwargs):
        self.beta_scheduler = kwargs.get('beta_scheduler', False)
        self.task_id = kwargs.get('task_id', 0)
        seen_classes = list()
        for g in groups:
            seen_classes += g
        unseen_classes = [i for i in range(logits_teacher.size(1)) if i not in seen_classes]
        
        groups_ce = groups
        groups_kd = groups if self.kd_new else groups[:-1]
        # groups_kd = groups

        if len(unseen_classes) > 0:
            groups_ce = groups_ce + [unseen_classes]
            groups_kd = groups_kd + [unseen_classes]

        # logits_teacher[:, unseen_classes] -= (1000*self.temperature)
        # logits_student[:, unseen_classes] -= (1000*self.temperature)        

        pred_true = F.one_hot(gt.long(), num_classes=logits_teacher.size(1)).squeeze(1)

        loss_ce = self.sep_ce(pred_true, logits_student, groups=groups_ce)
        loss_kd = self.sep_kl(logits_teacher, logits_student, groups=groups_kd, gt=pred_true)

        return loss_ce, loss_kd
    
    def sep_ce(self, logits1, logits2, groups):
        if len(groups) > 1:
            p2 = F.softmax(logits2, dim=1)
            
            p1_group = self.cat_mask(logits1, groups=groups)
            p2_group = self.cat_mask(p2, groups=groups)
            
            if self.inter_focal_ce:
                group_loss = (- ((1 - p2_group) ** self.kappa) * p1_group * torch.log(p2_group + 1e-8)).sum(1).mean() * self.alpha_ce
            else:
                # group_loss = (- p1_group * torch.log(p2_group + 1e-8)).sum(1).mean() * self.alpha_ce
                group_loss = (- p1_group * torch.log(p2_group)).sum(1).mean() * self.alpha_ce
        else:
            group_loss = torch.Tensor([0]).cuda()

        if self.use_wandb:
            wandb.log({
                "group_loss_ce": group_loss.item(),
                "task_id": self.task_id
            })
            
        for i, g in enumerate(groups):
            p1_g = logits1[:, g]
            p2_g = F.softmax(logits2[:, g], dim=1)
            # log_p2_g = torch.log(p2_g + 1e-8)
            log_p2_g = torch.log(p2_g)
            
            if self.intra_focal_ce:
                loss_g =  (- p1_g * ((1 - p2_g) ** self.kappa) * log_p2_g).sum(1)
            else:
                loss_g =  (- p1_g * log_p2_g).sum(1)
            
            # if self.inter_focal_ce:
                # loss_g = p1_group[:,i] * ((1 - p2_group[:,i]) ** self.kappa) * loss_g
                # loss_g = p1_group[:,i] * loss_g
            # else:
            
            loss_g = p1_group[:, i] * loss_g
            
            # if i >= (len(groups) - 2):
                # loss_g = p1_group[:, i] * loss_g * 5
            # else:
                # loss_g = p1_group[:, i] * loss_g
            
            # Weird focal
            # loss_g = ((2 - p2_group[:,i]) ** 2) * p1_group[:,i] * loss_g
            
            # if self.beta_scheduler:
                # loss_g =  (((i+1)/(len(groups)+1)) * loss_g)
            if not i:
                group_loss += loss_g.mean() * self.beta_ce
            else:
                group_loss += loss_g.mean() * self.gamma_ce
            
            if self.use_wandb:
                wandb.log({
                f"loss_g{i}": loss_g.mean().item(),
                "task_id": self.task_id
            })
        
        return group_loss

    def sep_kl(self, logits1, logits2, groups, **kwargs):
        gt = kwargs.get('gt', None)
        if gt is not None and len(groups) > 1:
            gt_logits = self.cat_mask(gt, groups=groups)
        p1 = F.softmax(logits1 / self.temperature, dim=1)
        p2 = F.softmax(logits2 / self.temperature, dim=1)
        
        if len(groups) > 1:
            p1_ngroups = self.cat_mask(p1, groups=groups)
            p2_ngroups = self.cat_mask(p2, groups=groups)

            log_p2_ngroups = torch.log(p2_ngroups)
            if self.inter_focal_kd:
                group_loss = (
                    (((p1_ngroups - p2_ngroups) ** self.kappa) * F.kl_div(log_p2_ngroups, p1_ngroups, reduction='none')).sum(1).mean(0)
                    * (self.temperature**2)
                    * self.alpha_kd
                )
            else:
                group_loss = (
                    F.kl_div(log_p2_ngroups, p1_ngroups, reduction='none').sum(1).mean(0)
                    * (self.temperature**2)
                    * self.alpha_kd
                    )
        else:
            group_loss = torch.Tensor([0]).cuda()
        
        if self.use_wandb:
            wandb.log({
                "group_loss_kd": group_loss.item(),
                "task_id": self.task_id
            })
            
        for i, g in enumerate(groups):
            p1_g = F.softmax(
                logits1[:, g] / self.temperature, dim=1
            )
            p2_g = F.softmax(
                logits2[:, g] / self.temperature, dim=1
            )
            log_p2_g = torch.log(p2_g)
            
            if self.intra_focal_kd:
                loss_g = (((p1_g - p2_g) ** self.kappa) * F.kl_div(log_p2_g, p1_g, reduction='none') * (self.temperature**2)).sum(1)
            else:
                loss_g = (F.kl_div(log_p2_g, p1_g, reduction='none') * (self.temperature**2)).sum(1)
                
            if self.beta_scheduler:
                loss_g =  ((1 - (i+1)/(len(groups)+1)) * loss_g) * len(groups)
            elif  self.beta_min_value == 'adaptative':
                loss_g = (1/(len(groups)+1) + p1_ngroups[:, i]) * loss_g * len(groups)
            elif  self.beta_min_value == 'reverse':
                loss_g = p1_ngroups[:, -(i+1)] * loss_g * len(groups)
            elif self.beta_min_value == "constant":
                loss_g = loss_g
            elif self.beta_min_value == "vanilla":
                loss_g =  p1_ngroups[:, i] * loss_g
            elif self.beta_min_value == "adaptative_no_min":
                loss_g =  p1_ngroups[:, i] * loss_g * len(groups)
                    
                    # if i < (len(groups) - 2):
                        # loss_g = (1/len(groups) + p1_ngroups[:, i]) * loss_g * len(groups)
                    # else:
                        # loss_g = (5/len(groups) + p1_ngroups[:, i]) * loss_g * len(groups)
                        
                    # loss_g = loss_g
                
            if not i:
                group_loss += loss_g.mean() * self.beta_kd
            else:
                group_loss += loss_g.mean() * self.gamma_kd
            
            if self.use_wandb:
                wandb.log({
                    f"p_{i}": p1_ngroups[:, i].mean().item() if len(groups) > 1 else 1,
                    f"kd_g{i}": loss_g.mean().item(),
                    "task_id": self.task_id
                })
        
        return group_loss
        
    def cat_mask(self, t, groups):
        to_cat = []
        for g in groups:
            to_cat.append(t[:, g].sum(dim=1, keepdims=True))
        rt = torch.cat(to_cat, dim=1)
        return rt

    def update_params(self, old_classes, new_classes):
        self.alpha = 1 - len(new_classes) / (len(new_classes) + len(old_classes))
        self.beta =  len(new_classes) / (len(new_classes) + len(old_classes))
        self.gamma = 0


class OrthoLoss(nn.Module):
    def __init__(self, var=1, **kwargs):
        super().__init__()
        self.var = var
        self.dim = kwargs.get('proj_dim', 128)
        self.mu = kwargs.get('mu', 1)
        self.init_class_seen()

    def init_class_seen(self):
        self.class_seen = torch.LongTensor(size=(0,)).to(device)
    
    def update_class_seen(self, labels):
        new_classes = labels.unique()
        self.class_seen = torch.cat([self.class_seen, new_classes]).unique()

    def get_means(self):
        return self.means
    
    # @profile
    def forward(self, features, labels=None, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):        
            nviews = features.shape[1]
            labels = labels.contiguous().view(-1).short()

            self.update_class_seen(labels)
            
            uniq = [i for i in range(int(max(self.class_seen) + 1))]

            mask = torch.eq(torch.tensor(uniq).unsqueeze(1).to(device),labels.unsqueeze(0))
            mask = mask.repeat(1, nviews).long().to(device)
            
            features_unrolled = (torch.cat(torch.unbind(features, dim=1), dim=0))
            
            # orthogonalisation
            features_ortho = make_orthogonal(features_unrolled, torch.cat([labels.long() for _ in range(nviews)]))
            features_expand = (features_ortho).expand(mask.shape[0], mask.shape[1], features.shape[-1])
            
            maskoh = F.one_hot(torch.ones_like(mask) * torch.arange(0, mask.shape[0]).to(device).view(-1, 1), features_expand.shape[-1])
            features_p = (features_ortho * maskoh).sum(-1)

            densities = AG_SawSeriesPT(
                y=features_p.double(),
                sigma2=torch.tensor([self.var], dtype=torch.float64).to(device),
                d=torch.tensor([self.dim], dtype=torch.float64).to(device),
                N=torch.arange(0,40)
                ).to(device)

            mask_mean = labels.unique().long()
            norms_densities = densities[mask_mean, :].sum(0, keepdim=True)

            # Compute final loss
            loss = - (torch.log(densities / norms_densities) * mask)[mask_mean, :].sum()

            return loss
        

class BYOLLoss(nn.Module):
    """
    Implements BYOL (https://arxiv.org/abs/2006.07733)
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, pred1, ema_pred1, pred2, ema_pred2):
        pred1 = F.normalize(pred1, dim=1)
        ema_pred1 = F.normalize(ema_pred1, dim=1)
        pred2 = F.normalize(pred2, dim=1)
        ema_pred2 = F.normalize(ema_pred2, dim=1)
        mse_loss = (self.mse_loss(pred1, ema_pred2) + self.mse_loss(pred2, ema_pred1)) / 2
        return mse_loss
