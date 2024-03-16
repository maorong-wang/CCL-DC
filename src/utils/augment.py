import numpy as np
import torch
import logging as lg
import random as r
import cv2
import pandas as pd

from torch import nn
from sklearn.cluster import KMeans
from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from torchvision.transforms import Resize

from src.utils.AdaIN import net
from src.utils.AdaIN.test import test_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StyleAugment(nn.Module):
    """Using a higher resolution for style transfer because lower resolution gives poor results
    """
    def __init__(self, input_size=32, transfer_size=128, return_style=False, **kwargs):
        super().__init__()
        self.ycbcr_ratio = kwargs.get('ycbcr_ratio', 1)
        self.samples = kwargs.get('samples',1)
        self.min_alpha = kwargs.get('min_alpha', 1)
        self.max_alpha = kwargs.get('max_alpha', 1)
        self.transfer_size = transfer_size
        self.input_size = input_size
        self.return_style = return_style

        self.decoder = net.decoder
        self.decoder.load_state_dict(torch.load("src/utils/AdaIN/models/decoder.pth"))

        self.vgg = net.vgg
        self.vgg.load_state_dict(torch.load("src/utils/AdaIN/models/vgg_normalised.pth"))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        # if torch.cuda.device_count() > 1 and self.parallel:
        #     lg.info(f"Training using multiple GPUs : {torch.cuda.device_count()}")
        #     self.decoder = nn.DataParallel(self.decoder)
        #     self.vgg = nn.DataParallel(self.vgg)
        
        self.decoder = self.decoder.to(device)
        self.vgg = self.vgg.to(device)

        self.n_added_so_far = 0
        self.register_buffer('style_std', torch.FloatTensor(300, 512).fill_(0))
        self.register_buffer('style_mean', torch.FloatTensor(300, 512).fill_(0))

        self.daa = kwargs.get('daa', True)

    def forward(self, content_im, style_im, **kwargs):
        with torch.no_grad():
            content_tf = test_transform(size=self.transfer_size, crop=True)
            style_tf = test_transform(size=self.transfer_size, crop=True)
            
            content = content_tf(content_im)
            style_id = np.random.random_integers(0, len(style_im)-1, size=self.samples).tolist()
            style = style_tf(style_im)[style_id]
            style = style.to(device)
            content = content.to(device)
            
            if self.return_style:
                output, std, mean = self.style_transfer(
                    self.vgg,
                    self.decoder,
                    content,
                    style,
                    alpha=r.uniform(self.min_alpha, self.max_alpha),
                    return_style=self.return_style
                )
                self.update_style(std, mean)
            else:
                output = self.style_transfer(
                    self.vgg ,
                    self.decoder,
                    content,
                    style,
                    alpha=r.uniform(self.min_alpha, self.max_alpha),
                    return_style=self.return_style
                    )
            
            content_ycb = rgb_to_ycbcr(content)
            output_ycb = rgb_to_ycbcr(output)

            output_ycb[:, 0, :, :] = output_ycb[:, 0, :, :] * self.ycbcr_ratio + content_ycb[:, 0, :, :] * (1 - self.ycbcr_ratio)

            output_rgb = ycbcr_to_rgb(output_ycb)

            output_rgb = Resize((self.input_size, self.input_size))(output_rgb)
            return output_rgb
    
    def update_style(self, std, mean):
        if self.n_added_so_far < len(self.style_std):
            self.style_std[self.n_added_so_far] = std.view(-1)
            self.style_mean[self.n_added_so_far] = mean.view(-1)
            self.n_added_so_far += 1
        else:
            idx = r.randint(0, len(self.style_std) -1)
            self.style_std[idx] = std.view(-1)
            self.style_mean[idx] = mean.view(-1)
    
    def get_style(self):
        idx = r.randint(0, min(len(self.style_std), self.n_added_so_far)-1)
        mem_std = self.style_std[idx].to(device)
        mem_mean = self.style_mean[idx].to(device)
        return mem_std, mem_mean
    
    def adaptive_instance_normalization(self, content_feat, style_feat, return_style=False, *kwargs):
        size = content_feat.size()
        n_content = content_feat.size(0)
        n_style = style_feat.size(0)
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

        if return_style:
            if self.n_added_so_far == 0:
                mem_std, mem_mean = style_std, style_mean
                new_feat = normalized_feat * mem_std.view(1, normalized_feat.size(1), 1, 1).expand(size) \
                            + mem_mean.view(1, normalized_feat.size(1), 1, 1).expand(size)
            else:
                mem_std = torch.cat([self.get_style()[0].view(1,-1,1,1) for _ in range(len(normalized_feat))], dim=0)
                mem_mean = torch.cat([self.get_style()[1].view(1,-1,1,1) for _ in range(len(normalized_feat))], dim=0)
                new_feat = normalized_feat * mem_std.view(normalized_feat.size(0), normalized_feat.size(1), 1, 1).expand(size) \
                            + mem_mean.view(normalized_feat.size(0), normalized_feat.size(1), 1, 1).expand(size)
                    
            return new_feat, style_std, style_mean
        else:
            return normalized_feat * style_std.repeat(n_content // n_style, 1, 1, 1).expand(size) \
                 + style_mean.repeat(n_content // n_style, 1, 1, 1).expand(size)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def style_transfer(self, vgg, decoder, content, style, alpha=1.0,
                    interpolation_weights=None, return_style=False, *kwargs):
        with torch.no_grad():
            assert (0.0 <= alpha <= 1.0)
            content_f = vgg(content)
            style_f = vgg(style)
            
            # k-means over stylish vectors if there is enough samples
            if style_f.size(0) >= 100:
                style_np = style_f.view(style_f.size(0), -1).cpu().numpy()
                kmeans = KMeans(n_clusters=5, random_state=0).fit(style_np)
                style_f = kmeans.cluster_centers_.reshape(5, 512, 16, 16)
                style_f = torch.from_numpy(style_f).to(device)

            if interpolation_weights:
                _, C, H, W = content_f.size()
                feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
                if return_style:
                    base_feat, std, mean = self.adaptive_instance_normalization(content_f, style_f, return_style)
                else:
                    base_feat = self.adaptive_instance_normalization(content_f, style_f, return_style)
                for i, w in enumerate(interpolation_weights):
                    feat = feat + w * base_feat[i:i + 1]
                content_f = content_f[0:1]
            else:
                if return_style:
                    feat, std, mean = self.adaptive_instance_normalization(content_f, style_f, return_style)
                else:
                    feat = self.adaptive_instance_normalization(content_f, style_f, return_style)
            feat = feat * alpha + content_f * (1 - alpha)
            del content_f
            del style_f
            torch.cuda.empty_cache()
            if return_style:
                return decoder(feat), std, mean
            else:
                return decoder(feat)


class MixUpAugment(nn.Module):
    """Mixup images from stream VS memory for data augmentation.
        It assumes len(batch1) > len(batch2).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.min_mix = kwargs.get('min_mix', 0.5)
        self.max_mix = kwargs.get('max_mix', 1.0)

    def forward(self, batch1, batch2, **kwargs):
        coef = r.uniform(self.min_mix, self.max_mix)
        n_concat = (len(batch1) // len(batch2)) + 1
        batch2_ext = torch.cat(
            [batch2[np.random.permutation(len(batch2)).tolist()] for _ in range(n_concat)],
            dim=0
        ).to(device)
        output = batch1 * coef + batch2_ext[:len(batch1)] * (1 - coef)

        return output


class CutMixAugment(nn.Module):
    """CutMix images from stream VS memory for data augmentation.
        It assumes len(batch1) > len(batch2).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.min_mix = kwargs.get('min_mix', 0.5)
        self.max_mix = kwargs.get('max_mix', 1.0)

    def forward(self, batch1, batch2, **kwargs):
        coef = r.uniform(self.min_mix, self.max_mix)
        n_concat = (len(batch1) // len(batch2)) + 1
        batch2_ext = torch.cat(
            [batch2[np.random.permutation(len(batch2)).tolist()] for _ in range(n_concat)],
            dim=0
        ).to(device)
        output = torch.cat([self.cutmix(im1, im2, coef).unsqueeze(0) for im1, im2 in zip(batch1, batch2_ext[:len(batch1)])], dim=0)

        return output

    def cutmix(self, im1, im2, lambd):
        """Cutmix im1 with im2 according to lambd
        """
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(im1.size(), lambd)

        output = im1.detach().clone()
        output[:, bbx1:bbx2, bby1:bby2] = im2[:, bbx1:bbx2, bby1:bby2]
        return output

    def rand_bbox(self, size, lam):
        lam = lam
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class SaliencyMixAugment(nn.Module):
    """ SaliencyMix augmentation.

    "SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better
    Regularization (https://arxiv.org/pdf/2006.01791.pdf)". In ICLR, 2021.
        https://github.com/SaliencyMix/SaliencyMix/blob/main/SaliencyMix_CIFAR/saliencymix.py
    
    Args:
        img (Tensor): Input images of shape (C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        min_mix = kwargs.get('min_mix', 0.5)
        self.min_mix = min_mix
    
    def saliency_bbox(self, img, lam):
        """Code inspired from : https://github.com/Westlake-AI/openmixup
            generate saliency box by lam
        """
        size = img.size()
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # force fp32 when convert to numpy
        img = img.type(torch.float32)

        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        maximum_indices = np.unravel_index(
            np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]

        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def saliency_mix(self, im1, im2, lam):
        # detect saliency box
        bbx1, bby1, bbx2, bby2 = self.saliency_bbox(im2, lam)
        im1[:, :, bbx1:bbx2, bby1:bby2] = im2[:, :, bbx1:bbx2, bby1:bby2]

        return im1

    def forward(self, batch1, batch2, **kwargs):       
        # Select image to mix with the rest
        coef = r.uniform(self.min_mix, 1)
        n_concat = (len(batch1) // len(batch2)) + 1
        batch2_ext = torch.cat(
            [batch2[np.random.permutation(len(batch2)).tolist()] for _ in range(n_concat)],
            dim=0
        ).to(device)
        output = torch.cat([self.saliency_mix(im1, im2, coef).unsqueeze(0) for im1, im2 in zip(batch1, batch2_ext[:len(batch1)])], dim=0)

        return output


class JFMixAugment(nn.Module):
    """bla bla bla
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.min_mix = kwargs.get('min_mix', 0.5)
        self.max_mix = kwargs.get('max_mix', 1.0)
        self.n_samples = kwargs.get('n_samples', 5)
        # self.selected_lambs = []

    def jfmix(self, im1, im2, model):
        if model is None:
            raise Warning("Bad implem. Need to get model.")
        model.eval()
        lams = [r.uniform(self.min_mix, self.max_mix) for _ in range(self.n_samples)]
        # lams = np.arange(0,1.05, 0.1).tolist()
        mixs = torch.cat([(im1 * lam + im2 * (1 - lam)).unsqueeze(0) for lam in lams], dim=0)
        target = model(im1.unsqueeze(0))[1]
        query = model(mixs)[1]
        sims = target @ query.T
        idx = sims.argmin()
        # self.selected_lambs.append(lams[idx])

        return mixs[idx]
    
    def save(self):
        pass
        # pd.DataFrame(self.selected_lambs).to_csv("./results/lambs.csv", index=False)

    def forward(self, batch1, batch2, **kwargs):
        model = kwargs.get('model', None)
        # Select image to mix with the rest
        n_concat = (len(batch1) // len(batch2)) + 1
        batch2_ext = torch.cat(
            [batch2[np.random.permutation(len(batch2)).tolist()] for _ in range(n_concat)],
            dim=0
        ).to(device)
        output = torch.cat([self.jfmix(im1, im2, model).unsqueeze(0) for im1, im2 in zip(batch1, batch2_ext[:len(batch1)])], dim=0)

        return output