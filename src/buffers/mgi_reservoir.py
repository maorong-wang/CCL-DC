
import torch
import random as r
import numpy as np
import copy

import torch.nn.functional as F

from src.buffers.reservoir import Reservoir
from src.utils.utils import get_device

device = get_device()


class MGIReservoir(Reservoir):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        """Reservoir sampling for memory update
        Args:
            max_size (int, optional): maximum buffer size. Defaults to 200.
            img_size (int, optional): Image width/height. Images are considered square. Defaults to 32.
            nb_ch (int, optional): Number of image channels. Defaults to 3.
            n_classes (int, optional): Number of classes expected total. For print purposes only. Defaults to 10.
        """
        self.shape = kwargs.get('shape', None)
        self.drop_method = kwargs.get('drop_method', 'random')
        super().__init__(
            max_size,
            img_size=img_size,
            nb_ch=nb_ch,
            n_classes=n_classes,
            )
        self.learning_rate = kwargs.get('lr', 0.1)
        self.transform = kwargs.get('tf', None)
        self.subsample = 50

    def mgi_retrieve(self, n_imgs, model, out_dim,  **kwargs):
        sub_x, sub_y = self.random_retrieve(n_imgs=self.subsample)
        sub_x, sub_y = sub_x.to(device), sub_y.to(device)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = self.get_grad_vector(model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                sub_x_aug = self.transform(sub_x)
                logits_pre = model(sub_x,sub_x_aug)
                logits_post = model_temp(sub_x,sub_x_aug)

                z_pre, zt_pre, zzt_pre,fea_z_pre = logits_pre
                z_post, zt_post, zzt_post,fea_z_post = logits_post


                grads_pre_z= torch.sum(torch.abs(F.softmax(z_pre, dim=1) - F.one_hot(sub_y, out_dim)), 1)
                mgi_pre_z = grads_pre_z * fea_z_pre[0].reshape(-1)
                grads_post_z = torch.sum(torch.abs(F.softmax(z_post, dim=1) - F.one_hot(sub_y, out_dim)), 1)  # N * 1
                mgi_post_z = grads_post_z * fea_z_post[0].reshape(-1)


                scores = mgi_post_z - mgi_pre_z

                big_ind = scores.sort(descending=True)[1][:int(n_imgs)]
            return sub_x[big_ind], sub_x_aug[big_ind],sub_y[big_ind]
        else:
            return sub_x, sub_x,sub_y
    
    def get_grad_vector(self, pp, grad_dims):
        """
            gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims)).to(device)
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads
    
    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
