import torch
import random as r
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.buffers.buffer import Buffer
from src.models.dti.model import get_model

class ProtoBuf(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, **kwargs):
        self.shape = kwargs.get('shape', None)
        if self.shape is not None:
            super().__init__(max_size, shape=self.shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.betas = [[] for _ in range(self.max_size)]
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.dti_params = {
            "n_prototypes": int(max_size / 5),
            "encoder_name": "resnet110",
            "init_type": "kmeans",
            "transformation_sequence": "identity_color_affine_tps",
            # "transformation_sequence": "identity_color_projective",
            # "curriculum_learning": [5, 15],
            # "grid_size": 4
        }

    def update(self, imgs, labels, **kwargs):
        N_IGNORE = 5000
        epochs = 500
        if not self.is_full():
            for stream_img, stream_label in zip(imgs, labels):
                self.stack_data(stream_img, stream_label)
                self.n_seen_so_far += 1
        else:
            reservoir_indices = []
            for im in imgs:
                reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
                if reservoir_idx < self.max_size:
                    # Do not DTI at the beginning to speed up training
                    if self.n_seen_so_far < N_IGNORE:
                        for res_id in reservoir_indices:
                            self.buffer_imgs[reservoir_idx] = im
                    else:
                        reservoir_indices.append(reservoir_idx)
                self.n_seen_so_far += 1
            if len(reservoir_indices) > 0 and self.n_seen_so_far > N_IGNORE:
                # Initialisation for DTI training
                buf_im, _ = self.random_retrieve(n_imgs=self.max_size)
                # buf_im = self.buffer_imgs
                cluster_im = torch.cat([imgs, buf_im], dim=0)
                B = cluster_im.size(0)
                model = get_model("dtikmeans")(data_batch=cluster_im, **self.dti_params).to(self.device)
                self.tf_seq = model.transformer.tsf_sequences[0].tsf_modules[1:]
                optim = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
                scheduler = MultiStepLR(optim, gamma=0.1, milestones=[300, 400])
                # DTI training
                for _ in range(epochs):
                    model.train()
                    cluster_im = cluster_im.to(self.device)
                    optim.zero_grad()
                    loss, distances = model(cluster_im)
                    loss.backward()
                    optim.step()
                    scheduler.step()
                with torch.no_grad():
                    # Getting the best betas - prototypes associations
                    prototypes = model.prototypes.unsqueeze(1).expand(-1, cluster_im.size(0), cluster_im.size(1), -1, -1)
                    inp, target, betas = model.transformer(cluster_im, prototypes, ret_beta=True)
                    
                    distances = (inp - target)**2
                    distances = distances.flatten(2).mean(2)
                    
                    best_proto = distances.argmin(1)
                    
                    # ranking the protos according to the distance to the closest prototype
                    ranks = []
                    for proto_id in range(self.dti_params['n_prototypes']):
                        indices = (best_proto == proto_id).nonzero()
                        loss_proto = ((target[indices] - inp[indices]) ** 2).mean()
                        ranks.append([loss_proto.cpu().item(), proto_id])
                    ranks = sorted(ranks, key=lambda x: x[0], reverse=True)
                    ranked_protos = [r[1] for r in ranks]

                    # Filling proto and betas
                    # selected_protos = r.sample(range(prototypes.size(0)), prototypes.size(0))
                    for i, res_id in enumerate(reservoir_indices):
                        proto_idx = ranked_protos[i]
                        selected_betas = []
                        candidate_betas_ids = (best_proto == proto_idx).nonzero()
                        if len(candidate_betas_ids) >= 3:
                            for beta_id in candidate_betas_ids:
                                selected_betas.append(betas[beta_id, best_proto[beta_id], :])
                            selected_betas = torch.stack(selected_betas).squeeze(1)
                            self.buffer_imgs[res_id] = model.prototypes[proto_idx]
                            self.betas[res_id] = selected_betas

    def random_retrieve(self, n_imgs=100):
        with torch.no_grad():
            if self.n_added_so_far == 0:
                return torch.Tensor(), torch.Tensor()
            ret_imgs = []
            for _ in range(n_imgs):
                idx = r.randint(0, min(self.max_size, self.n_added_so_far) - 1)
                if len(self.betas[idx]) == 0:
                    ret_imgs.append(self.buffer_imgs[idx])
                else:
                    beta = self.betas[idx][r.randint(0, len(self.betas[idx])-1)]
                    beta_l = beta.split([6,6,32])
                    base_im = self.buffer_imgs[idx]
                    base_im = base_im.unsqueeze(0)
                    for beta, tf in zip(beta_l, self.tf_seq):
                        base_im = tf.transform(base_im.to(self.device), beta.unsqueeze(0).to(self.device))
                    ret_imgs.append(base_im[0].to('cpu'))

            return torch.stack(ret_imgs, dim=0), torch.Tensor()