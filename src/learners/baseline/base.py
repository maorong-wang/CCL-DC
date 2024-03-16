from random import random
import torch
import logging as lg
import os
import pickle
import time
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random as r
import torchvision
import wandb
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from torchvision import transforms
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert
from copy import deepcopy
from torchvision.transforms import RandAugment
from umap import UMAP
from torch.distributions import Categorical

from src.utils.utils import save_model
from src.models import resnet
from src.utils.metrics import forgetting_line 
from src.utils.utils import get_device

device = get_device()

class BaseLearner(torch.nn.Module):
    def __init__(self, args):
        """Learner abstract class
        Args:
            args: argparser arguments
        """
        super().__init__()
        self.params = args
        self.multiplier = self.params.multiplier
        self.device = get_device()
        self.init_tag()
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.buffer = None
        self.start = time.time()
        self.criterion = self.load_criterion()
        if self.params.tensorboard:
            self.writer = self.load_writer()
        self.classifiers_list = ['ncm']  # Classifiers used for evaluating representation
        self.loss = 0
        self.stream_idx = 0
        self.results = []
        self.results_clustering = []
        self.results_forgetting = []
        self.results_clustering_forgetting = []
        
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.params.dataset == 'tiny' else nn.Identity()
        normalize = nn.Identity()
        if self.params.tf_type == 'partial':
            self.transform_train = nn.Sequential(
                torchvision.transforms.RandomCrop(self.params.img_size, padding=4),
                RandomHorizontalFlip(),
                normalize
            ).to(device)
        elif self.params.tf_type == 'moderate':
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.)),
                RandomHorizontalFlip(),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)
        else:
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)

        self.transform_test = nn.Sequential(
            normalize
        ).to(device)

    def init_tag(self):
        """Initialise tag for experiment
        """
        if self.params.training_type == 'inc':
            self.params.tag = f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}{self.params.tag}"
        elif self.params.training_type == "blurry":
            self.params.tag =\
                 f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}blurry{self.params.blurry_scale}{self.params.tag}"
        else:
            self.params.tag = f"{self.params.learner},{self.params.dataset},{self.params.epochs}b{self.params.batch_size},uni{self.params.tag}"
        print(f"Using the following tag for this experiment : {self.params.tag}")
    
    def load_writer(self):
        """Initialize tensorboard summary writer
        """
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(self.params.tb_root, self.params.tag))
        return writer

    def save(self, path):
        lg.debug("Saving checkpoint...")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        
        with open(os.path.join(path, f"memory.pkl"), 'wb') as memory_file:
            pickle.dump(self.buffer, memory_file)

    def resume(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        with open(os.path.join(path, f"memory.pkl"), 'rb') as f:
            self.buffer = pickle.load(f)
        f.close()
        torch.cuda.empty_cache()


    def load_model(self):
        """Load model
        Returns:
            untrained torch backbone model
        """
        return NotImplementedError


    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                )
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_scheduler(self):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError
    
    def train(self, dataloader, task, **kwargs):
        raise NotImplementedError

    def evaluate(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()
            accs = []
            preds = []
            all_targets = []
            tag = '' 
            for j in range(task_id + 1):
                test_preds, test_targets = self.encode_logits(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)

                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds = np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds= [self.params.labels_order.index(int(i)) for i in preds]
                cm= np.log(1 + confusion_matrix(all_targets, preds))
                fig = plt.matshow(cm)
                wandb.log({
                        tag + f"cm": fig,
                        "task_id": task_id
                    })
                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1]) 

    def evaluate_offline(self, dataloaders, epoch=0, eval_ema=False):
        with torch.no_grad():
            self.model.eval()
            test_preds, test_targets = self.encode_logits(dataloaders['test'])
            acc= accuracy_score(test_targets, test_preds)
            self.results.append(acc)

        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
    
    def evaluate_clustering(self, dataloaders, task_id, eval_ema=False, **kwargs):
        try:
            results = self._evaluate_clustering(dataloaders, task_id, eval_ema=eval_ema, **kwargs)
        except:
            results = 0, 0
        return results

    def _evaluate_clustering(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            mem_representations, mem_labels = self.get_mem_rep_labels(use_proj=self.params.eval_proj)

            # UMAP visualization
            # reduction = self.umap_reduction(mem_representations.cpu().numpy())
            # plt.figure()
            # figure = plt.scatter(reduction[:, 0], reduction[:, 1], c=mem_labels, cmap='Spectral', s=1)
            # if not self.params.no_wandb:
            #     wandb.log({
            #         "umap": wandb.Image(figure),
            #         "task_id": task_id
            #     })

            
            classifiers= self.init_classifiers()
            classifiers= self.fit_classifiers(classifiers=classifiers, representations=mem_representations, labels=mem_labels)
            
            accs = []
            representations= {}
            targets= {}
            preds= []
            all_targets = []
            tag = ''

            for j in range(task_id + 1):
                test_representation, test_targets = self.encode_fea(dataloaders[f"test{j}"])
                representations[f"test{j}"] = test_representation
                targets[f"test{j}"] = test_targets

                test_preds= classifiers[0].predict(representations[f'test{j}'])

                acc= accuracy_score(targets[f"test{j}"], test_preds) 

                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds= np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ncm_acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds= [self.params.labels_order.index(int(i)) for i in preds]
                cm= np.log(1 + confusion_matrix(all_targets, preds))
                fig = plt.matshow(cm)
                wandb.log({
                        tag + f"ncm_cm": fig,
                        "task_id": task_id
                    })

                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_clustering.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results_clustering), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_clustering_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results_clustering[-1]), np.nanmean(self.results_clustering_forgetting[-1])


    def encode_fea(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)

                feat = self.model(self.transform_test(inputs))

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= feat.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.vstack([all_feat, feat.cpu().numpy()])

        return all_feat, all_labels

    def encode_logits(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)

                feat = self.model.logits(self.transform_test(inputs))

                preds= feat.argmax(dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def init_classifiers(self):
        """Initiliaze every classifier for representation transfer learning
        Returns:
            List of initialized classifiers
        """
        # logreg = LogisticRegression(random_state=self.params.seed)
        # knn = KNeighborsClassifier(3)
        # linear = MLPClassifier(hidden_layer_sizes=(200), activation='identity', max_iter=500, random_state=self.params.seed)
        # svm = SVC()
        ncm = NearestCentroid()
        # return [logreg, knn, linear, svm, ncm]
        return [ncm]
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit_classifiers(self, classifiers, representations, labels):
        """Fit every classifiers on representation - labels pairs
        Args:
            classifiers : List of sklearn classifiers
            representations (torch.Tensor): data representations 
            labels (torch.Tensor): data labels
        Returns:
            List of trained classifiers
        """
        for clf in classifiers:
            clf.fit(representations.cpu(), labels)
        return classifiers

    def after_eval(self, **kwargs):
        pass

    def before_eval(self, **kwargs):
        pass

    def train_inc(self, **kwargs):
        raise NotImplementedError

    def train_blurry(self, **kwargs):
        raise NotImplementedError

    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: 
            self.model.eval()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations= torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels

    def save_results(self):
        pass
    
    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")


    def umap_reduction(self, representation):
        umap_reducer = UMAP()
        umap_result = umap_reducer.fit_transform(representation)
        return umap_result

    def backward_transfer(self):
        n_tasks = len(self.results)
        bt = 0
        for i in range(1, n_tasks):
            for j in range(i):
                bt += self.results[i][j] - self.results[j][j]
        
        return bt / (n_tasks * (n_tasks - 1) / 2)
                
    def learning_accuracy(self):
        n_tasks = len(self.results)
        la = 0
        for i in range(n_tasks):
                la += self.results[i][i]
        return la / n_tasks
    
    def reletive_forgetting(self):
        n_tasks = len(self.results)
        rf = 0
        max = np.nanmax(np.array(self.results), axis=0)
        for i in range(n_tasks-1):
            if max[i] != 0:
                rf += self.results_forgetting[-1][i] / max[i]
            else:
                rf += 1
        
        return rf / n_tasks
    
    def get_entropy(self, dataloaders, task_id):
        trainloader = dataloaders[f"train{task_id}"]
        testloader = dataloaders[f"test{task_id}"]

        train_ce = 0
        train_en = 0
        test_ce = 0
        test_en = 0
        samples = 0

        self.model.eval()

        for i, batch in enumerate(trainloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            train_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            train_en += Categorical(probs=prob).entropy().sum().item()

        train_ce /= samples
        train_en /= samples

        samples = 0

        for i, batch in enumerate(testloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            test_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            test_en += Categorical(probs=prob).entropy().sum().item()

        test_ce /= samples
        test_en /= samples

        self.model.train()
        return train_ce, train_en, test_ce, test_en