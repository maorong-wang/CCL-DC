# Improving Plasticity in Online Continual Learning via Collaborative Learning
Official implementation of the paper "Improving Plasticity in Online Continual Learning via Collaborative Learning". This paper is accepted by CVPR2024.

[![arXiv](https://img.shields.io/badge/arXiv-2312.00600-b31b1b.svg)](https://arxiv.org/abs/2312.00600)

## 1. Dataset
### CIFAR-10/100
Torchvision should be able to handle the CIFAR-10/100 dataset automatically. If not, please download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in the `data` folder.

### TinyImageNet
This codebase should be able to handle TinyImageNet dataset automatically and save them in the `data` folder. If not, please refer to [this github gist](https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54).

### ImageNet-100
Download the ImageNet dataset from [here](http://www.image-net.org/) and follow [this](https://github.com/danielchyeh/ImageNet-100-Pytorch) for ImageNet-100 dataset generation. Put the dataset in the `imagenet100_data` folder. Symbolic links are highly recommended.

## 2. Reproduce our results
We use the following hardware and software for our experiments:
- Hardware: NVIDIA Tesla A100 GPUs
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## 3. Training
Weight and bias is highly recommended to run the training, and some features in this codebase is only available with weight and bias. However, it is possible to excute the training without weight and bias.

### Training with a configuration file
Training can be done by specifying the dataset path and params in a configuration file, for example:

```
python main.py --data-root ./data --config ./config/CVPR24/cifar10/ER,c10,m500.yaml
```

Although we have attached the best hyperparameter with our search in `./config/CVPR24`, we highly suggest to use weight and bias for the training.

### Training with weight and bias sweep (Recommended)
Weight and bias sweep is originally designed for hyperparameter search. However, it make the multiple runs much easier. Training can be done with W&B sweep more elegantly, for example:

```
wandb sweep sweeps/CVPR/ER,cifar10.yaml
```

Note that you need to set the dataset path in .yaml file by specify `--data-root-dir`. And run the sweep agent with:

```
wandb agent $sweepID
```

The hyperparameters after our hyperparameter search is located at `./sweeps/CVPR`.

## 4. Model / memory buffer snapshots
We save the model and memory buffer status after training for evaluation. After the training process, the model should be saved at `./checkpoints/$dataset/$learner/$memory_size/$rand_seed/model.pth` and memory buffer should be located at  `./checkpoints/$dataset/$learner/$memory_size/$rand_seed/memory.pkl`.  

## Acknowledgement
Special thanks to co-author Nicolas. Our implementation is based on his work [AGD-FD's codebase](https://github.com/Nicolas1203/ocl-fd).