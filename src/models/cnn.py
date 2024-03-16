"""Customs CNN model for simple trainings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class SimpleNet(nn.Module):
    def __init__(self, proj_dim=128, rep_dim=512, proj=True, bn=False):
        super(SimpleNet, self).__init__()
        self.proj = proj

        # in_channels, out_channels, kernel_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2 * 2 * 512, rep_dim)
        )

        if self.proj:
            self.projector = nn.Linear(rep_dim, proj_dim)

        self.m = nn.MaxPool2d(2)

    def forward(self, x):
        feature = self.encoder(x)

        if self.proj:
            projection = self.projector(feature)

        return feature, projection
