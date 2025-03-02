import numpy as np
import os
import re
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from typing import Callable


class ClientModel(nn.Module):
    
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        
        self.densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        self.num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
                                            nn.Linear(self.num_features, self.num_classes),
                                            nn.Sigmoid()
                                    )

        self.size = self.model_size()

    def forward(self, x):
        return self.densenet(x)

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size