import os, sys, time
import typing as tp
import numpy as np
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter


def resize_tensor(
    x:    torch.Tensor, 
    size: int|tp.Tuple[int,int]|torch.Size, 
    mode: tp.Literal['nearest', 'bilinear'],
    align_corners: bool|None = None,
) -> torch.Tensor:
    assert torch.is_tensor(x)
    assert len(x.shape) in [3,4]
    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, size, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# Define the model
class MobileNetV3LC(nn.Module):
    def __init__(self, num_classes=2):
        # super(MobileNetV3LC, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights='DEFAULT', pretrained=True)
        # Freeze the base model (optional)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        # Replace the final classification layer
        self.mobilenet.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(72, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x