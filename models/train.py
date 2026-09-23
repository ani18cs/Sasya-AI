"""
Script to train the ResNet-50 model.
"""
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V2")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    pass
