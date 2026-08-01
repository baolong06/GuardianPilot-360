import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TrafficLightClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)