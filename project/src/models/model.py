import torch.nn as nn
from torchvision.models import resnet50

def create_model(num_classes=4, pretrained=True):
    model = resnet50(weights=None if not pretrained else "DEFAULT")
    model.fc = nn.Linear(2048, num_classes)
    return model