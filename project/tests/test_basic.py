import sys
from pathlib import Path



import torch
from src.models.model import create_model
from src.utils.helpers import get_device

def test_model_forward():
    model = create_model(num_classes=4, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 4)

def test_device():
    device = get_device()
    assert device.type in ["cpu", "cuda"]