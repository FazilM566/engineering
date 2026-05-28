from functools import lru_cache
import io
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["COVID", "Lung opacity", "Normal", "Viral Pneumonia"]
MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "weights" / "best_model_full.pth"


class MedicalClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)


@lru_cache(maxsize=1)
def load_model():
    model = MedicalClassifier()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    fixed_state_dict = {
        f"backbone.{k}" if not k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(fixed_state_dict)
    model.to(DEVICE)
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_from_bytes(image_bytes: bytes) -> Dict:
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "prediction": CLASS_NAMES[pred_idx.item()],
        "confidence": round(conf.item(), 4),
        "probabilities": {name: round(p.item(), 4) for name, p in zip(CLASS_NAMES, probs)},
        "latency_ms": round(latency_ms, 2)
    }