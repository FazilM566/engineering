import argparse
import torch
import torch.nn as nn
from src.data.load import create_loaders
from src.models.model import create_model
from src.models.train import evaluate
from src.utils.helpers import set_seed, get_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    _, _, test_loader = create_loaders(args.data, batch_size=args.batch_size)
    model = create_model(num_classes=4, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()