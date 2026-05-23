import argparse
import torch
import torch.nn as nn
from src.data.load import create_loaders
from src.models.model import create_model
from src.models.train import fit
from src.utils.helpers import set_seed, get_device, set_requires_grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--epochs-head", type=int, default=15, help="Эпохи для головы")
    parser.add_argument("--epochs-ft", type=int, default=10, help="Эпохи для fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="LR для головы")
    parser.add_argument("--lr-ft", type=float, default=1e-4, help="LR для fine-tuning")
    parser.add_argument("--save", default="artifacts/weights/best_model_full.pth", help="Куда сохранить модель")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _ = create_loaders(args.data, batch_size=args.batch_size)
    model = create_model(num_classes=4, pretrained=True).to(device)

    # ---------- Phase 1: head-only training ----------
    print("\n" + "="*60)
    print("Phase 1: head-only training")
    set_requires_grad(model, False)
    set_requires_grad(model.fc, True)
    optimizer_head = torch.optim.Adam(model.fc.parameters(), lr=args.lr_head)
    criterion = nn.CrossEntropyLoss()
    fit(model, train_loader, val_loader, optimizer_head, criterion, device,
        epochs=args.epochs_head, save_path=args.save, verbose=True)

    # ---------- Phase 2: fine-tuning layer4 + fc ----------
    print("\n" + "="*60)
    print("Phase 2: fine-tuning layer4 + fc")
    set_requires_grad(model, False)
    set_requires_grad(model.layer4, True)
    set_requires_grad(model.fc, True)
    params = [
        {"params": model.layer4.parameters(), "lr": args.lr_ft},
        {"params": model.fc.parameters(), "lr": args.lr_head},
    ]
    optimizer_ft = torch.optim.Adam(params, weight_decay=1e-4)

    fit(model, train_loader, val_loader, optimizer_ft, criterion, device,
        epochs=args.epochs_ft, save_path=args.save, verbose=True)

if __name__ == "__main__":
    main()