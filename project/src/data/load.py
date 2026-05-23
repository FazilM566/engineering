from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_loaders(data_root, batch_size=64, img_size=224, num_workers=2):
    transform_train = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = Path(data_root)
    train_loader = DataLoader(
        datasets.ImageFolder(data_root / "train", transform=transform_train),

    )
    val_loader = DataLoader(
        datasets.ImageFolder(data_root / "val", transform=transform_val),

    )
    test_loader = DataLoader(
        datasets.ImageFolder(data_root / "test", transform=transform_val),

    )
    return train_loader, val_loader, test_loader
