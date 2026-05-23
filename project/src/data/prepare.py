# src/data/prepare.py
import argparse
import random
import shutil
from pathlib import Path

def prepare_dataset(input_dir, output_dir, ratios=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        files = list(class_dir.glob("*.*"))
        random.shuffle(files)
        n_train = int(len(files) * ratios[0])
        n_val = int(len(files) * ratios[1])
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        for split, flist in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            dest = output_dir / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in flist:
                shutil.copy2(f, dest / f.name)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_dataset(args.input, args.output, tuple(args.ratios), args.seed)