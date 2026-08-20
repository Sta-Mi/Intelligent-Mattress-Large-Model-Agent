"""Train the reproducible pressure-only BodyPressureSD pose baseline."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from BodyPressure.pose_estimation.dataset import BodyPressureSDPoseDataset
from BodyPressure.pose_estimation.model import PressurePoseTransformer, mpjpe


def read_file_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


class FileLocalBatchSampler(Sampler):
    """Shuffle files and samples while keeping batches within one pickle.

    Python pickles are not random-access. A global RandomSampler would repeatedly
    reload several-hundred-MB files, so batches retain file locality instead.
    """

    def __init__(self, dataset, batch_size, seed=42):
        self.ranges, self.batch_size, self.generator = dataset.file_ranges, batch_size, torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        for file_index in torch.randperm(len(self.ranges), generator=self.generator).tolist():
            indices = list(self.ranges[file_index])
            order = torch.randperm(len(indices), generator=self.generator).tolist()
            for start in range(0, len(order), self.batch_size):
                yield [indices[i] for i in order[start:start + self.batch_size]]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self.ranges)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_error = total_samples = 0
    for batch in loader:
        pressure = batch["pressure"].to(device)
        target = batch["joints"].to(device)
        with torch.set_grad_enabled(training):
            error = mpjpe(model(pressure), target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                error.backward()
                optimizer.step()
        total_error += error.item() * pressure.shape[0]
        total_samples += pressure.shape[0]
    return total_error / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_files", required=True)
    parser.add_argument("--val_files", required=True)
    parser.add_argument("--out_dir", default="runs/pressure_pose_transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, out_dir = torch.device(args.device), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_data = BodyPressureSDPoseDataset(args.data_root, read_file_list(args.train_files))
    val_data = BodyPressureSDPoseDataset(args.data_root, read_file_list(args.val_files))
    print(f"data_root={train_data.root} train={len(train_data)} val={len(val_data)}")
    train_sampler = FileLocalBatchSampler(train_data, args.batch_size, args.seed)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    model = PressurePoseTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_error = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_error = run_epoch(model, val_loader, None, device)
        metrics = {"epoch": epoch, "train_mpjpe_mm": train_error * 1000,
                   "val_mpjpe_mm": val_error * 1000}
        print(json.dumps(metrics))
        with (out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if val_error < best:
            best = val_error
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "metrics": metrics}, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
