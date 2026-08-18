import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IdentityDataset, BASE_PATH
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a closed-set body identity classifier.")
    parser.add_argument("--mode", default="pressure", choices=["pressure", "depth_cover1", "depth_cover2", "depth_uncover"])
    parser.add_argument("--train_split", default="real_train.txt")
    parser.add_argument("--val_split", default="real_val.txt")
    parser.add_argument("--model", default="convnextv2_base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(Path("/home/shnh/DATA/zjy/BodyMAP_identity")))
    parser.add_argument("--limit_subjects", type=int, default=None)
    parser.add_argument("--limit_poses", type=int, default=None)
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(all_logits, dim=1)
    subject_correct = 0
    subject_total = 0
    for label_idx in torch.unique(all_labels).tolist():
        mask = all_labels == label_idx
        subject_prob = probs[mask].mean(dim=0)
        subject_pred = subject_prob.argmax().item()
        subject_correct += int(subject_pred == label_idx)
        subject_total += 1

    return {
        "loss": total_loss / max(total, 1),
        "acc_sample": correct / max(total, 1),
        "acc_top5": top5_correct / max(total, 1),
        "acc_subject": subject_correct / max(subject_total, 1),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    train_dataset = IdentityDataset(
        args.train_split,
        mode=args.mode,
        limit_subjects=args.limit_subjects,
        limit_poses=args.limit_poses,
    )
    val_dataset = IdentityDataset(
        args.val_split,
        mode=args.mode,
        limit_subjects=args.limit_subjects,
        limit_poses=args.limit_poses,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(args.model, train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "metrics.jsonl"

    config = vars(args)
    config["base_path"] = str(BASE_PATH)
    config["train_subjects"] = train_dataset.subject_ids
    config["val_subjects"] = val_dataset.subject_ids
    config["train_samples"] = len(train_dataset)
    config["val_samples"] = len(val_dataset)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    best_subject_acc = 0.0
    metrics_fp = metrics_path.open("a", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        val_metrics = evaluate(model, val_loader, criterion, device)
        train_acc = running_correct / max(running_total, 1)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_total, 1),
            "train_acc_sample": train_acc,
            **val_metrics,
            "seconds": round(time.time() - start, 2),
        }
        metrics_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_fp.flush()

        print(
            f"epoch {epoch:02d} train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['acc_sample']:.4f} "
            f"val_top5={val_metrics['acc_top5']:.4f} "
            f"val_subject_acc={val_metrics['acc_subject']:.4f}"
        )

        if val_metrics["acc_subject"] > best_subject_acc:
            best_subject_acc = val_metrics["acc_subject"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": train_dataset.num_classes,
                    "mode": args.mode,
                    "label_to_idx": train_dataset.label_to_idx,
                    "idx_to_label": train_dataset.idx_to_label,
                },
                checkpoint_path,
            )

    metrics_fp.close()
    print(f"Best subject-level accuracy: {best_subject_acc:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()