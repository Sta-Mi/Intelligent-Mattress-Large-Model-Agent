import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IdentityDataset
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained identity classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="real_val.txt")
    parser.add_argument("--mode", default="pressure", choices=["pressure", "depth_cover1", "depth_cover2", "depth_uncover"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default=str(Path("/home/shnh/DATA/zjy/BodyMAP_identity/eval")))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = IdentityDataset(args.split, mode=args.mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item() / max(len(labels), 1)
    top5 = (
        torch.topk(logits, k=min(5, logits.shape[1]), dim=1)
        .indices.eq(labels.unsqueeze(1))
        .any(dim=1)
        .sum()
        .item()
        / max(len(labels), 1)
    )

    probs = torch.softmax(logits, dim=1)
    subject_correct = 0
    subject_total = 0
    per_subject = {}
    for label_idx in torch.unique(labels).tolist():
        mask = labels == label_idx
        subject_prob = probs[mask].mean(dim=0)
        subject_pred = subject_prob.argmax().item()
        subject_correct += int(subject_pred == label_idx)
        subject_total += 1
        per_subject[ckpt["idx_to_label"][str(label_idx)]] = {
            "true_label_idx": label_idx,
            "pred_label_idx": subject_pred,
            "pred_subject_id": ckpt["idx_to_label"].get(str(subject_pred), str(subject_pred)),
            "correct": int(subject_pred == label_idx),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "acc_sample": correct,
        "acc_top5": top5,
        "acc_subject": subject_correct / max(subject_total, 1),
        "subject_total": subject_total,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "predictions.json").write_text(json.dumps(per_subject, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Predictions saved to {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()