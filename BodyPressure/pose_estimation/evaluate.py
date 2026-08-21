"""Evaluate a saved pressure-pose checkpoint with diagnostic joint metrics."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from BodyPressure.pose_estimation.dataset import BodyPressureSDPoseDataset
from BodyPressure.pose_estimation.model import PressurePoseTransformer, pose_errors
from BodyPressure.pose_estimation.train import read_file_list


SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PressurePoseTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = BodyPressureSDPoseDataset(args.data_root, read_file_list(args.files))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    absolute_sum = relative_sum = 0.0
    per_joint_sum = torch.zeros(24, device=device)
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            target = batch["joints"].to(device)
            errors = pose_errors(model(batch["pressure"].to(device)), target)
            count = target.shape[0]
            absolute_sum += errors["mpjpe"].item() * count
            relative_sum += errors["root_mpjpe"].item() * count
            per_joint_sum += errors["per_joint"] * count
            sample_count += count
    result = {
        "protocol": {
            "dataset": "BodyPressureSD",
            "domain": "synthetic",
            "modality": "PI",
            "split_file": str(Path(args.files).resolve()),
            "joint_convention": "SMPL-24",
            "coordinate_alignment": "absolute synthetic-bed frame",
        },
        "samples": sample_count,
        "checkpoint_epoch": checkpoint.get("metrics", {}).get("epoch"),
        "mpjpe_mm": absolute_sum / sample_count * 1000,
        "pelvis_aligned_mpjpe_mm": relative_sum / sample_count * 1000,
        "per_joint_mpjpe_mm": {
            name: value for name, value in zip(
                SMPL_JOINT_NAMES, (per_joint_sum / sample_count * 1000).cpu().tolist()
            )
        },
    }
    output = Path(args.out) if args.out else Path(args.checkpoint).with_name("evaluation.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
