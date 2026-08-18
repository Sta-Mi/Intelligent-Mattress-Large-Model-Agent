from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data_BP" / "slp_real_cleaned"
SPLIT_PATH = BASE_PATH / "BodyMAP" / "data_files"


MODES = {
    "pressure": "pressure_recon_Pplus_gt_0to102.npy",
    "depth_cover1": "depth_cover1_cleaned_0to102.npy",
    "depth_cover2": "depth_cover2_cleaned_0to102.npy",
    "depth_uncover": "depth_uncover_cleaned_0to102.npy",
}


def load_subject_ids(split_file: str, limit_subjects: int | None = None):
    split_path = SPLIT_PATH / split_file
    lines = [
        line.strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    ]
    if limit_subjects is not None:
        lines = lines[:limit_subjects]
    return lines


class IdentityDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        mode: str = "pressure",
        limit_subjects: int | None = None,
        limit_poses: int | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}, choose from {list(MODES)}")

        self.mode = mode
        self.subject_ids = load_subject_ids(split_file, limit_subjects)
        self.label_to_idx = {sid: i for i, sid in enumerate(self.subject_ids)}
        self.idx_to_label = {i: sid for sid, i in self.label_to_idx.items()}
        self.num_classes = len(self.subject_ids)

        array_path = DATA_PATH / MODES[mode]
        self.data = np.load(array_path, mmap_mode="r")

        self.samples = []
        for local_idx, subject_id in enumerate(self.subject_ids):
            person_idx = int(subject_id) - 1
            pose_count = self.data.shape[1]
            if limit_poses is not None:
                pose_count = min(pose_count, limit_poses)
            for pose_idx in range(pose_count):
                self.samples.append((person_idx, pose_idx, local_idx, subject_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        person_idx, pose_idx, label_idx, subject_id = self.samples[index]
        image = self.data[person_idx, pose_idx].astype(np.float32)

        if self.mode == "pressure":
            image = np.clip(image, 0.0, 100.0) / 100.0
        else:
            image = np.clip(image, 0.0, 102.0) / 102.0

        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image), label_idx