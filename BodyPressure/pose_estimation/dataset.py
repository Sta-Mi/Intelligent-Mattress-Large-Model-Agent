"""Streaming reader for the public, password-free BodyPressureSD pickles."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_pickle(path):
    with open(path, "rb") as stream:
        try:
            return pickle.load(stream)
        except UnicodeDecodeError:  # BodyPressureSD was serialized by Python 2.
            stream.seek(0)
            return pickle.load(stream, encoding="latin1")


class BodyPressureSDPoseDataset(Dataset):
    """Pressure maps and 24 metric SMPL joints from BodyPressureSD.

    ``file_list`` makes subject/range-disjoint splits explicit. Each worker only
    keeps one source pickle cached, avoiding the original loader's >8 GB RAM use.
    """

    def __init__(self, root, file_list, pressure_clip=100.0):
        self.root = Path(root)
        self.files = [self.root / "synth" / name for name in file_list]
        if not self.files:
            raise ValueError("file_list cannot be empty")
        self.pressure_clip = float(pressure_clip)
        self.index = []
        for file_index, path in enumerate(self.files):
            if not path.is_file():
                raise FileNotFoundError(path)
            data = _load_pickle(path)
            self.index.extend((file_index, i) for i in range(len(data["images"])))
        self._cached_file_index = None
        self._cached_data = None

    def __len__(self):
        return len(self.index)

    def _data(self, file_index):
        if file_index != self._cached_file_index:
            self._cached_data = _load_pickle(self.files[file_index])
            self._cached_file_index = file_index
        return self._cached_data

    def __getitem__(self, item):
        file_index, sample_index = self.index[item]
        data = self._data(file_index)
        pressure = np.asarray(data["images"][sample_index], dtype=np.float32).reshape(64, 27)
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=self.pressure_clip, neginf=0.0)
        pressure = np.clip(pressure, 0.0, self.pressure_clip) / self.pressure_clip
        joints = np.asarray(data["markers_xyz_m"][sample_index], dtype=np.float32)
        joints = joints.reshape(-1, 3)[:24]
        return {
            "pressure": torch.from_numpy(pressure.copy()).unsqueeze(0),
            "joints": torch.from_numpy(joints.copy()),
            "sample_id": f"{self.files[file_index].stem}:{sample_index}",
        }
