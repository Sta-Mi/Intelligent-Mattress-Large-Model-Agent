"""Streaming reader for the public, password-free BodyPressureSD pickles."""

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
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

    FEMALE_MASS_NORMALIZER = (62.5, 0.06878933937454557)
    MALE_MASS_NORMALIZER = (78.4, 0.0828308574658067)

    def __init__(self, root, file_list, pressure_clip=100.0):
        self.root = self.resolve_data_root(root)
        self.files = [self.root / "synth" / name for name in file_list]
        if not self.files:
            raise ValueError("file_list cannot be empty")
        self.pressure_clip = float(pressure_clip)
        self.index = []
        self.file_ranges = []
        for file_index, path in enumerate(self.files):
            if not path.is_file():
                raise FileNotFoundError(path)
            data = _load_pickle(path)
            start = len(self.index)
            self.index.extend((file_index, i) for i in range(len(data["images"])))
            self.file_ranges.append(range(start, len(self.index)))
        self._cached_file_index = None
        self._cached_data = None

    @staticmethod
    def resolve_data_root(root):
        """Accept the repository, ``BodyPressure``, ``data_BP``, or synth dir."""
        root = Path(root).expanduser().resolve()
        candidates = (root, root / "data_BP", root / "BodyPressure" / "data_BP")
        if root.name == "synth":
            return root.parent
        for candidate in candidates:
            if (candidate / "synth").is_dir():
                return candidate
        raise FileNotFoundError(
            f"cannot find a synth directory below {root}; pass data_BP or synth"
        )

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
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
        pressure = gaussian_filter(pressure, sigma=0.5)
        # BodyPressureSD stores simulator force intensity, not mmHg. Match the
        # official BodyMAP conversion using body volume and gender-specific mass.
        normalizer = (self.FEMALE_MASS_NORMALIZER if "_f_" in self.files[file_index].name
                      else self.MALE_MASS_NORMALIZER)
        mass = float(data["body_volume"][sample_index]) * normalizer[0] / normalizer[1]
        force_sum = float(pressure.sum())
        if force_sum > 0:
            pressure *= (mass * 9.81) / (force_sum * 0.0264 * 0.0286) / 133.322
        pressure = np.clip(pressure, 0.0, self.pressure_clip) / self.pressure_clip
        joints = np.asarray(data["markers_xyz_m"][sample_index], dtype=np.float32)
        joints = joints.reshape(-1, 3)[:24].copy()
        # Same synthetic-bed coordinate transform used by BodyMAP/BPDataset.py.
        joints -= np.asarray((0.298, 0.321, 0.075), dtype=np.float32)
        return {
            "pressure": torch.from_numpy(pressure.copy()).unsqueeze(0),
            "joints": torch.from_numpy(joints.copy()),
            "sample_id": f"{self.files[file_index].stem}:{sample_index}",
        }
