import pickle

import numpy as np
import torch

from BodyPressure.pose_estimation.dataset import BodyPressureSDPoseDataset
from BodyPressure.pose_estimation.model import PressurePoseTransformer, mpjpe, pose_errors


def test_model_shape_and_gradient():
    model = PressurePoseTransformer(embed_dim=32, depth=1, num_heads=4)
    pressure = torch.rand(2, 1, 64, 27)
    prediction = model(pressure)
    assert prediction.shape == (2, 24, 3)
    mpjpe(prediction, torch.zeros_like(prediction)).backward()
    assert model.patch_embed.weight.grad is not None


def test_dataset_reads_python_pickle(tmp_path):
    synth = tmp_path / "synth"
    synth.mkdir()
    payload = {"images": [np.arange(64 * 27, dtype=np.float32)],
               "body_volume": [0.07],
               "markers_xyz_m": [np.arange(72, dtype=np.float32)]}
    with (synth / "sample.p").open("wb") as stream:
        pickle.dump(payload, stream)
    sample = BodyPressureSDPoseDataset(tmp_path, ["sample.p"])[0]
    assert sample["pressure"].shape == (1, 64, 27)
    assert sample["pressure"].max() == 1
    assert sample["joints"].shape == (24, 3)


def test_data_root_accepts_synth_directory(tmp_path):
    synth = tmp_path / "data_BP" / "synth"
    synth.mkdir(parents=True)
    assert BodyPressureSDPoseDataset.resolve_data_root(synth) == synth.parent


def test_pose_errors_remove_global_translation():
    target = torch.rand(2, 24, 3)
    prediction = target + torch.tensor([[[0.1, -0.2, 0.3]]])
    errors = pose_errors(prediction, target)
    assert errors["mpjpe"] > 0
    assert torch.allclose(errors["root_mpjpe"], torch.tensor(0.0), atol=1e-6)
    assert errors["per_joint"].shape == (24,)
