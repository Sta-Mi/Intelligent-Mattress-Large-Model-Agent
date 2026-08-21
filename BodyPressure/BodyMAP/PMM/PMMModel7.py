"""Pressure-only transformer mesh and per-vertex pressure model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import X_BUMP, Y_BUMP
from MeshEstimator import MeshEstimator
from PMEModel16 import PMEstimator


class PMMModel(nn.Module):
    """ViT pressure encoder with the official BodyMAP SMPL and PME16 decoders."""

    def __init__(self, model_fn, feature_size, out_size, vertex_size, batch_size,
                 modality, use_contact=False, indexing_mode=9):
        super().__init__()
        if modality != "pressure":
            raise ValueError("PMM7 is pressure-only; set modality='pressure'")
        if model_fn != "PME16":
            raise ValueError("PMM7 currently supports PME16 only")
        self.modality = modality
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(1, 512, self.patch_size, self.patch_size)
        layer = nn.TransformerEncoderLayer(
            512, 8, 2048, dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 6, nn.LayerNorm(512))
        self.smpl_head = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, out_size),
        )
        self.mesh_model = MeshEstimator(batch_size)
        self.pme = PMEstimator(modality, vertex_size, indexing_mode, use_contact)

    @staticmethod
    def _position(rows, cols, dim, device, dtype):
        y, x = torch.meshgrid(torch.arange(rows, device=device, dtype=dtype),
                              torch.arange(cols, device=device, dtype=dtype), indexing="ij")
        omega = 1.0 / (10000 ** (torch.arange(dim // 4, device=device, dtype=dtype)
                                  / (dim // 4 - 1)))
        x, y = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)[None]

    def _forward_smpl(self, pressure_map):
        image_features = pressure_map
        feature_map = self.patch_embed(pressure_map)
        rows, cols = feature_map.shape[-2:]
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self._position(
            rows, cols, tokens.shape[-1], tokens.device, tokens.dtype))
        encoder_features = tokens.transpose(1, 2).reshape(-1, 512, rows, cols)
        global_features = tokens.mean(dim=1)
        return self.smpl_head(global_features), encoder_features, image_features

    @staticmethod
    def _grid(verts):
        y = 64 - (verts[:, :, 0:1] - Y_BUMP) / (0.0286 * 1.04)
        x = (verts[:, :, 1:2] - X_BUMP) / 0.0286
        return torch.cat((2 * (x.clamp(0, 26) / 26) - 1,
                          2 * (y.clamp(0, 63) / 63) - 1), dim=-1).unsqueeze(-2)

    def _pressure_features(self, verts, image_features, encoder_features):
        grid = self._grid(verts)
        start = F.grid_sample(image_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        encoded = F.grid_sample(encoder_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        global_features = encoder_features.mean(dim=(2, 3))
        above_mat = verts[:, :, -1] < 0
        start[above_mat], encoded[above_mat] = 0.0, 0.0
        return start, encoded, global_features

    def forward(self, depth_map, pressure_map, gender):
        smpl_pred, encoded, image = self._forward_smpl(pressure_map)
        mesh = self.mesh_model.infer(smpl_pred, gender)
        start, vertex_encoded, global_features = self._pressure_features(
            mesh["out_verts"], image, encoded)
        pmap, contact = self.pme(mesh["out_verts"], start, vertex_encoded, None,
                                 global_features)
        return mesh, pmap, contact, smpl_pred

    def infer(self, depth_map, pressure_map, gender):
        mesh, pmap, contact, smpl = self.forward(depth_map, pressure_map, gender)
        if contact is not None:
            pmap = pmap * contact.argmax(dim=1)
        return mesh, pmap, contact, smpl

    def mesh_infer_gt(self, x_gt, gender):
        with torch.no_grad():
            return self.mesh_model.infer(x_gt, gender, is_gt=True)
