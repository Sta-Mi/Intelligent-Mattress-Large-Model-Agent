"""A compact ViT-style pressure-to-3D-joints baseline.

The model deliberately has no RGB/depth branch, SMPL dependency, or SLP data
dependency.  Joint queries cross-attend to pressure taxel tokens, which retains
spatial detail better than regressing every joint from one pooled feature.
"""

import torch
from torch import nn


class PressurePoseTransformer(nn.Module):
    """Regress metric 3D joint locations from a ``[B, 1, H, W]`` pressure map."""

    def __init__(self, num_joints=24, embed_dim=192, depth=6, num_heads=6,
                 patch_size=4, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, nn.LayerNorm(embed_dim))
        self.joint_queries = nn.Parameter(torch.empty(1, num_joints, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.joint_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 3)
        )
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

    @staticmethod
    def _sincos_position(rows, cols, dim, device, dtype):
        if dim % 4:
            raise ValueError("embed_dim must be divisible by 4")
        y, x = torch.meshgrid(
            torch.arange(rows, device=device, dtype=dtype),
            torch.arange(cols, device=device, dtype=dtype), indexing="ij",
        )
        omega = torch.arange(dim // 4, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
        y, x = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)[None]

    def forward(self, pressure):
        if pressure.ndim != 4 or pressure.shape[1] != 1:
            raise ValueError("pressure must have shape [batch, 1, height, width]")
        features = self.patch_embed(pressure)
        rows, cols = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._sincos_position(
            rows, cols, tokens.shape[-1], tokens.device, tokens.dtype
        )
        tokens = self.encoder(tokens)
        queries = self.joint_queries.expand(pressure.shape[0], -1, -1)
        queries, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        # Coordinates are in metres in the pressure-mat coordinate frame.
        return self.joint_head(self.query_norm(queries))


def mpjpe(prediction, target):
    """Mean per-joint position error in metres."""
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()
