import torch
import torch.nn as nn
import torchvision.models as models

from PMMModel5 import PMMModel as EarlyFusionPMM


def make_encoder():
    encoder = models.__dict__["resnet18"](pretrained=False)
    encoder.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    return nn.Sequential(*list(encoder.children()))[:-2]


class PMMModel(EarlyFusionPMM):
    """BodyMAP-PointNet with independent depth/pressure encoders and gated fusion."""

    def __init__(
        self,
        model_fn,
        feature_size,
        out_size,
        vertex_size,
        batch_size,
        modality,
        use_contact=False,
        indexing_mode=0,
    ):
        if modality != "both":
            raise ValueError("PMM6 is a dual-modality model and requires modality='both'")
        super().__init__(
            model_fn,
            feature_size,
            out_size,
            vertex_size,
            batch_size,
            modality,
            use_contact,
            indexing_mode,
        )
        del self.encoder
        self.depth_encoder = make_encoder()
        self.pressure_encoder = make_encoder()
        self.modality_gate = nn.Linear(1024, 2)
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.zeros_(self.modality_gate.bias)

    def _forward_smpl(self, depth_map, pressure_map):
        if pressure_map.shape[-2:] != depth_map.shape[-2:]:
            pressure_map = nn.functional.interpolate(
                pressure_map,
                size=depth_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        depth_features = self.depth_encoder(depth_map)
        pressure_features = self.pressure_encoder(pressure_map)
        depth_global = self.flatten(self.global_pool(depth_features))
        pressure_global = self.flatten(self.global_pool(pressure_features))
        modality_weights = torch.softmax(
            self.modality_gate(torch.cat((depth_global, pressure_global), dim=1)),
            dim=1,
        )
        depth_weight = modality_weights[:, 0].reshape(-1, 1, 1, 1)
        pressure_weight = modality_weights[:, 1].reshape(-1, 1, 1, 1)
        encoder_features = (
            depth_weight * depth_features + pressure_weight * pressure_features
        )
        local_features = self.global_pool(encoder_features)

        out = self.flatten(local_features)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        smpl_pred = self.fc3(out)
        image_features = torch.cat((depth_map, pressure_map), dim=1)
        return smpl_pred, local_features, encoder_features, image_features
