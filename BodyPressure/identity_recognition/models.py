import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

try:
    import timm
except Exception:
    timm = None

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

CONVNEXT_V2_BASE_IN1K = Path(
    "/home/shnh/DATA/zjy/BodyMAP_identity_pretrained/convnextv2_base.fcmae_ft_in22k_in1k.safetensors"
)
CONVNEXT_V2_BASE_22K = Path(
    "/home/shnh/DATA/zjy/BodyMAP_identity_pretrained/convnextv2_base_22k_224_ema.pt"
)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18Identity(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        if x.shape[-2] < 224 or x.shape[-1] < 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class TimmIdentity(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is not installed. Run: pip install timm")

        local_state = None
        local_source = None
        if model_name == "convnextv2_base":
            if CONVNEXT_V2_BASE_IN1K.exists():
                local_source = CONVNEXT_V2_BASE_IN1K
                if load_safetensors is None:
                    raise RuntimeError("safetensors is required to load the ConvNeXt V2 checkpoint")
                local_state = load_safetensors(local_source)
            elif CONVNEXT_V2_BASE_22K.exists():
                local_source = CONVNEXT_V2_BASE_22K
                checkpoint = torch.load(local_source, map_location="cpu")
                local_state = checkpoint.get("model", checkpoint)

        if local_state is not None:
            local_state = {
                k: v for k, v in local_state.items()
                if not k.startswith("head.fc.")
            }
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )
            missing, unexpected = self.backbone.load_state_dict(
                local_state,
                strict=False,
            )
            print(
                f"Loaded local ConvNeXt V2 checkpoint from {local_source}. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
            )

    def forward(self, x):
        if x.shape[-2] < 224 or x.shape[-1] < 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def build_model(name: str, num_classes: int):
    if name == "small_cnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        return ResNet18Identity(num_classes)
    if name == "convnextv2_base":
        return TimmIdentity("convnextv2_base", num_classes)
    if name.startswith("timm:"):
        return TimmIdentity(name.split(":", 1)[1], num_classes)
    raise ValueError(f"Unknown model {name}")