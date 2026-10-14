import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EEGNetDEClassifier(nn.Module):
    """
    EEGNet-Inspired Convolutional Neural Network for SEED-VII Pre-Extracted EEG DE Features.
    
    Input shape: (batch_size, 1, 5, 62)
      - 1 input channel
      - 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
      - 62 EEG electrode channels
    """
    def __init__(
        self,
        num_classes: int = 7,
        num_bands: int = 5,
        num_channels: int = 62,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        dropout_rate: float = 0.25
    ):
        super(EEGNetDEClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.num_channels = num_channels
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate

        self.input_bn = nn.BatchNorm2d(1)

        # 1. Spatial Conv across channels (1, 62)
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, num_channels),
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. Depthwise Conv across frequency bands (5, 1)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(num_bands, 1),
            stride=1,
            padding=0,
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 3. Separable Conv (1, 1)
        self.separable_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier Linear Layer (7 classes or 3 classes or 1 regression output)
        self.classifier = nn.Linear(F2, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.view(-1, 1, self.num_bands, self.num_channels)

        out = self.input_bn(x)
        out = self.spatial_conv(out)
        out = self.bn1(out)
        out = F.elu(out)

        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = F.elu(out)
        out = self.dropout1(out)

        out = self.separable_conv(out)
        out = self.bn3(out)
        out = F.elu(out)
        out = self.dropout2(out)

        return out.view(out.size(0), -1)  # (batch_size, F2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class MultimodalEEGNetClassifier(nn.Module):
    """
    Dual-branch Multimodal Fusion Architecture:
      - EEG Branch: EEGNetDEClassifier feature extractor (32-dim)
      - EYE Branch: 2-layer MLP for 33 eye feature dimensions (32-dim)
      - Fusion Layer: Linear classifier on combined 64-dim representation
    """
    def __init__(
        self,
        num_classes: int = 7,
        num_eye_features: int = 33,
        dropout_rate: float = 0.25
    ):
        super(MultimodalEEGNetClassifier, self).__init__()

        self.eeg_branch = EEGNetDEClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

        self.eye_branch = nn.Sequential(
            nn.BatchNorm1d(num_eye_features),
            nn.Linear(num_eye_features, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )

        self.fusion_classifier = nn.Linear(32 + 32, num_classes)

    def forward(self, x_eeg: torch.Tensor, x_eye: Optional[torch.Tensor] = None) -> torch.Tensor:
        eeg_feat = self.eeg_branch.extract_features(x_eeg)

        if x_eye is not None:
            eye_feat = self.eye_branch(x_eye)
            joint_feat = torch.cat([eeg_feat, eye_feat], dim=1)
            return self.fusion_classifier(joint_feat)
        else:
            return self.eeg_branch.classifier(eeg_feat)


def get_device() -> torch.device:
    """Detect CUDA GPU, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
