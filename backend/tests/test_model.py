import pytest
import torch
from backend.app.models.eegnet_de import EEGNetDEClassifier, MultimodalEEGNetClassifier, get_device

def test_model_forward_pass_7_classes():
    batch_size = 8
    model = EEGNetDEClassifier(num_classes=7, num_bands=5, num_channels=62)
    x = torch.randn(batch_size, 1, 5, 62)
    logits = model(x)

    assert logits.shape == (batch_size, 7)
    assert not torch.isnan(logits).any()

def test_multimodal_model_forward_pass():
    batch_size = 4
    model = MultimodalEEGNetClassifier(num_classes=7, num_eye_features=33)
    x_eeg = torch.randn(batch_size, 1, 5, 62)
    x_eye = torch.randn(batch_size, 33)

    logits = model(x_eeg, x_eye)

    assert logits.shape == (batch_size, 7)
    assert not torch.isnan(logits).any()

def test_device_detection():
    dev = get_device()
    assert isinstance(dev, torch.device)
