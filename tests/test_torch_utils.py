"""
Tests for shared PyTorch utilities (device selection policy).
"""

import torch

from ethnicolr.torch_utils import select_inference_device


class TestDevicePolicy:
    def test_env_override_cpu(self, monkeypatch):
        monkeypatch.setenv("ETHNICOLR_DEVICE", "cpu")
        assert select_inference_device() == torch.device("cpu")

    def test_env_override_mps(self, monkeypatch):
        monkeypatch.setenv("ETHNICOLR_DEVICE", "mps")
        assert select_inference_device() == torch.device("mps")

    def test_default_never_selects_mps(self, monkeypatch):
        monkeypatch.delenv("ETHNICOLR_DEVICE", raising=False)
        assert select_inference_device().type in ("cpu", "cuda")
