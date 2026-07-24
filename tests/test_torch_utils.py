"""
Tests for shared PyTorch utilities (device selection policy).
"""

import torch

from ethnicolr.pred_census_ln import CensusLnModel
from ethnicolr.torch_utils import get_device


class TestDevicePolicy:
    def test_env_override_cpu(self, monkeypatch):
        monkeypatch.setenv("ETHNICOLR_DEVICE", "cpu")
        assert get_device() == torch.device("cpu")

    def test_env_override_mps(self, monkeypatch):
        monkeypatch.setenv("ETHNICOLR_DEVICE", "mps")
        assert get_device() == torch.device("mps")

    def test_default_never_selects_mps(self, monkeypatch):
        monkeypatch.delenv("ETHNICOLR_DEVICE", raising=False)
        assert get_device().type in ("cpu", "cuda")

    def test_census_model_uses_shared_policy(self, monkeypatch):
        monkeypatch.setenv("ETHNICOLR_DEVICE", "cpu")
        monkeypatch.setattr(CensusLnModel, "_device", None)
        assert CensusLnModel.get_device() == torch.device("cpu")
