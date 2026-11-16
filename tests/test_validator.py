# Unit tests for InputValidator (Blueprint Section 5.3)
import pytest
import torch
import numpy as np

from vajra.defenses.input_validator import InputValidator, ValidationResult


def test_validator_accepts_clean_vector():
    val = InputValidator(clip=True, reject_nan=True)
    x = torch.tensor([0.1, 0.5, 0.9, 0.0, 1.0], dtype=torch.float32)
    res = val.validate(x)
    assert res.is_valid
    assert not res.clipped
    assert res.rejection_reason is None
    assert torch.allclose(res.sanitized_tensor, x.unsqueeze(0))


def test_validator_rejects_nan():
    val = InputValidator(reject_nan=True)
    x = torch.tensor([0.2, float("nan"), 0.8])
    res = val.validate(x)
    assert not res.is_valid
    assert "NaN" in res.rejection_reason


def test_validator_rejects_inf():
    val = InputValidator(reject_nan=True)
    x = torch.tensor([0.2, float("inf"), 0.8])
    res = val.validate(x)
    assert not res.is_valid
    assert "Inf" in res.rejection_reason


def test_validator_clips_out_of_bounds():
    names = ["port", "fwd_len", "iat"]
    val = InputValidator(clip=True, feature_names=names, clip_min=0.0, clip_max=1.0)
    x = torch.tensor([-0.15, 1.25, 0.5])
    res = val.validate(x)
    assert res.is_valid
    assert res.clipped
    assert res.out_of_bounds_count == 2
    assert "port" in res.clipped_features
    assert "fwd_len" in res.clipped_features
    assert res.sanitized_tensor[0, 0].item() == 0.0
    assert res.sanitized_tensor[0, 1].item() == 1.0

