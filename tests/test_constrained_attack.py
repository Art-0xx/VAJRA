# Unit test for Constrained PGD Attack (Blueprint Section 7.2)
import pytest
import torch
import joblib

from vajra.config import load_config
from vajra.models.registry import get_model
from vajra.attacks.constrained_pgd import constrained_pgd_attack, build_controllable_mask


def test_constrained_pgd_masks_non_controllable_features():
    pre = joblib.load("artifacts/preprocessor.joblib")
    config = load_config("configs/model_transformer.yaml")
    device = torch.device("cpu")

    model = get_model(config, len(pre.feature_columns)).to(device)
    model.eval()

    mask = build_controllable_mask(pre.feature_columns, device=device)
    # Check that mask is non-empty but less than total features
    assert 0 < mask.sum().item() < len(pre.feature_columns)

    # Destination Port (index 0) must NOT be controllable
    dest_port_idx = pre.feature_columns.index("Destination Port")
    assert mask[dest_port_idx].item() == 0.0

    # Test single flow execution
    x = torch.zeros((1, len(pre.feature_columns)), dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.long)

    res = constrained_pgd_attack(
        model=model,
        x=x,
        y=y,
        epsilon=0.05,
        feature_names=pre.feature_columns,
        num_iter=3,
    )

    assert res.adversarial.shape == x.shape
    # Ensure non-controllable features were strictly NOT perturbed
    non_controllable_indices = torch.where(mask == 0.0)[0]
    for idx in non_controllable_indices:
        assert res.adversarial[0, idx].item() == x[0, idx].item()

