# Tests for FGSM attack implementation.
import torch

from vajra.attacks.fgsm import attack_success_rate, fgsm_attack
from vajra.models.transformer import TransformerNIDS


def _make_model() -> TransformerNIDS:
    torch.manual_seed(0)
    m = TransformerNIDS(n_features=77, d_model=32, n_heads=4, n_layers=2)
    m.eval()
    return m


def test_perturbation_bounded_by_epsilon():
    """Every element of x_adv - x must have magnitude <= epsilon."""
    model = _make_model()
    x = torch.rand(8, 77)   # already in [0, 1]
    y = torch.randint(0, 2, (8,))
    for eps in [0.01, 0.05, 0.1]:
        result = fgsm_attack(model, x, y, epsilon=eps)
        max_pert = (result.adversarial - x).abs().max().item()
        assert max_pert <= eps + 1e-6, f"eps={eps} produced max_pert={max_pert}"


def test_adversarial_stays_in_valid_range():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    result = fgsm_attack(model, x, y, epsilon=0.1)
    assert result.adversarial.min() >= 0.0
    assert result.adversarial.max() <= 1.0


def test_each_element_moves_by_exactly_epsilon_when_not_clipped():
    """FGSM uses sign(grad); interior values should move by exactly +/- epsilon."""
    model = _make_model()
    x = torch.full((4, 77), 0.5)   # far from boundaries, no clipping
    y = torch.tensor([0, 1, 0, 1])
    eps = 0.1
    result = fgsm_attack(model, x, y, epsilon=eps)
    delta = (result.adversarial - x).abs()
    # Each element should be 0 (zero gradient) or eps
    assert torch.all((delta == 0) | (torch.abs(delta - eps) < 1e-6))


def test_zero_epsilon_is_noop():
    model = _make_model()
    x = torch.rand(4, 77)
    y = torch.randint(0, 2, (4,))
    result = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(result.adversarial, x)


def test_shapes_preserved():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    result = fgsm_attack(model, x, y, epsilon=0.05)
    assert result.adversarial.shape == x.shape
    assert result.perturbation.shape == x.shape
    assert result.clean_logits.shape == (8, 2)
    assert result.adversarial_logits.shape == (8, 2)


def test_attack_success_rate_logic():
    """Sanity check the ASR helper arithmetic."""
    clean = torch.tensor([0.9, 0.9, 0.9, 0.9])   # all above threshold -> all detected
    adv = torch.tensor([0.2, 0.2, 0.9, 0.9])     # first two flipped, last two not
    labels = torch.tensor([1, 1, 1, 1])
    asr = attack_success_rate(clean, adv, threshold=0.88, true_labels=labels)
    assert abs(asr - 0.5) < 1e-6   # 2 of 4 flipped