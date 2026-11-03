# Tests for FGSM, PGD, APGD, and transfer attacks.
import torch

from vajra.attacks.autoattack import apgd_ce_attack
from vajra.attacks.fgsm import attack_success_rate, fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.attacks.transfer import SurrogateCNN
from vajra.models.transformer import TransformerNIDS


def _make_model() -> TransformerNIDS:
    torch.manual_seed(0)
    m = TransformerNIDS(n_features=77, d_model=32, n_heads=4, n_layers=2)
    m.eval()
    return m


# --- FGSM tests (unchanged) ---

def test_fgsm_perturbation_bounded():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    for eps in [0.01, 0.05, 0.1]:
        r = fgsm_attack(model, x, y, epsilon=eps)
        assert (r.adversarial - x).abs().max().item() <= eps + 1e-6


def test_fgsm_valid_range():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    r = fgsm_attack(model, x, y, epsilon=0.1)
    assert r.adversarial.min() >= 0.0
    assert r.adversarial.max() <= 1.0


def test_fgsm_zero_eps_is_noop():
    model = _make_model()
    x = torch.rand(4, 77)
    y = torch.randint(0, 2, (4,))
    r = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(r.adversarial, x)


def test_asr_helper():
    clean = torch.tensor([0.9, 0.9, 0.9, 0.9])
    adv = torch.tensor([0.2, 0.2, 0.9, 0.9])
    labels = torch.tensor([1, 1, 1, 1])
    assert abs(attack_success_rate(clean, adv, 0.88, labels) - 0.5) < 1e-6


# --- PGD tests ---

def test_pgd_perturbation_bounded():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    for eps in [0.01, 0.05, 0.1]:
        r = pgd_attack(model, x, y, epsilon=eps, num_iter=5)
        assert (r.adversarial - x).abs().max().item() <= eps + 1e-6


def test_pgd_valid_range():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    r = pgd_attack(model, x, y, epsilon=0.1, num_iter=10)
    assert r.adversarial.min() >= 0.0
    assert r.adversarial.max() <= 1.0


def test_pgd_shapes():
    model = _make_model()
    x = torch.rand(4, 77)
    y = torch.randint(0, 2, (4,))
    r = pgd_attack(model, x, y, epsilon=0.05, num_iter=5)
    assert r.adversarial.shape == (4, 77)
    assert r.clean_logits.shape == (4, 2)
    assert r.adversarial_logits.shape == (4, 2)


# --- APGD tests ---

def test_apgd_perturbation_bounded():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    r = apgd_ce_attack(model, x, y, epsilon=0.05, num_iter=20)
    assert (r.adversarial - x).abs().max().item() <= 0.05 + 1e-6


def test_apgd_valid_range():
    model = _make_model()
    x = torch.rand(8, 77)
    y = torch.randint(0, 2, (8,))
    r = apgd_ce_attack(model, x, y, epsilon=0.1, num_iter=20)
    assert r.adversarial.min() >= 0.0
    assert r.adversarial.max() <= 1.0


def test_apgd_strength_at_least_as_good_as_fgsm():
    """APGD should achieve >= FGSM ASR on average (it is strictly stronger)."""
    torch.manual_seed(1)
    model = _make_model()
    x = torch.rand(32, 77)
    y = torch.randint(0, 2, (32,))

    fgsm_r = fgsm_attack(model, x, y, epsilon=0.05)
    apgd_r = apgd_ce_attack(model, x, y, epsilon=0.05, num_iter=50)

    fgsm_loss = torch.nn.CrossEntropyLoss(reduction="none")(
        fgsm_r.adversarial_logits, y).mean()
    apgd_loss = torch.nn.CrossEntropyLoss(reduction="none")(
        apgd_r.adversarial_logits, y).mean()

    # APGD maximizes loss, so its adversarial loss should be >= FGSM's
    assert apgd_loss >= fgsm_loss - 1e-3


# --- Surrogate CNN tests ---

def test_surrogate_forward_shape():
    model = SurrogateCNN(n_features=77)
    model.eval()
    out = model(torch.rand(4, 77))
    assert out.shape == (4, 2)