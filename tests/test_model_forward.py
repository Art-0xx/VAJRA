# Shape and attention tests for TransformerNIDS.
import torch

from vajra.models.transformer import TransformerNIDS


def test_forward_shape():
    model = TransformerNIDS(n_features=77, d_model=64, n_heads=4, n_layers=2)
    model.eval()
    x = torch.randn(8, 77)
    logits = model(x)
    assert logits.shape == (8, 2)


def test_attention_maps_stored():
    model = TransformerNIDS(
        n_features=20, d_model=32, n_heads=4, n_layers=3, store_attention=True
    )
    model.eval()
    x = torch.randn(4, 20)
    _ = model(x)
    maps = model.get_attention_maps()
    assert len(maps) == 3
    for m in maps:
        # (batch, n_heads, seq_len, seq_len); seq_len = n_features + 1 (CLS)
        assert m.shape == (4, 4, 21, 21)
        # Attention rows should sum to ~1.
        row_sums = m.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_no_attention_without_flag():
    model = TransformerNIDS(n_features=10, d_model=32, n_heads=4, n_layers=2)
    model.eval()
    _ = model(torch.randn(2, 10))
    assert model.attention_maps == []


def test_deterministic_in_eval():
    torch.manual_seed(0)
    model = TransformerNIDS(n_features=15, d_model=32, n_heads=4, n_layers=2)
    model.eval()
    x = torch.randn(3, 15)
    with torch.no_grad():
        logits1 = model(x)
        logits2 = model(x)
    assert torch.allclose(logits1, logits2)