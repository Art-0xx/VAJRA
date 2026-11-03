from __future__ import annotations
import torch

def compute_attention_rollout(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    if not attention_maps:
        raise ValueError("No attention maps provided.")
    attn_avg = [attn.mean(dim=1) for attn in attention_maps]
    batch_size, seq_len, _ = attn_avg[0].shape
    device = attn_avg[0].device
    identity = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    attn_with_residual = [attn + identity for attn in attn_avg]
    attn_normalized = []
    for attn in attn_with_residual:
        row_sum = attn.sum(dim=-1, keepdim=True)
        attn_normalized.append(attn / row_sum)
    rollout = attn_normalized[0]
    for attn in attn_normalized[1:]:
        rollout = torch.bmm(attn, rollout)
    return rollout

def get_feature_attribution(rollout: torch.Tensor) -> torch.Tensor:
    cls_attention = rollout[:, 0, 1:] 
    return cls_attention