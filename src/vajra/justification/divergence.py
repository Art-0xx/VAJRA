# Attention Divergence Score (ACS) for VAJRA.
#
# Computes the Jensen-Shannon Divergence between two attention rollout
# distributions. Used as a run-time adversarial detection signal.
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


def compute_divergence(clean_attention: np.ndarray, adv_attention: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence between two 1D probability arrays.
    
    Args:
        clean_attention: Shape (n_features,). Sums to 1.
        adv_attention: Shape (n_features,). Sums to 1.
        
    Returns:
        JSD score between 0 and 1.
    """
    # JSD expects probabilities that sum to 1. If they don't (due to
    # floating point precision), normalize them.
    c_norm = clean_attention / clean_attention.sum()
    a_norm = adv_attention / adv_attention.sum()
    
    # jensenshannon returns the square root of the divergence.
    # We square it to get the actual divergence score.
    return float(jensenshannon(c_norm, a_norm) ** 2)


def compute_batch_divergence(clean_attn_batch: torch.Tensor, adv_attn_batch: torch.Tensor) -> list[float]:
    """Compute JSD for a batch of attention vectors."""
    c_np = clean_attn_batch.cpu().numpy()
    a_np = adv_attn_batch.cpu().numpy()
    return [compute_divergence(c, a) for c, a in zip(c_np, a_np)]