# VAJRA Input Validator - Defense & Sanity Layer (Blueprint Section 5.3)
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("vajra.validator")


@dataclass
class ValidationResult:
    """Outcome of feature vector validation."""
    is_valid: bool
    sanitized_tensor: torch.Tensor | None = None
    clipped: bool = False
    rejection_reason: str | None = None
    clipped_features: list[str] = field(default_factory=list)
    out_of_bounds_count: int = 0
    max_excursion: float = 0.0


class InputValidator:
    """Validates and sanitizes incoming scaled flow vectors before neural inference.
    
    Protects against numerical instability, out-of-distribution attacks,
    and invalid sensor payloads.
    """

    def __init__(
        self,
        clip: bool = True,
        reject_nan: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        excursion_tolerance: float = 0.5,
        feature_names: list[str] | None = None,
    ) -> None:
        self.clip = clip
        self.reject_nan = reject_nan
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.excursion_tolerance = excursion_tolerance
        self.feature_names = feature_names or []
        self.rejection_count = 0
        self.clipped_event_count = 0

    def validate(
        self,
        x: torch.Tensor | np.ndarray | list[float],
        device: torch.device | None = None,
    ) -> ValidationResult:
        """Validate, check ranges, and clip input tensor.
        
        Args:
            x: Raw or scaled tensor/array of shape (n_features,) or (batch_size, n_features).
            device: Optional target device for output tensor.
            
        Returns:
            ValidationResult with validity flag, rejection reason, and sanitized tensor.
        """
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except Exception as e:
                self.rejection_count += 1
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Failed to parse input as float tensor: {e}",
                )

        if device is not None:
            x = x.to(device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 1. Reject NaN or Inf
        if self.reject_nan:
            if torch.isnan(x).any():
                self.rejection_count += 1
                logger.warning("[SECURITY EVENT] Malformed payload rejected: contains NaN")
                return ValidationResult(
                    is_valid=False,
                    rejection_reason="Input vector contains NaN values",
                )
            if torch.isinf(x).any():
                self.rejection_count += 1
                logger.warning("[SECURITY EVENT] Malformed payload rejected: contains Inf")
                return ValidationResult(
                    is_valid=False,
                    rejection_reason="Input vector contains Inf values",
                )

        # 2. Check for values far outside [0, 1] (out-of-bounds excursion)
        lower_excursion = (self.clip_min - x).clamp(min=0.0)
        upper_excursion = (x - self.clip_max).clamp(min=0.0)
        max_excursion = float(torch.max(torch.cat([lower_excursion, upper_excursion])).item())

        out_of_bounds_mask = (x < self.clip_min) | (x > self.clip_max)
        out_of_bounds_count = int(out_of_bounds_mask.sum().item())

        clipped_features: list[str] = []
        if out_of_bounds_count > 0 and self.feature_names:
            feature_indices = torch.where(out_of_bounds_mask)[1].cpu().numpy().tolist()
            for idx in set(feature_indices):
                if idx < len(self.feature_names):
                    clipped_features.append(self.feature_names[idx])

        # If values exceed extreme excursion tolerance, flag/reject as adversarial perturbation
        if max_excursion > self.excursion_tolerance:
            logger.warning(
                f"[SECURITY EVENT] Out-of-bounds anomaly: max excursion {max_excursion:.3f} > {self.excursion_tolerance:.3f} "
                f"across {out_of_bounds_count} features"
            )

        # 3. Clip scaled features to [clip_min, clip_max]
        clipped = False
        if self.clip and out_of_bounds_count > 0:
            x_sanitized = torch.clamp(x, min=self.clip_min, max=self.clip_max)
            clipped = True
            self.clipped_event_count += 1
        else:
            x_sanitized = x.clone()

        return ValidationResult(
            is_valid=True,
            sanitized_tensor=x_sanitized,
            clipped=clipped,
            clipped_features=clipped_features,
            out_of_bounds_count=out_of_bounds_count,
            max_excursion=max_excursion,
        )

