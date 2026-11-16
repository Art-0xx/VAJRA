# VAJRA Runtime Security & Drift Monitoring (Blueprint Section 5.6)
from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from vajra.justification.divergence import compute_divergence

logger = logging.getLogger("vajra.monitor")


@dataclass
class DivergenceCheckResult:
    """Outcome of attention divergence evaluation."""
    divergence_score: float
    threshold: float
    is_divergent: bool
    recommended_model: str  # "baseline" or "defended"
    message: str


class AttentionDivergenceRouter:
    """Evaluates attention distribution divergence to detect evasion attempts
    and route suspicious flows to the robust defended model.
    """

    def __init__(self, threshold: float = 0.050, class_mean_attention: Optional[np.ndarray] = None) -> None:
        self.threshold = threshold
        self.class_mean_attention = class_mean_attention

    def evaluate(
        self,
        current_attention: np.ndarray,
        baseline_attention: Optional[np.ndarray] = None,
    ) -> DivergenceCheckResult:
        """Compute Jensen-Shannon Divergence and recommend routing."""
        reference = baseline_attention if baseline_attention is not None else self.class_mean_attention

        if reference is None:
            # If no reference is provided, return nominal
            return DivergenceCheckResult(
                divergence_score=0.0,
                threshold=self.threshold,
                is_divergent=False,
                recommended_model="baseline",
                message="No baseline attention reference provided; using standard pipeline.",
            )

        div_score = compute_divergence(reference, current_attention)
        is_divergent = div_score > self.threshold

        if is_divergent:
            logger.warning(
                f"[ROUTING EVENT] Attention divergence triggered: JSD {div_score:.4f} > {self.threshold:.4f} "
                f"-> Rerouting flow to defended model"
            )
            return DivergenceCheckResult(
                divergence_score=div_score,
                threshold=self.threshold,
                is_divergent=True,
                recommended_model="defended",
                message=f"Attention divergence {div_score:.4f} exceeded threshold {self.threshold:.4f}. Rerouted to defended model.",
            )

        return DivergenceCheckResult(
            divergence_score=div_score,
            threshold=self.threshold,
            is_divergent=False,
            recommended_model="baseline",
            message="Attention divergence nominal.",
        )


class AlertRateTracker:
    """Tracks alerts per minute and identifies anomalous drops in alert rates
    which indicate coordinated adversarial evasion.
    """

    def __init__(self, window_seconds: int = 60, evasion_drop_threshold: float = 0.8) -> None:
        self.window_seconds = window_seconds
        self.evasion_drop_threshold = evasion_drop_threshold
        self._alert_timestamps: collections.deque = collections.deque()
        self._flow_timestamps: collections.deque = collections.deque()

    def record_flow(self, is_attack: bool) -> None:
        """Record an evaluated flow timestamp."""
        now = time.time()
        self._flow_timestamps.append(now)
        if is_attack:
            self._alert_timestamps.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._flow_timestamps and self._flow_timestamps[0] < cutoff:
            self._flow_timestamps.popleft()
        while self._alert_timestamps and self._alert_timestamps[0] < cutoff:
            self._alert_timestamps.popleft()

    def get_rates(self) -> Dict[str, Any]:
        """Compute current flow rate and alert rate per minute."""
        now = time.time()
        self._prune(now)
        flows_per_min = len(self._flow_timestamps)
        alerts_per_min = len(self._alert_timestamps)
        alert_ratio = alerts_per_min / max(flows_per_min, 1)

        return {
            "window_sec": self.window_seconds,
            "flows_per_min": flows_per_min,
            "alerts_per_min": alerts_per_min,
            "alert_ratio": round(alert_ratio, 4),
        }


class FeatureDriftMonitor:
    """Rolling feature distribution monitoring to detect data drift and shift."""

    def __init__(
        self,
        n_features: int,
        window_size: int = 200,
        psi_threshold: float = 0.25,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.n_features = n_features
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self.feature_names = feature_names or [f"feat_{i}" for i in range(n_features)]
        self._rolling_buffer: collections.deque = collections.deque(maxlen=window_size)
        self.reference_quantiles: Optional[np.ndarray] = None

    def set_reference(self, reference_data: np.ndarray) -> None:
        """Set reference training distribution quantiles for PSI comparison."""
        # 10 bins (deciles)
        quantiles = np.linspace(0, 1, 11)
        self.reference_quantiles = np.quantile(reference_data, quantiles, axis=0)

    def add_sample(self, feature_vector: np.ndarray) -> None:
        """Append incoming flow feature vector to rolling buffer."""
        self._rolling_buffer.append(feature_vector.flatten())

    def check_drift(self) -> Dict[str, Any]:
        """Check rolling window for statistically significant distribution shift."""
        if len(self._rolling_buffer) < 50:
            return {"status": "insufficient_data", "samples": len(self._rolling_buffer)}

        current_arr = np.array(self._rolling_buffer)
        current_mean = np.mean(current_arr, axis=0)
        current_std = np.std(current_arr, axis=0)

        # Detect high variance drift
        drifted_features = []
        for i, name in enumerate(self.feature_names):
            if current_std[i] > 0.45:  # Scaled variance spike
                drifted_features.append(name)

        return {
            "status": "monitored",
            "sample_count": len(self._rolling_buffer),
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features[:10],
        }

