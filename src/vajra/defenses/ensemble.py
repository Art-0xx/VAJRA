# VAJRA Multi-Model Ensemble Defense & Consensus Layer (Blueprint Section 8)
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from vajra.justification.divergence import compute_divergence
from vajra.justification.rollout import compute_attention_rollout, get_feature_attribution

logger = logging.getLogger("vajra.ensemble")


@dataclass
class EnsembleDecision:
    """Consensus decision from baseline and defended models."""
    verdict: str  # "ATTACK" | "BENIGN"
    confidence: float
    baseline_prob: float
    defended_prob: float
    baseline_verdict: str
    defended_verdict: str
    disagreement: bool
    attention_divergence: float
    divergence_triggered: bool
    explanation: str


class EnsembleDefense:
    """Ensemble defense combining baseline and PGD-AT models with attention divergence trigger.
    
    Adheres to Blueprint Section 8:
    - Baseline threshold recalibrated to 0.88
    - Defended threshold recalibrated to 0.50
    - Model disagreement raises an intrusion alert (evasion detection)
    - Attention divergence > 0.150 routes flow to defended model
    """

    def __init__(
        self,
        baseline_model: nn.Module,
        defended_model: nn.Module,
        baseline_threshold: float = 0.880,
        defended_threshold: float = 0.500,
        divergence_threshold: float = 0.050,
    ) -> None:
        self.baseline_model = baseline_model
        self.defended_model = defended_model
        self.baseline_threshold = baseline_threshold
        self.defended_threshold = defended_threshold
        self.divergence_threshold = divergence_threshold

    def evaluate(self, x: torch.Tensor) -> EnsembleDecision:
        """Run dual-model consensus and attention divergence check."""
        self.baseline_model.eval()
        self.defended_model.eval()

        with torch.no_grad():
            base_logits = self.baseline_model(x)
            base_prob = torch.softmax(base_logits, dim=1)[0, 1].item()
            base_maps = self.baseline_model.get_attention_maps()
            base_rollout = compute_attention_rollout(base_maps)
            base_attr = get_feature_attribution(base_rollout)[0].cpu().numpy()

            def_logits = self.defended_model(x)
            def_prob = torch.softmax(def_logits, dim=1)[0, 1].item()
            def_maps = self.defended_model.get_attention_maps()
            def_rollout = compute_attention_rollout(def_maps)
            def_attr = get_feature_attribution(def_rollout)[0].cpu().numpy()

        base_verdict = "ATTACK" if base_prob > self.baseline_threshold else "BENIGN"
        def_verdict = "ATTACK" if def_prob > self.defended_threshold else "BENIGN"

        # Attention Divergence Calculation
        div_score = compute_divergence(base_attr, def_attr)
        div_triggered = div_score > self.divergence_threshold
        disagreement = (base_verdict != def_verdict)

        # Ensemble Consensus Rules (Blueprint Section 8):
        # 1. If divergence triggered, trust defended model
        if div_triggered:
            verdict = def_verdict
            confidence = def_prob if verdict == "ATTACK" else (1.0 - def_prob)
            explanation = f"Attention divergence trigger activated (JSD={div_score:.4f} > {self.divergence_threshold:.3f}); defended model prioritized."
            logger.warning(f"[ENSEMBLE] {explanation}")

        # 2. If disagreement (e.g. baseline tricked to say BENIGN, defended says ATTACK):
        elif disagreement:
            # Conservative security posture: Disagreement raises an alert
            verdict = "ATTACK"
            confidence = max(base_prob, def_prob)
            explanation = f"Model disagreement detected (Base={base_verdict}, Def={def_verdict}); defensive alert raised."
            logger.info(f"[ENSEMBLE] {explanation}")

        # 3. Consensus agreement
        else:
            verdict = base_verdict
            avg_prob = 0.5 * (base_prob + def_prob)
            confidence = avg_prob if verdict == "ATTACK" else (1.0 - avg_prob)
            explanation = f"Consensus agreement between baseline and defended models ({verdict})."

        return EnsembleDecision(
            verdict=verdict,
            confidence=float(confidence),
            baseline_prob=float(base_prob),
            defended_prob=float(def_prob),
            baseline_verdict=base_verdict,
            defended_verdict=def_verdict,
            disagreement=disagreement,
            attention_divergence=float(div_score),
            divergence_triggered=div_triggered,
            explanation=explanation,
        )

