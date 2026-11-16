# VAJRA FastAPI Service Schemas (Blueprint Section 5.4)
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Flow classification request containing raw or scaled 77 features."""
    features: List[float] = Field(..., description="77 network flow features in CIC-IDS2017 schema")
    flow_id: Optional[str] = Field(default=None, description="Unique flow identifier")
    src_ip: Optional[str] = Field(default="192.168.1.100", description="Source IP address")
    dst_ip: Optional[str] = Field(default="192.168.1.1", description="Destination IP address")
    src_port: Optional[int] = Field(default=0, description="Source port")
    dst_port: Optional[int] = Field(default=80, description="Destination port")
    protocol: Optional[str] = Field(default="TCP", description="Protocol")


class PredictResponse(BaseModel):
    """Classification response matching Blueprint Section 5.4 format."""
    flow_id: str
    verdict: str  # "ATTACK" | "BENIGN"
    confidence: float
    model: str
    threshold: float
    attention_divergence: float
    top_features: List[str]
    probabilities: Dict[str, float]
    blocked: bool
    latency_ms: float


class ExplainRequest(BaseModel):
    """Request for attention rollout explanation and feature justification."""
    features: List[float]
    top_k: int = Field(default=5, ge=1, le=77)
    flow_id: Optional[str] = None


class ExplainResponse(BaseModel):
    """Attention rollout explanation outcome."""
    flow_id: str
    top_features: List[str]
    attributions: Dict[str, float]
    entropy: float
    model: str


class ModelSwapRequest(BaseModel):
    """Hot-swap request to switch active model version without service restart."""
    target_model: Optional[str] = None
    model_name: Optional[str] = None


class ModelSwapResponse(BaseModel):
    """Confirmation of model swap."""
    status: str
    active_model: str
    threshold: float
    message: str


class HealthResponse(BaseModel):
    """System liveness and device telemetry."""
    status: str
    active_model: str
    active_threshold: float = 0.880
    interface: str = "lo"
    device: str
    models_loaded: Dict[str, bool]
    preprocessor_loaded: bool
    features_count: int


class MetricsResponse(BaseModel):
    """Operational latency and throughput benchmarks matching Blueprint Section 9.3."""
    uptime_seconds: float
    total_requests: int
    total_attacks: int
    total_blocks: int
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_fps: float
    alert_rate_per_min: int


# ==============================================================================
# ATTACK & DEFENSE WORKFLOW SCHEMAS
# ==============================================================================
class AttackRunRequest(BaseModel):
    """Request to launch a targeted attack campaign against active model."""
    attack_type: str = Field(default="constrained", description="'constrained' | 'pgd' | 'fgsm' | 'heuristic'")
    target_port: int = Field(default=80, description="Destination port: 80 (HTTP), 22 (SSH), 53 (DNS)")
    epsilon: float = Field(default=0.05, ge=0.001, le=0.5, description="Perturbation budget epsilon")
    target_service: Optional[str] = Field(default=None, description="Human readable service name")


class AttackStepResult(BaseModel):
    """Evaluation result for one packet/flow step."""
    step_name: str
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_len: int
    prob_attack: float
    prob_benign: float
    threshold: float
    verdict: str  # "ATTACK" | "BENIGN"
    confidence: float
    is_evaded: bool
    attention_divergence: float
    top_features: List[str]
    attributions: Dict[str, float]
    latency_ms: float
    timestamp: str


class FeatureDiffItem(BaseModel):
    """Difference in an individual feature attribute before and after perturbation."""
    feature: str
    clean_value: float
    adversarial_value: float
    delta: float
    is_controllable: bool


class AttackRunResponse(BaseModel):
    """Complete campaign results showing clean vs adversarial attack and feature diff."""
    model_config = {"protected_namespaces": ()}
    attack_type: str
    target_port: int
    target_service: str
    epsilon: float
    model_used: str
    threshold: float
    clean_step: AttackStepResult
    adversarial_step: AttackStepResult
    evasion_successful: bool
    rerouted_to_defended: bool
    feature_diffs: List[FeatureDiffItem]
    packet_diff: Dict[str, Any]


class BenignTrafficRequest(BaseModel):
    """Request to inject benign baseline traffic samples."""
    count: int = Field(default=3, ge=1, le=10)


class BenignTrafficResponse(BaseModel):
    """Evaluation results for benign verification traffic."""
    samples: List[AttackStepResult]
    false_positives: int
    total_samples: int


class ScoreboardModelStats(BaseModel):
    clean_detected: int
    clean_total: int
    adv_evaded: int
    adv_total: int
    false_positives: int
    benign_total: int


class ScoreboardResponse(BaseModel):
    """Real-time head-to-head score tracking baseline vs defended model."""
    baseline: ScoreboardModelStats
    defended: ScoreboardModelStats
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_fps: float
    total_alerts: int


class AlertItem(BaseModel):
    """Single persisted alert record."""
    model_config = {"protected_namespaces": ()}
    id: int
    timestamp: str
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    verdict: str
    confidence: float
    model_version: str
    threshold: float
    attention_divergence: float
    top_features: List[str]
    blocked: bool


class AlertsListResponse(BaseModel):
    """Paginated response of alerts."""
    alerts: List[AlertItem]
    total: int
    limit: int
    offset: int


class DemoSequenceResponse(BaseModel):
    """Complete 3-phase automated presentation run result."""
    status: str
    phase1_benign: List[AttackStepResult]
    phase2_baseline_attack: AttackRunResponse
    phase3_defended_attack: AttackRunResponse
    summary: str


class FullMetricsResponse(BaseModel):
    """Exhaustive benchmark metrics from test evaluations and table_B."""
    clean_f1: float
    robust_f1_baseline: float
    robust_f1_defended: float
    clean_accuracy: float
    robust_accuracy_baseline: float
    robust_accuracy_defended: float
    false_positive_rate: float
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_fps: float
    asr_by_attack: List[Dict[str, Any]]
    epsilon_curves: List[Dict[str, Any]]


