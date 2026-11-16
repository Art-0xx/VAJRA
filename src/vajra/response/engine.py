# VAJRA Automated Incident Response Engine (Blueprint Section 5.5)
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vajra.response")


@dataclass
class ResponseAction:
    """Represents an automated containment or firewall block action."""
    action: str  # "BLOCK" | "ALLOW" | "QUARANTINE"
    flow_id: str
    src_ip: str
    dst_ip: str
    dst_port: int
    rule: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: str = "SIMULATED"
    details: Dict[str, Any] = field(default_factory=dict)


class ResponseEngine:
    """Simulates or executes automated firewall/iptables containment actions
    in response to high-confidence Transformer intrusion classifications.
    """

    def __init__(self, mode: str = "simulate", block_duration_sec: int = 300) -> None:
        self.mode = mode.lower()  # "simulate" or "active"
        self.block_duration_sec = block_duration_sec
        self.active_blocks: Dict[str, ResponseAction] = {}
        self.history: List[ResponseAction] = []

    def handle_verdict(
        self,
        flow_id: str,
        verdict: str,
        confidence: float,
        src_ip: str = "192.168.1.100",
        dst_ip: str = "192.168.1.1",
        dst_port: int = 80,
        protocol: str = "TCP",
    ) -> Optional[ResponseAction]:
        """Trigger an automated incident response action upon intrusion detection."""
        if verdict.upper() != "ATTACK":
            return None

        # Build simulated / active iptables rule
        proto = protocol.lower() if protocol else "tcp"
        rule = f"iptables -A INPUT -s {src_ip} -p {proto} --dport {dst_port} -j DROP"

        action = ResponseAction(
            action="BLOCK",
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=dst_port,
            rule=rule,
            status="SIMULATED" if self.mode == "simulate" else "ACTIVE",
            details={
                "confidence": confidence,
                "protocol": protocol,
                "mode": self.mode,
            },
        )

        if self.mode == "active":
            try:
                # In active mode (e.g. Mininet container), execute the firewall rule
                subprocess.run(rule.split(), check=True, capture_output=True, timeout=2)
                action.status = "EXECUTED"
                logger.info(f"[RESPONSE ENGINE] ACTIVE RULE EXECUTED: {rule}")
            except Exception as e:
                action.status = "EXECUTION_ERROR"
                action.details["error"] = str(e)
                logger.error(f"[RESPONSE ENGINE] Failed to apply iptables rule: {e}")
        else:
            logger.info(f"[RESPONSE ENGINE] SIMULATED BLOCK: {rule}")

        self.active_blocks[src_ip] = action
        self.history.append(action)
        return action

    def get_active_blocks(self) -> List[Dict[str, Any]]:
        """List currently blocked IP addresses and rules."""
        return [
            {
                "src_ip": act.src_ip,
                "rule": act.rule,
                "timestamp": act.timestamp,
                "status": act.status,
            }
            for act in self.active_blocks.values()
        ]

    def clear_blocks(self) -> None:
        """Clear active block state and reset emulated rules."""
        self.active_blocks.clear()

