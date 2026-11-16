# VAJRA Persistent Alert Store (Blueprint Section 5.5)
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AlertStore:
    """SQLite-backed thread-safe alert and telemetry event store.
    
    Guarantees every classification verdict and block action is persisted
    and retrievable by the dashboard and response engine.
    """

    def __init__(self, db_path: str = "artifacts/alerts/alerts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    flow_id TEXT NOT NULL,
                    src_ip TEXT DEFAULT '192.168.1.100',
                    dst_ip TEXT DEFAULT '192.168.1.1',
                    src_port INTEGER DEFAULT 0,
                    dst_port INTEGER DEFAULT 80,
                    protocol TEXT DEFAULT 'TCP',
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    attention_divergence REAL DEFAULT 0.0,
                    top_features TEXT DEFAULT '[]',
                    blocked INTEGER DEFAULT 0,
                    raw_details TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scoreboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    clean_detected INTEGER DEFAULT 0,
                    clean_total INTEGER DEFAULT 0,
                    adv_evaded INTEGER DEFAULT 0,
                    adv_total INTEGER DEFAULT 0,
                    false_positives INTEGER DEFAULT 0,
                    benign_total INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_verdict ON alerts (verdict)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_flow_id ON alerts (flow_id)")
            conn.commit()

    def record_verdict(
        self,
        flow_id: str,
        verdict: str,
        confidence: float,
        model_version: str,
        threshold: float,
        src_ip: str = "192.168.1.100",
        dst_ip: str = "192.168.1.1",
        src_port: int = 0,
        dst_port: int = 80,
        protocol: str = "TCP",
        attention_divergence: float = 0.0,
        top_features: Optional[List[str]] = None,
        blocked: bool = False,
        raw_details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Write an evaluation record to the SQLite store.
        
        Returns:
            The inserted row ID.
        """
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        features_json = json.dumps(top_features or [])
        details_json = json.dumps(raw_details or {})

        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (
                    timestamp, flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    verdict, confidence, model_version, threshold,
                    attention_divergence, top_features, blocked, raw_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    verdict, float(confidence), model_version, float(threshold),
                    float(attention_divergence), features_json, int(blocked), details_json
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_recent(
        self,
        limit: int = 50,
        verdict: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent alert records."""
        query = "SELECT * FROM alerts"
        params: List[Any] = []
        if verdict:
            query += " WHERE verdict = ?"
            params.append(verdict.upper())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["top_features"] = json.loads(item.get("top_features") or "[]")
                item["raw_details"] = json.loads(item.get("raw_details") or "{}")
                item["blocked"] = bool(item.get("blocked"))
                results.append(item)
            return results

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute live telemetry summary statistics directly from stored records."""
        with self._lock, self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            attacks = conn.execute("SELECT COUNT(*) FROM alerts WHERE verdict = 'ATTACK'").fetchone()[0]
            benign = conn.execute("SELECT COUNT(*) FROM alerts WHERE verdict = 'BENIGN'").fetchone()[0]
            blocked = conn.execute("SELECT COUNT(*) FROM alerts WHERE blocked = 1").fetchone()[0]
            avg_div = conn.execute("SELECT AVG(attention_divergence) FROM alerts").fetchone()[0] or 0.0

            return {
                "total_flows": total,
                "attacks_detected": attacks,
                "benign_verified": benign,
                "flows_blocked": blocked,
                "average_divergence": round(float(avg_div), 4),
            }

    def get_paginated(
        self,
        limit: int = 20,
        offset: int = 0,
        verdict: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve paginated alert records and total count."""
        where_clause = ""
        params: List[Any] = []
        if verdict:
            where_clause = " WHERE verdict = ?"
            params.append(verdict.upper())

        count_query = f"SELECT COUNT(*) FROM alerts{where_clause}"
        query = f"SELECT * FROM alerts{where_clause} ORDER BY id DESC LIMIT ? OFFSET ?"

        with self._lock, self._get_connection() as conn:
            total = conn.execute(count_query, params).fetchone()[0]
            query_params = list(params) + [limit, offset]
            rows = conn.execute(query, query_params).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["top_features"] = json.loads(item.get("top_features") or "[]")
                item["raw_details"] = json.loads(item.get("raw_details") or "{}")
                item["blocked"] = bool(item.get("blocked"))
                results.append(item)
            return {
                "alerts": results,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    def record_scoreboard_event(
        self,
        model_version: str,
        event_type: str,
        clean_detected: int = 0,
        clean_total: int = 0,
        adv_evaded: int = 0,
        adv_total: int = 0,
        false_positives: int = 0,
        benign_total: int = 0,
        latency_ms: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> int:
        """Write an evaluation campaign event to the scoreboard table."""
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO scoreboard (
                    timestamp, model_version, event_type,
                    clean_detected, clean_total, adv_evaded, adv_total,
                    false_positives, benign_total, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, model_version, event_type,
                    clean_detected, clean_total, adv_evaded, adv_total,
                    false_positives, benign_total, latency_ms
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_scoreboard(self) -> Dict[str, Any]:
        """Aggregate running scoreboard totals grouped by model."""
        with self._lock, self._get_connection() as conn:
            # Baseline aggregation
            base_row = conn.execute("""
                SELECT 
                    COALESCE(SUM(clean_detected), 0) as clean_det,
                    COALESCE(SUM(clean_total), 0) as clean_tot,
                    COALESCE(SUM(adv_evaded), 0) as adv_eva,
                    COALESCE(SUM(adv_total), 0) as adv_tot,
                    COALESCE(SUM(false_positives), 0) as fp,
                    COALESCE(SUM(benign_total), 0) as ben_tot
                FROM scoreboard WHERE model_version = 'baseline'
            """).fetchone()

            # Defended aggregation
            def_row = conn.execute("""
                SELECT 
                    COALESCE(SUM(clean_detected), 0) as clean_det,
                    COALESCE(SUM(clean_total), 0) as clean_tot,
                    COALESCE(SUM(adv_evaded), 0) as adv_eva,
                    COALESCE(SUM(adv_total), 0) as adv_tot,
                    COALESCE(SUM(false_positives), 0) as fp,
                    COALESCE(SUM(benign_total), 0) as ben_tot
                FROM scoreboard WHERE model_version = 'defended'
            """).fetchone()

            total_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

            return {
                "baseline": {
                    "clean_detected": base_row[0],
                    "clean_total": base_row[1],
                    "adv_evaded": base_row[2],
                    "adv_total": base_row[3],
                    "false_positives": base_row[4],
                    "benign_total": base_row[5],
                },
                "defended": {
                    "clean_detected": def_row[0],
                    "clean_total": def_row[1],
                    "adv_evaded": def_row[2],
                    "adv_total": def_row[3],
                    "false_positives": def_row[4],
                    "benign_total": def_row[5],
                },
                "total_alerts": total_alerts,
            }

    def clear(self) -> None:
        """Clear all records from the database."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM scoreboard")
            conn.commit()

