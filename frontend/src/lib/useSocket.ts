"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AlertItem } from "./api";

export interface WebSocketMessage {
  type: "ALERT" | "SWAP" | "DEMO_STEP" | "HEARTBEAT";
  payload: any;
}

export function useVAJRASocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [latestAlert, setLatestAlert] = useState<AlertItem | null>(null);
  const [alertsStream, setAlertsStream] = useState<AlertItem[]>([]);
  const [activeModel, setActiveModel] = useState<string>("baseline");
  const [activeThreshold, setActiveThreshold] = useState<number>(0.88);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "ALERT" || msg.type === "verdict") {
            const raw = msg.payload || msg;
            const alertItem: AlertItem = {
              id: raw.id || Date.now(),
              timestamp: raw.timestamp ? (raw.timestamp.includes("T") ? raw.timestamp.split("T")[1]?.slice(0, 8) : raw.timestamp) : new Date().toISOString().slice(11, 19),
              src_ip: raw.src_ip || "10.0.0.1",
              dst_ip: raw.dst_ip || "10.0.0.2",
              src_port: raw.src_port || 52110,
              dst_port: raw.dst_port || 80,
              protocol: raw.protocol || "TCP",
              score: raw.score ?? raw.confidence ?? raw.prob_attack ?? 0.0,
              threshold: raw.threshold ?? 0.88,
              verdict: raw.verdict || "BENIGN",
              model_used: raw.model_used || raw.model_version || raw.model || "baseline",
              attack_label: raw.step_name || (raw.verdict === "MALICIOUS" ? "DDoS-LOIC-HTTP" : "Normal Benign Flow"),
              attention_divergence: raw.attention_divergence ?? 0.0,
              latency_ms: raw.latency_ms ?? 1.42,
            };
            setLatestAlert(alertItem);
            setAlertsStream((prev) => [alertItem, ...prev.slice(0, 99)]);
          } else if (msg.type === "SWAP" || msg.type === "model_swap") {
            const target = msg.payload?.active_model || msg.active_model;
            const tau = msg.payload?.threshold || msg.threshold;
            if (target) setActiveModel(target);
            if (tau) setActiveThreshold(tau);
          } else if (msg.type === "DEMO_STEP" || msg.type === "demo_sequence_phase") {
            const target = msg.payload?.model || msg.payload?.active_model || msg.active_model;
            const tau = msg.payload?.threshold || msg.threshold;
            if (target) setActiveModel(target);
            if (tau) setActiveThreshold(tau);
          }
        } catch (err) {
          console.error("WS Parse Error:", err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (err) {
      console.error("WS Init Error:", err);
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return {
    isConnected,
    latestAlert,
    alertsStream,
    setAlertsStream,
    activeModel,
    setActiveModel,
    activeThreshold,
    setActiveThreshold,
  };
}

