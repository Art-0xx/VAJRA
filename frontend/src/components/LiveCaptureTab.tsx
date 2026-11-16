"use client";

import React, { useState } from "react";
import { Sigint3DCanvas } from "./Sigint3DCanvas";
import { AlertItem, ScoreboardResponse, runAttack } from "../lib/api";
import { ShieldCheck, ShieldAlert, Activity, Send, Skull, RotateCcw } from "lucide-react";

interface LiveCaptureTabProps {
  alerts: AlertItem[];
  scoreboard: ScoreboardResponse | null;
  activeModel: string;
  activeThreshold: number;
  onSendBenign: () => Promise<void>;
  isSendingBenign: boolean;
  onRefresh?: () => void;
}

export const LiveCaptureTab: React.FC<LiveCaptureTabProps> = ({
  alerts,
  scoreboard,
  activeModel,
  activeThreshold,
  onSendBenign,
  isSendingBenign,
  onRefresh,
}) => {
  const [filterVerdict, setFilterVerdict] = useState<string>("ALL");
  const [hasInjectedFlow, setHasInjectedFlow] = useState<boolean>(false);
  const [flowState, setFlowState] = useState<"IDLE" | "BENIGN" | "ATTACK" | "DEFENDED">("IDLE");
  const [isInjectingAttack, setIsInjectingAttack] = useState<boolean>(false);

  const latestAlert = alerts.length > 0 ? alerts[0] : null;
  const isAttacking = flowState === "ATTACK" || (latestAlert ? latestAlert.score >= latestAlert.threshold : false);
  const isDefended = activeModel === "defended";

  const handleInjectBenign = async () => {
    setHasInjectedFlow(true);
    setFlowState("BENIGN");
    await onSendBenign();
    onRefresh?.();
  };

  const handleInjectAttack = async () => {
    setIsInjectingAttack(true);
    setHasInjectedFlow(true);
    setFlowState("ATTACK");
    try {
      await runAttack({
        target_port: 80,
        attack_name: "DDoS-LOIC-HTTP",
        mode: "gradient",
        epsilon: 0.05,
      });
      onRefresh?.();
    } catch (err) {
      console.error("Attack injection error:", err);
    } finally {
      setIsInjectingAttack(false);
    }
  };

  const handleResetStandby = () => {
    setHasInjectedFlow(false);
    setFlowState("IDLE");
  };

  const filteredAlerts = alerts.filter((a) => {
    if (filterVerdict === "ALL") return true;
    if (filterVerdict === "ATTACK") return a.score >= a.threshold;
    if (filterVerdict === "BENIGN") return a.score < a.threshold;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Top Hero: Tactical 3D Data Surface Topology and Matrix */}
      <div className="space-y-2">
        <div className="flex flex-col md:flex-row md:items-center justify-between px-1 gap-2">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-cyan-400" />
            <h2 className="text-sm font-bold tracking-widest text-zinc-100 uppercase font-mono">
              3D DATA SURFACE TOPOLOGY // REAL-TIME NETWORK MATRIX
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
            <button
              onClick={handleInjectBenign}
              disabled={isSendingBenign}
              className="flex items-center gap-1.5 rounded border border-emerald-500/40 bg-emerald-950/20 px-3 py-1 text-emerald-400 hover:bg-emerald-900/30 transition-all font-bold"
            >
              <Send className="h-3 w-3" />
              <span>{isSendingBenign ? "INJECTING BENIGN..." : "INJECT BENIGN FLOW"}</span>
            </button>

            <button
              onClick={handleInjectAttack}
              disabled={isInjectingAttack}
              className="flex items-center gap-1.5 rounded border border-red-500/40 bg-red-950/20 px-3 py-1 text-red-400 hover:bg-red-900/30 transition-all font-bold"
            >
              <Skull className="h-3 w-3" />
              <span>{isInjectingAttack ? "INJECTING ATTACK..." : "INJECT ATTACK FLOW"}</span>
            </button>

            {hasInjectedFlow && (
              <button
                onClick={handleResetStandby}
                className="flex items-center gap-1 rounded border border-[#1f293d] bg-black/40 px-2.5 py-1 text-zinc-400 hover:text-zinc-200 transition-all text-[11px]"
              >
                <RotateCcw className="h-3 w-3" />
                <span>STANDBY</span>
              </button>
            )}
          </div>
        </div>

        {/* 3D Canvas */}
        <Sigint3DCanvas
          hasInjectedFlow={hasInjectedFlow}
          flowState={flowState}
          isAttacking={isAttacking}
          isDefended={isDefended}
          attackMethod={latestAlert?.attack_label || "DDoS-LOIC-HTTP"}
          epsilon={0.05}
          activeTau={activeThreshold}
          cleanProb={latestAlert ? latestAlert.score : 0.005}
          advProb={0.014}
          divergenceScore={latestAlert?.attention_divergence || 0.168}
          latencyMs={latestAlert?.latency_ms || 1.42}
        />
      </div>

      {/* KPI Status Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <div className="text-[11px] font-mono uppercase text-zinc-500">CLEAN ATTACKS DETECTED</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {scoreboard ? `${scoreboard.detected_clean_attacks}/${scoreboard.total_clean_attacks}` : "0/0"}
            </span>
            <span className="text-xs font-mono text-emerald-500">
              {scoreboard ? `${(scoreboard.clean_detection_rate ?? 100).toFixed(1)}%` : "100%"}
            </span>
          </div>
        </div>

        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <div className="text-[11px] font-mono uppercase text-zinc-500">ADVERSARIAL EVADED</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span
              className={`text-2xl font-bold font-mono ${
                scoreboard && scoreboard.evaded_adversarial_attacks > 0 ? "text-red-400" : "text-emerald-400"
              }`}
            >
              {scoreboard ? `${scoreboard.evaded_adversarial_attacks}/${scoreboard.total_adversarial_attacks}` : "0/0"}
            </span>
            <span className="text-xs font-mono text-zinc-400">
              {scoreboard ? `${(scoreboard.adversarial_evasion_rate ?? 0).toFixed(1)}%` : "0%"}
            </span>
          </div>
        </div>

        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <div className="text-[11px] font-mono uppercase text-zinc-500">BENIGN FALSE POSITIVES</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {scoreboard ? `${scoreboard.false_positives}/${scoreboard.total_benign_packets}` : "0/0"}
            </span>
            <span className="text-xs font-mono text-emerald-400">
              {scoreboard ? `${(scoreboard.false_positive_rate ?? 0).toFixed(2)}%` : "0.00%"}
            </span>
          </div>
        </div>

        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <div className="text-[11px] font-mono uppercase text-zinc-500">OPERATIONAL BACKBONE</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-lg font-bold font-mono text-cyan-400 uppercase">
              {activeModel}
            </span>
            <span className="text-xs font-mono text-zinc-400">
              τ={(activeThreshold ?? 0.88).toFixed(3)}
            </span>
          </div>
        </div>
      </div>

      {/* Real-time Alerts Table */}
      <div className="rounded border border-[#1f293d] bg-[#090d16] overflow-hidden">
        <div className="flex items-center justify-between border-b border-[#1f293d] px-4 py-3 bg-black/40">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold font-mono tracking-wider text-zinc-200 uppercase">
              STREAM INGESTION FEED ({filteredAlerts.length} LOGGED FLOWS)
            </span>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs">
            <button
              onClick={() => setFilterVerdict("ALL")}
              className={`px-2 py-0.5 rounded ${
                filterVerdict === "ALL" ? "bg-zinc-800 text-white font-bold" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              ALL
            </button>
            <button
              onClick={() => setFilterVerdict("ATTACK")}
              className={`px-2 py-0.5 rounded ${
                filterVerdict === "ATTACK" ? "bg-red-950 text-red-400 font-bold border border-red-800/40" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              ATTACKS
            </button>
            <button
              onClick={() => setFilterVerdict("BENIGN")}
              className={`px-2 py-0.5 rounded ${
                filterVerdict === "BENIGN" ? "bg-emerald-950 text-emerald-400 font-bold border border-emerald-800/40" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              BENIGN
            </button>
          </div>
        </div>

        <div className="overflow-x-auto max-h-[340px] overflow-y-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-[#0d1322] text-zinc-400 border-b border-[#1f293d] sticky top-0">
              <tr>
                <th className="py-2.5 px-4 font-semibold">TIMESTAMP</th>
                <th className="py-2.5 px-4 font-semibold">FLOW (SRC → DST)</th>
                <th className="py-2.5 px-4 font-semibold">VERDICT</th>
                <th className="py-2.5 px-4 font-semibold">CONFIDENCE</th>
                <th className="py-2.5 px-4 font-semibold">MODEL / THRESHOLD</th>
                <th className="py-2.5 px-4 font-semibold">JSD DIVERGENCE</th>
                <th className="py-2.5 px-4 font-semibold">LATENCY</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1f293d]/50 text-zinc-300">
              {filteredAlerts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-zinc-600">
                    No packet flows ingested yet. Click &quot;Inject Benign Flow&quot; or &quot;Run 3-Phase Demo&quot;.
                  </td>
                </tr>
              ) : (
                filteredAlerts.map((item, idx) => {
                  const isAttack = item.score >= item.threshold;
                  return (
                    <tr key={idx} className="hover:bg-zinc-900/40 transition-colors">
                      <td className="py-2 px-4 text-zinc-400">{item.timestamp}</td>
                      <td className="py-2 px-4">
                        <span className="text-zinc-300">{item.src_ip}</span>
                        <span className="text-zinc-500"> → </span>
                        <span className="text-cyan-400">:{item.dst_port}</span>
                      </td>
                      <td className="py-2 px-4">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                            isAttack
                              ? "bg-red-500/10 text-red-400 border border-red-500/30"
                              : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                          }`}
                        >
                          {isAttack ? <ShieldAlert className="h-3 w-3" /> : <ShieldCheck className="h-3 w-3" />}
                          {item.verdict}
                        </span>
                      </td>
                      <td className="py-2 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                            <div
                              className={`h-full ${isAttack ? "bg-red-500" : "bg-emerald-500"}`}
                              style={{ width: `${Math.min(100, item.score * 100)}%` }}
                            />
                          </div>
                          <span className={isAttack ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>
                            {((item.score ?? 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-4 text-zinc-400">
                        <span className="uppercase text-zinc-300">{item.model_used}</span>
                        <span className="text-zinc-600"> (τ={(item.threshold ?? 0.88).toFixed(3)})</span>
                      </td>
                      <td className="py-2 px-4">
                        <span
                          className={`font-semibold ${
                            (item.attention_divergence || 0) > 0.15
                              ? "text-amber-400"
                              : "text-zinc-500"
                          }`}
                        >
                          {(item.attention_divergence || 0.12).toFixed(3)}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-zinc-400">
                        {item.latency_ms ? `${item.latency_ms.toFixed(2)}ms` : "1.42ms"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
