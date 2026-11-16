"use client";

import React, { useState } from "react";
import { Shield, Cpu, Activity, Radio } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface BlueTeamTabProps {
  activeModel: string;
  activeThreshold: number;
  onSwapModel: (model: "baseline" | "defended") => Promise<void>;
}

export const BlueTeamTab: React.FC<BlueTeamTabProps> = ({
  activeModel,
  activeThreshold,
  onSwapModel,
}) => {
  const [swapping, setSwapping] = useState(false);
  const jsdScore = 0.168; // Observed JSD during adversarial attack
  const jsdThreshold = 0.150;
  const isAutoRerouted = jsdScore > jsdThreshold;

  const handleSwap = async (target: "baseline" | "defended") => {
    if (swapping || activeModel === target) return;
    setSwapping(true);
    try {
      await onSwapModel(target);
    } finally {
      setSwapping(false);
    }
  };

  // Feature Attention Rollout Mocked Data (Clean vs Adversarial)
  const attentionData = [
    { feature: "Dst Port", clean: 0.185, adv: 0.042 },
    { feature: "Init Win Fwd", clean: 0.162, adv: 0.038 },
    { feature: "Flow Duration", clean: 0.141, adv: 0.138 },
    { feature: "Total Fwd Pkts", clean: 0.128, adv: 0.052 },
    { feature: "Pkt Len Mean", clean: 0.115, adv: 0.029 },
    { feature: "SYN Flag Cnt", clean: 0.098, adv: 0.015 },
    { feature: "Fwd IAT Mean", clean: 0.076, adv: 0.071 },
    { feature: "Header Len", clean: 0.052, adv: 0.048 },
  ];

  return (
    <div className="space-y-6 font-mono">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-[#1f293d] pb-4 gap-2">
        <div>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-emerald-400" />
            <h2 className="text-base font-bold tracking-widest text-zinc-100 uppercase">
              BLUE TEAM // DEFENSIVE POSTURE & ATTENTION ROLLOUT
            </h2>
          </div>
          <p className="text-xs text-zinc-400 mt-0.5">
            Real-time attention divergence tracking and PGD-AT robust model calibration.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-400">ACTIVE BACKBONE:</span>
          <span
            className={`px-2.5 py-0.5 rounded text-xs font-bold uppercase ${
              activeModel === "defended"
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                : "bg-amber-500/20 text-amber-400 border border-amber-500/40"
            }`}
          >
            {activeModel} (τ={(activeThreshold ?? 0.88).toFixed(3)})
          </span>
        </div>
      </div>

      {/* JSD Divergence & Auto-Reroute Alert */}
      <div
        className={`rounded border p-4 transition-all ${
          isAutoRerouted
            ? "border-amber-500/50 bg-amber-950/20 text-amber-200"
            : "border-emerald-500/40 bg-emerald-950/20 text-emerald-200"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Radio className="h-5 w-5 text-amber-400 animate-pulse" />
            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-amber-300">
                DEFENSIVE ATTENTION DIVERGENCE MONITOR // JENSEN-SHANNON DIVERGENCE
              </div>
              <div className="text-[11px] text-zinc-400 mt-0.5">
                Calculates drift between Layer 4 Attention Rollout and Benign Prior Distribution.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-[10px] text-zinc-500 uppercase">OBSERVED D_JS</div>
              <div className="text-xl font-bold text-amber-400">{jsdScore.toFixed(3)}</div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-zinc-500 uppercase">THRESHOLD D_TAU</div>
              <div className="text-xl font-bold text-zinc-400">{jsdThreshold.toFixed(3)}</div>
            </div>
            <span className="px-3 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 text-xs font-bold">
              [!] AUTO-REROUTE TO DEFENDED
            </span>
          </div>
        </div>
      </div>

      {/* Main Grid: Model Switcher & Attention Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Switcher & Calibration */}
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-5 space-y-5">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-zinc-200 flex items-center gap-1.5">
              <Cpu className="h-4 w-4 text-cyan-400" />
              HOT-SWAP TRANSFORMER BACKBONE
            </span>
            <p className="text-[11px] text-zinc-500 mt-1">
              Zero-downtime model swap via FastAPI runtime dispatcher.
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={() => handleSwap("baseline")}
              disabled={swapping || activeModel === "baseline"}
              className={`w-full p-4 rounded text-left border transition-all ${
                activeModel === "baseline"
                  ? "bg-amber-500/10 border-amber-500/50 shadow-lg shadow-amber-500/10"
                  : "bg-black/40 border-[#1f293d] hover:border-zinc-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-zinc-200">BASELINE TRANSFORMER</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">
                  checkpoint: baseline_model.pt
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-2">
                Standard Cross-Entropy training on CIC-IDS2017. Highly vulnerable to gradient perturbation.
              </p>
              <div className="mt-3 flex items-center justify-between text-xs pt-2 border-t border-[#1f293d]/50">
                <span className="text-zinc-500">OPTIMAL THRESHOLD:</span>
                <span className="text-amber-400 font-bold">τ = 0.880</span>
              </div>
            </button>

            <button
              onClick={() => handleSwap("defended")}
              disabled={swapping || activeModel === "defended"}
              className={`w-full p-4 rounded text-left border transition-all ${
                activeModel === "defended"
                  ? "bg-emerald-500/10 border-emerald-500/50 shadow-lg shadow-emerald-500/10"
                  : "bg-black/40 border-[#1f293d] hover:border-zinc-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-emerald-400">DEFENDED (PGD-AT)</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/40">
                  checkpoint: defended_model.pt
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-2">
                Projected Gradient Descent Adversarial Training (ε=0.05). Preserves attention on invariant features.
              </p>
              <div className="mt-3 flex items-center justify-between text-xs pt-2 border-t border-[#1f293d]/50">
                <span className="text-zinc-500">CALIBRATED THRESHOLD:</span>
                <span className="text-emerald-400 font-bold">τ = 0.612</span>
              </div>
            </button>
          </div>
        </div>

        {/* Attention Rollout Chart */}
        <div className="lg:col-span-2 rounded border border-[#1f293d] bg-[#090d16] p-5">
          <div className="flex items-center justify-between border-b border-[#1f293d] pb-3 mb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-zinc-200 flex items-center gap-1.5">
                <Activity className="h-4 w-4 text-cyan-400" />
                TRANSFORMER ATTENTION ROLLOUT (LAYER 4)
              </span>
              <p className="text-[11px] text-zinc-500 mt-0.5">
                Comparison of self-attention weights between Clean Malicious Flow vs Adversarial Perturbed Flow.
              </p>
            </div>
          </div>

          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={attentionData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis
                  dataKey="feature"
                  stroke="#52525b"
                  tick={{ fill: "#a1a1aa", fontSize: 10 }}
                  interval={0}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis stroke="#52525b" tick={{ fill: "#a1a1aa", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#090d16",
                    borderColor: "#1f293d",
                    color: "#f43f5e",
                    fontFamily: "monospace",
                    fontSize: "11px",
                  }}
                />
                <Legend
                  wrapperStyle={{
                    fontSize: "11px",
                    fontFamily: "monospace",
                    paddingTop: "10px",
                  }}
                />
                <Bar dataKey="clean" name="Clean Attack Attention" fill="#38bdf8" />
                <Bar dataKey="adv" name="Adversarial Attention (Evaded)" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
