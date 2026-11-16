"use client";

import React, { useState } from "react";
import { Shield, ShieldAlert, Cpu, Network, RefreshCw, Zap } from "lucide-react";

interface NavbarProps {
  activeModel: string;
  activeThreshold: number;
  isWsConnected: boolean;
  device?: string;
  networkInterface?: string;
  onSwapModel: (model: "baseline" | "defended") => Promise<void>;
  onRunDemoSequence: () => Promise<void>;
  isDemoRunning: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeModel,
  activeThreshold,
  isWsConnected,
  device = "CPU",
  networkInterface = "lo",
  onSwapModel,
  onRunDemoSequence,
  isDemoRunning,
}) => {
  const [swapping, setSwapping] = useState(false);

  const handleToggleModel = async () => {
    if (swapping) return;
    setSwapping(true);
    const target = activeModel === "baseline" ? "defended" : "baseline";
    try {
      await onSwapModel(target);
    } finally {
      setSwapping(false);
    }
  };

  return (
    <header className="w-full border-b border-[#1f293d] bg-[#090d16] px-6 py-3">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Left: Brand & SOC Badge */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded border border-cyan-500/30 bg-cyan-950/20 text-cyan-400">
            <Shield className="h-6 w-6 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-widest text-zinc-100 font-mono">VAJRA</span>
              <span className="rounded bg-zinc-800/80 px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wider text-cyan-400 border border-cyan-800/40">
                NIDS // TAC-OPS
              </span>
            </div>
            <p className="text-xs text-zinc-400 font-mono">
              Adversarially Robust Transformer Intrusion Detection System
            </p>
          </div>
        </div>

        {/* Center: System Telemetry Chips */}
        <div className="flex flex-wrap items-center gap-2">
          {/* WS Status */}
          <div className="flex items-center gap-2 rounded border border-[#1f293d] bg-black/40 px-3 py-1.5 text-xs font-mono">
            <span
              className={`h-2 w-2 rounded-full ${
                isWsConnected ? "bg-emerald-400 animate-pulse" : "bg-red-500"
              }`}
            />
            <span className={isWsConnected ? "text-emerald-400" : "text-red-400"}>
              {isWsConnected ? "WS LIVE" : "WS RECONNECTING"}
            </span>
          </div>

          {/* Interface Chip */}
          <div className="flex items-center gap-1.5 rounded border border-[#1f293d] bg-black/40 px-3 py-1.5 text-xs font-mono text-zinc-300">
            <Network className="h-3.5 w-3.5 text-zinc-400" />
            <span className="text-zinc-500">IFACE:</span>
            <span className="font-semibold text-zinc-200">{networkInterface}</span>
          </div>

          {/* Device Chip */}
          <div className="flex items-center gap-1.5 rounded border border-[#1f293d] bg-black/40 px-3 py-1.5 text-xs font-mono text-zinc-300">
            <Cpu className="h-3.5 w-3.5 text-zinc-400" />
            <span className="text-zinc-500">DEV:</span>
            <span className="font-semibold text-zinc-200">{device}</span>
          </div>

          {/* Active Model & Threshold Toggle */}
          <div className="flex items-center rounded border border-[#1f293d] bg-black/60 p-0.5 font-mono text-xs">
            <button
              onClick={() => onSwapModel("baseline")}
              disabled={swapping || activeModel === "baseline"}
              className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
                activeModel === "baseline"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <ShieldAlert className="h-3.5 w-3.5" />
              <span>BASELINE (τ=0.880)</span>
            </button>
            <button
              onClick={() => onSwapModel("defended")}
              disabled={swapping || activeModel === "defended"}
              className={`px-3 py-1 rounded transition-all flex items-center gap-1.5 ${
                activeModel === "defended"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Shield className="h-3.5 w-3.5" />
              <span>DEFENDED (τ=0.612)</span>
            </button>
          </div>
        </div>

        {/* Right: One-Click Demo Sequence Button */}
        <div>
          <button
            onClick={onRunDemoSequence}
            disabled={isDemoRunning}
            className={`flex items-center gap-2 rounded px-4 py-2 font-mono text-xs font-bold tracking-wider uppercase transition-all shadow-lg ${
              isDemoRunning
                ? "bg-cyan-900/50 text-cyan-200 border border-cyan-500/40 cursor-wait"
                : "bg-cyan-500 hover:bg-cyan-400 text-black shadow-cyan-500/20 active:scale-95"
            }`}
          >
            {isDemoRunning ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin text-cyan-300" />
                <span>EXECUTING 3-PHASE DEMO...</span>
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 fill-black" />
                <span>RUN 3-PHASE DEMO</span>
              </>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
