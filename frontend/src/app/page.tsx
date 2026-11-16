"use client";

import React, { useState, useEffect } from "react";
import { Navbar } from "../components/Navbar";
import { LiveCaptureTab } from "../components/LiveCaptureTab";
import { RedTeamTab } from "../components/RedTeamTab";
import { BlueTeamTab } from "../components/BlueTeamTab";
import { MetricsTab } from "../components/MetricsTab";
import { useVAJRASocket } from "../lib/useSocket";
import {
  getHealth,
  swapModel,
  sendBenignTraffic,
  runDemoSequence,
  getScoreboard,
  getAlerts,
  AlertItem,
  ScoreboardResponse,
  DemoStep,
} from "../lib/api";
import { Activity, Skull, Shield, BarChart2, CheckCircle2 } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"capture" | "redteam" | "blueteam" | "metrics">("capture");
  const [device, setDevice] = useState<string>("CPU");
  const [networkInterface, setNetworkInterface] = useState<string>("lo");
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [scoreboard, setScoreboard] = useState<ScoreboardResponse | null>(null);
  const [isSendingBenign, setIsSendingBenign] = useState(false);
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const [demoSteps, setDemoSteps] = useState<DemoStep[] | null>(null);

  const {
    isConnected: isWsConnected,
    latestAlert,
    activeModel,
    setActiveModel,
    activeThreshold,
    setActiveThreshold,
  } = useVAJRASocket();

  // Load initial health, scoreboard, and alerts history
  useEffect(() => {
    getHealth()
      .then((h) => {
        if (h.active_model) setActiveModel(h.active_model);
        if (h.active_threshold != null) setActiveThreshold(h.active_threshold);
        if (h.device) setDevice(h.device);
        if (h.interface) setNetworkInterface(h.interface);
      })
      .catch((err) => console.error("Initial health error:", err));

    refreshScoreboard();
    refreshAlerts();
  }, [setActiveModel, setActiveThreshold]);

  // When WS stream yields new alert, update local state
  useEffect(() => {
    if (latestAlert) {
      setAlerts((prev) => [latestAlert, ...prev.slice(0, 99)]);
      refreshScoreboard();
    }
  }, [latestAlert]);

  const refreshScoreboard = () => {
    getScoreboard(activeModel)
      .then((sc) => setScoreboard(sc))
      .catch((err) => console.error("Scoreboard error:", err));
  };

  useEffect(() => {
    refreshScoreboard();
  }, [activeModel]);

  const refreshAlerts = () => {
    getAlerts(50, 0)
      .then((res) => {
        if (res && res.items) setAlerts(res.items);
      })
      .catch((err) => console.error("Alerts error:", err));
  };

  const handleSwapModel = async (model: "baseline" | "defended") => {
    const res = await swapModel(model);
    setActiveModel(res.active_model);
    setActiveThreshold(res.threshold);
    refreshScoreboard();
  };

  const handleSendBenign = async () => {
    setIsSendingBenign(true);
    try {
      await sendBenignTraffic(3);
      refreshScoreboard();
      refreshAlerts();
    } catch (err) {
      console.error("Benign traffic error:", err);
    } finally {
      setIsSendingBenign(false);
    }
  };

  const handleRunDemoSequence = async () => {
    setIsDemoRunning(true);
    setDemoSteps(null);
    try {
      const res = await runDemoSequence();
      setDemoSteps(res.steps);
      refreshScoreboard();
      refreshAlerts();
    } catch (err) {
      console.error("Demo sequence error:", err);
    } finally {
      setIsDemoRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col font-mono selection:bg-cyan-500 selection:text-black">
      {/* Top Navbar */}
      <Navbar
        activeModel={activeModel}
        activeThreshold={activeThreshold}
        isWsConnected={isWsConnected}
        device={device}
        networkInterface={networkInterface}
        onSwapModel={handleSwapModel}
        onRunDemoSequence={handleRunDemoSequence}
        isDemoRunning={isDemoRunning}
      />

      {/* Demo Sequence Banner if completed */}
      {demoSteps && (
        <div className="bg-cyan-950/40 border-b border-cyan-800/40 px-6 py-3">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-cyan-400" />
              <div>
                <span className="text-xs font-bold text-cyan-300 uppercase tracking-wider">
                  3-PHASE LIVE DEMO SEQUENCE COMPLETED SUCCESSFULLY
                </span>
                <p className="text-[11px] text-zinc-400">
                  Continuous visual story validated: Benign flow passed → Baseline model evaded → Defended model held.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {demoSteps.map((s, idx) => (
                <div
                  key={idx}
                  className="rounded border border-[#1f293d] bg-black/60 px-3 py-1 text-[11px] flex items-center gap-1.5"
                >
                  <span className="text-zinc-500 font-bold">P{idx + 1}:</span>
                  <span className="text-zinc-300 font-bold">{s.label}</span>
                  <span
                    className={
                      s.outcome.includes("HELD") || s.outcome.includes("PASS")
                        ? "text-emerald-400 font-bold"
                        : "text-amber-400 font-bold"
                    }
                  >
                    ({((s?.confidence ?? 0) * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
              <button
                onClick={() => setDemoSteps(null)}
                className="text-xs text-zinc-500 hover:text-zinc-300 ml-2"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab Navigation Navigation Bar */}
      <div className="border-b border-[#1f293d] bg-[#050811] px-6">
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab("capture")}
            className={`py-3 px-4 text-xs font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
              activeTab === "capture"
                ? "border-cyan-400 text-cyan-300 bg-cyan-950/20"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Activity className="h-4 w-4" />
            <span>1. LIVE CAPTURE & 3D TOPOLOGY</span>
          </button>

          <button
            onClick={() => setActiveTab("redteam")}
            className={`py-3 px-4 text-xs font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
              activeTab === "redteam"
                ? "border-red-500 text-red-400 bg-red-950/20"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Skull className="h-4 w-4" />
            <span>2. RED TEAM // ATTACK STUDIO</span>
          </button>

          <button
            onClick={() => setActiveTab("blueteam")}
            className={`py-3 px-4 text-xs font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
              activeTab === "blueteam"
                ? "border-emerald-400 text-emerald-300 bg-emerald-950/20"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Shield className="h-4 w-4" />
            <span>3. BLUE TEAM // DEFENSE & ATTENTION</span>
          </button>

          <button
            onClick={() => setActiveTab("metrics")}
            className={`py-3 px-4 text-xs font-bold uppercase tracking-wider border-b-2 transition-all flex items-center gap-2 ${
              activeTab === "metrics"
                ? "border-amber-400 text-amber-300 bg-amber-950/20"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <BarChart2 className="h-4 w-4" />
            <span>4. METRICS & BENCHMARKS</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        {activeTab === "capture" && (
          <LiveCaptureTab
            alerts={alerts}
            scoreboard={scoreboard}
            activeModel={activeModel}
            activeThreshold={activeThreshold}
            onSendBenign={handleSendBenign}
            isSendingBenign={isSendingBenign}
            onRefresh={() => {
              refreshAlerts();
              refreshScoreboard();
            }}
          />
        )}

        {activeTab === "redteam" && (
          <RedTeamTab
            activeModel={activeModel}
            activeThreshold={activeThreshold}
            onAttackCompleted={() => {
              refreshScoreboard();
              refreshAlerts();
            }}
          />
        )}

        {activeTab === "blueteam" && (
          <BlueTeamTab
            activeModel={activeModel}
            activeThreshold={activeThreshold}
            onSwapModel={handleSwapModel}
          />
        )}

        {activeTab === "metrics" && <MetricsTab />}
      </main>

      {/* SOC Footer */}
      <footer className="border-t border-[#1f293d] bg-[#050811] px-6 py-3 text-[11px] text-zinc-500 font-mono flex flex-col md:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span>VAJRA ADVERSARIAL NIDS // V2.4 TACTICAL ENGINE</span>
          <span className="text-zinc-700">|</span>
          <span>TRANSFORMER ENCODER (4 LAYERS, 4 HEADS, D_MODEL=64)</span>
        </div>
        <div className="flex items-center gap-4 text-zinc-400">
          <span>CLASSIFIED TOPOLOGY VISUALIZER</span>
          <span className="text-emerald-400 flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400" />
            OPERATIONAL
          </span>
        </div>
      </footer>
    </div>
  );
}
