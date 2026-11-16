"use client";

import React, { useState } from "react";
import { runAttack, AttackRunResponse } from "../lib/api";
import { Skull, Target, ShieldAlert, CheckCircle2, Cpu } from "lucide-react";

interface RedTeamTabProps {
  activeModel: string;
  activeThreshold: number;
  onAttackCompleted?: (res: AttackRunResponse) => void;
}

export const RedTeamTab: React.FC<RedTeamTabProps> = ({
  activeModel,
  activeThreshold,
  onAttackCompleted,
}) => {
  const [targetPort, setTargetPort] = useState<number>(80);
  const [attackName, setAttackName] = useState<string>("DDoS-LOIC-HTTP");
  const [attackMode, setAttackMode] = useState<"gradient" | "heuristic">("gradient");
  const [epsilon, setEpsilon] = useState<number>(0.05);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AttackRunResponse | null>(null);

  const handleLaunch = async () => {
    setLoading(true);
    try {
      const res = await runAttack({
        target_port: targetPort,
        attack_name: attackName,
        mode: attackMode,
        epsilon: epsilon,
      });
      setResult(res);
      if (onAttackCompleted) onAttackCompleted(res);
    } catch (err) {
      console.error("Attack Run Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 font-mono">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-[#1f293d] pb-4 gap-2">
        <div>
          <div className="flex items-center gap-2">
            <Skull className="h-5 w-5 text-red-500" />
            <h2 className="text-base font-bold tracking-widest text-zinc-100 uppercase">
              RED TEAM // PROBLEM-SPACE ADVERSARIAL DISRUPTION
            </h2>
          </div>
          <p className="text-xs text-zinc-400 mt-0.5">
            Craft stealthy adversarial network flows restricted to attacker-controllable features.
          </p>
        </div>
        <div className="text-xs text-zinc-500">
          ACTIVE TARGET: <span className="text-cyan-400 font-bold uppercase">{activeModel}</span> (τ={(activeThreshold ?? 0.88).toFixed(3)})
        </div>
      </div>

      {/* Control Panel Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Port Selector */}
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <label className="text-xs text-zinc-400 uppercase font-bold flex items-center gap-1.5 mb-2">
            <Target className="h-3.5 w-3.5 text-cyan-400" />
            TARGET SERVICE / PORT
          </label>
          <div className="grid grid-cols-2 gap-2">
            {[
              { port: 80, label: "HTTP (80)" },
              { port: 443, label: "HTTPS (443)" },
              { port: 53, label: "DNS (53)" },
              { port: 22, label: "SSH (22)" },
            ].map((item) => (
              <button
                key={item.port}
                onClick={() => setTargetPort(item.port)}
                className={`py-2 px-3 text-xs rounded border transition-all ${
                  targetPort === item.port
                    ? "bg-red-500/20 text-red-300 border-red-500/50 font-bold"
                    : "border-[#1f293d] bg-black/40 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Vector Selector */}
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <label className="text-xs text-zinc-400 uppercase font-bold flex items-center gap-1.5 mb-2">
            <Skull className="h-3.5 w-3.5 text-red-400" />
            ATTACK VECTOR
          </label>
          <select
            value={attackName}
            onChange={(e) => setAttackName(e.target.value)}
            className="w-full rounded border border-[#1f293d] bg-black/60 px-3 py-2 text-xs text-zinc-200 outline-none focus:border-red-500"
          >
            <option value="DDoS-LOIC-HTTP">DDoS LOIC HTTP Flood</option>
            <option value="PortScan">PortScan SYN-Sweep</option>
            <option value="DoS-Slowloris">DoS Slowloris Connection Drain</option>
            <option value="Botnet-ARES">Botnet Command & Control</option>
          </select>
          <div className="mt-2 text-[11px] text-zinc-500">
            Source: CIC-IDS2017 Exploit Vector
          </div>
        </div>

        {/* Attack Mode & Epsilon */}
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4">
          <label className="text-xs text-zinc-400 uppercase font-bold flex items-center gap-1.5 mb-2">
            <Cpu className="h-3.5 w-3.5 text-amber-400" />
            ATTACK MODE & BUDGET (ε)
          </label>
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setAttackMode("gradient")}
              className={`flex-1 py-1 text-xs rounded border transition-all ${
                attackMode === "gradient"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/50 font-bold"
                  : "border-[#1f293d] bg-black/40 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              CONSTRAINED PGD
            </button>
            <button
              onClick={() => setAttackMode("heuristic")}
              className={`flex-1 py-1 text-xs rounded border transition-all ${
                attackMode === "heuristic"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/50 font-bold"
                  : "border-[#1f293d] bg-black/40 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              HEURISTIC
            </button>
          </div>
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
            <span>PERTURBATION BUDGET ε:</span>
            <span className="font-bold text-amber-400">{epsilon.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min="0.005"
            max="0.100"
            step="0.005"
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            className="w-full accent-amber-500 cursor-pointer"
          />
        </div>

        {/* Launch Button Action */}
        <div className="rounded border border-[#1f293d] bg-[#090d16] p-4 flex flex-col justify-between">
          <div>
            <div className="text-xs text-zinc-400 font-bold uppercase mb-1">EXECUTION ENGINE</div>
            <p className="text-[11px] text-zinc-500">
              Evaluates baseline malicious packet followed by stealth perturbation.
            </p>
          </div>
          <button
            onClick={handleLaunch}
            disabled={loading}
            className={`w-full py-2.5 px-4 rounded text-xs font-bold uppercase tracking-wider transition-all shadow-lg ${
              loading
                ? "bg-red-950 text-red-300 border border-red-800 cursor-wait"
                : "bg-red-600 hover:bg-red-500 text-white shadow-red-600/30 active:scale-95"
            }`}
          >
            {loading ? "INJECTING PERTURBATION..." : "LAUNCH ADVERSARIAL WAVE"}
          </button>
        </div>
      </div>

      {/* Side-by-Side Verdict Display */}
      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Clean Attack Card */}
            <div className="rounded border border-[#1f293d] bg-[#090d16] p-5">
              <div className="flex items-center justify-between border-b border-[#1f293d] pb-3">
                <span className="text-xs font-bold text-zinc-300 uppercase">
                  STEP 1: UNMODIFIED MALICIOUS PACKET
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/30">
                  CLEAN ATTACK
                </span>
              </div>
              <div className="mt-4 flex items-baseline justify-between">
                <div>
                  <div className="text-[11px] text-zinc-500 uppercase">DETECTION CONFIDENCE</div>
                  <div className="text-3xl font-bold text-red-400 mt-1">
                    {(((result.step_clean?.confidence ?? 0.998)) * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[11px] text-zinc-500 uppercase">NIDS VERDICT</div>
                  <div className="text-lg font-bold text-red-400 mt-1 flex items-center gap-1 justify-end">
                    <ShieldAlert className="h-4 w-4" />
                    {result.step_clean?.prediction ?? "MALICIOUS"}
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-[#1f293d]/50 flex justify-between text-xs text-zinc-400">
                <span>THRESHOLD: τ={(result.threshold ?? 0.88).toFixed(3)}</span>
                <span>LATENCY: {(result.step_clean?.latency_ms ?? 1.42).toFixed(2)}ms</span>
              </div>
            </div>

            {/* Adversarial Variant Card */}
            <div className="rounded border border-[#1f293d] bg-[#090d16] p-5">
              <div className="flex items-center justify-between border-b border-[#1f293d] pb-3">
                <span className="text-xs font-bold text-zinc-300 uppercase">
                  STEP 2: PERTURBED ADVERSARIAL VARIANT (ε={(result.epsilon ?? 0.05).toFixed(3)})
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    result.evasion_successful
                      ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                      : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                  }`}
                >
                  {result.evasion_successful ? "EVASION SUCCESSFUL" : "EVASION FAILED"}
                </span>
              </div>
              <div className="mt-4 flex items-baseline justify-between">
                <div>
                  <div className="text-[11px] text-zinc-500 uppercase">DETECTION CONFIDENCE</div>
                  <div
                    className={`text-3xl font-bold mt-1 ${
                      (result.step_adversarial?.confidence ?? 0.014) < (result.threshold ?? 0.88)
                        ? "text-emerald-400"
                        : "text-red-400"
                    }`}
                  >
                    {(((result.step_adversarial?.confidence ?? 0.014)) * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[11px] text-zinc-500 uppercase">NIDS VERDICT</div>
                  <div
                    className={`text-lg font-bold mt-1 flex items-center gap-1 justify-end ${
                      (result.step_adversarial?.confidence ?? 0.014) < (result.threshold ?? 0.88)
                        ? "text-emerald-400"
                        : "text-red-400"
                    }`}
                  >
                    {(result.step_adversarial?.confidence ?? 0.014) < (result.threshold ?? 0.88) ? (
                      <>
                        <CheckCircle2 className="h-4 w-4" />
                        BENIGN (EVADED)
                      </>
                    ) : (
                      <>
                        <ShieldAlert className="h-4 w-4" />
                        FLAGGED (DEFENSE HELD)
                      </>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-[#1f293d]/50 flex justify-between text-xs text-zinc-400">
                <span>THRESHOLD: τ={(result.threshold ?? 0.88).toFixed(3)}</span>
                <span>LATENCY: {(result.step_adversarial?.latency_ms ?? 1.55).toFixed(2)}ms</span>
              </div>
            </div>
          </div>

          {/* Feature Diff Table */}
          <div className="rounded border border-[#1f293d] bg-[#090d16] overflow-hidden">
            <div className="border-b border-[#1f293d] px-4 py-3 bg-black/40 flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">
                PROBLEM-SPACE FEATURE PERTURBATION DIFF (TOP 5 SCALED FEATURES)
              </span>
              <span className="text-[11px] text-zinc-500">
                ATTACK METHOD: {result.mode.toUpperCase()} CONSTRAINED
              </span>
            </div>
            <table className="w-full text-left text-xs">
              <thead className="bg-[#0d1322] text-zinc-400 border-b border-[#1f293d]">
                <tr>
                  <th className="py-2.5 px-4 font-semibold">FEATURE NAME</th>
                  <th className="py-2.5 px-4 font-semibold">CLEAN VALUE</th>
                  <th className="py-2.5 px-4 font-semibold">ADVERSARIAL VALUE</th>
                  <th className="py-2.5 px-4 font-semibold">DELTA (Δ)</th>
                  <th className="py-2.5 px-4 font-semibold">CONTROLLABILITY</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f293d]/50 text-zinc-300">
                {result.feature_diffs.map((diff, idx) => (
                  <tr key={idx} className="hover:bg-zinc-900/40">
                    <td className="py-2.5 px-4 font-semibold text-zinc-200">{diff.feature_name}</td>
                    <td className="py-2.5 px-4 text-zinc-400">{diff.original_value.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-amber-300">{diff.perturbed_value.toFixed(4)}</td>
                    <td className="py-2.5 px-4">
                      <span className={diff.delta >= 0 ? "text-amber-400 font-bold" : "text-cyan-400 font-bold"}>
                        {diff.delta >= 0 ? `+${diff.delta.toFixed(4)}` : diff.delta.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2.5 px-4">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                          diff.is_controllable
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                            : "bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {diff.is_controllable ? "CONTROLLABLE" : "NON-MUTABLE"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
