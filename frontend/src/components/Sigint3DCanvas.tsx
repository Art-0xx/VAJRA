"use client";

import React, { useRef, useEffect } from "react";

interface Sigint3DCanvasProps {
  hasInjectedFlow?: boolean;
  flowState?: "IDLE" | "BENIGN" | "ATTACK" | "DEFENDED";
  isAttacking?: boolean;
  isDefended?: boolean;
  attackMethod?: string;
  epsilon?: number;
  activeTau?: number;
  cleanProb?: number;
  advProb?: number;
  divergenceScore?: number;
  latencyMs?: number;
}

export const Sigint3DCanvas: React.FC<Sigint3DCanvasProps> = ({
  hasInjectedFlow = false,
  flowState = "IDLE",
  isAttacking = false,
  isDefended = false,
  attackMethod = "DDoS-LOIC-HTTP",
  epsilon = 0.05,
  activeTau = 0.88,
  cleanProb = 0.992,
  advProb = 0.014,
  divergenceScore = 0.168,
  latencyMs = 1.42,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = 0;
    let height = 0;

    const resize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.parentElement?.clientWidth || 1080;
      height = canvas.height = 745;
    };
    resize();
    window.addEventListener("resize", resize);

    let angleX = 0.38;
    let angleY = 0.0;
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    };
    const onMouseUp = () => {
      isDragging = false;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        angleY += (e.clientX - lastX) * 0.005;
        angleX = Math.max(0.15, Math.min(0.85, angleX + (e.clientY - lastY) * 0.004));
        lastX = e.clientX;
        lastY = e.clientY;
      }
    };

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mousemove", onMouseMove);

    let tick = 0;
    let currentHeightScale = hasInjectedFlow ? 1.0 : 0.05;
    const ROWS = 56;
    const COLS = 82;
    const stepX = 16.0;
    const stepZ = 7.5;

    function getElevation(r: number, c: number, t: number, scaleFactor: number) {
      const x = (c - COLS / 2) / (COLS / 2);
      const z = (r - ROWS / 2) / (ROWS / 2);

      const envelope = Math.exp(-Math.pow(x * 1.65, 2) - Math.pow(z * 1.45, 2));

      const p1x = x - 0.02, p1z = z + 0.05;
      const spire = Math.exp(-(p1x * p1x * 24.0 + p1z * p1z * 18.0)) * 2.85;

      const r1 = Math.exp(-(Math.pow(x + 0.26, 2) * 12.0 + Math.pow(z + 0.1, 2) * 10.0)) * 1.65;
      const r2 = Math.exp(-(Math.pow(x - 0.28, 2) * 14.0 + Math.pow(z - 0.04, 2) * 11.0)) * 1.5;
      const r3 = Math.exp(-(Math.pow(x + 0.48, 2) * 16.0 + Math.pow(z + 0.15, 2) * 12.0)) * 1.15;
      const r4 = Math.exp(-(Math.pow(x - 0.48, 2) * 18.0 + Math.pow(z - 0.08, 2) * 12.0)) * 1.05;
      const r5 = Math.exp(-(Math.pow(x + 0.68, 2) * 22.0 + Math.pow(z + 0.2, 2) * 14.0)) * 0.7;
      const r6 = Math.exp(-(Math.pow(x - 0.68, 2) * 22.0 + Math.pow(z - 0.1, 2) * 14.0)) * 0.65;

      const crags =
        Math.cos(x * 18 + z * 10) * Math.sin(z * 22) * 0.4 +
        Math.cos(x * 36) * Math.sin(x * 18 + z * 28) * 0.25 +
        Math.sin(x * 54 + t * 0.03) * Math.cos(z * 36) * 0.08;

      let h = (spire + r1 + r2 + r3 + r4 + r5 + r6 + crags) * envelope;

      if (isAttacking) {
        const attackSpires = Math.exp(-Math.pow(x * 4.0, 2)) * Math.sin(x * 24 + t * 0.14) * 0.85;
        h = h * 1.25 + attackSpires * envelope;
      }

      // Idle baseline scan ripple
      const idleRipple = (Math.sin(x * 10 + t * 0.04) * Math.cos(z * 8 + t * 0.03) * 0.08 + 0.03);
      const blended = h * scaleFactor + idleRipple * (1.0 - scaleFactor);

      return Math.max(0, blended) * 128;
    }

    function project3D(x: number, y: number, z: number) {
      const cosY = Math.cos(angleY), sinY = Math.sin(angleY);
      const cosX = Math.cos(angleX), sinX = Math.sin(angleX);

      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;

      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      const fov = 580;
      const scale = fov / (fov + z2 + 380);
      return {
        x: width / 2 + x1 * scale,
        y: 405 + y1 * scale,
        scale,
        depth: z2,
      };
    }

    const render = () => {
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, width, height);

      tick++;
      if (!isDragging) angleY += 0.0008;

      const targetScale = !hasInjectedFlow || flowState === "IDLE" ? 0.08 : flowState === "BENIGN" ? 0.45 : 1.0;
      currentHeightScale += (targetScale - currentHeightScale) * 0.08;

      // -------------------------------------------------------------
      // ZONE 1: TOP 3D DATA SURFACE TOPOLOGY (Y: 0 to 455)
      // -------------------------------------------------------------
      const grid: Array<Array<{ x: number; y: number; scale: number; depth: number; elev: number }>> = [];
      for (let r = 0; r < ROWS; r++) {
        grid[r] = [];
        for (let c = 0; c < COLS; c++) {
          const wx = (c - COLS / 2) * stepX;
          const wz = (r - ROWS / 2) * stepZ;
          const elev = getElevation(r, c, tick, currentHeightScale);
          const wy = -elev;
          grid[r][c] = { ...project3D(wx, wy, wz), elev };
        }
      }

      // Draw Wireframe Isolines (Joy Division style hidden surface occlusion)
      for (let r = 0; r < ROWS; r++) {
        let maxElevInRow = 0;
        for (let c = 0; c < COLS; c++) {
          if (grid[r][c].elev > maxElevInRow) maxElevInRow = grid[r][c].elev;
        }
        if (maxElevInRow < 1.0) continue;

        let firstC = -1, lastC = -1;
        for (let c = 0; c < COLS; c++) {
          if (grid[r][c].elev > 0.5) {
            if (firstC === -1) firstC = Math.max(0, c - 1);
            lastC = Math.min(COLS - 1, c + 1);
          }
        }

        if (firstC !== -1 && lastC > firstC) {
          const depthRatio = r / ROWS;
          const strokeAlpha = 0.35 + depthRatio * 0.65;

          // Occlusion Fill Underneath
          ctx.beginPath();
          for (let c = firstC; c <= lastC; c++) {
            const p = grid[r][c];
            if (c === firstC) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
          }
          ctx.lineTo(grid[r][lastC].x, 435);
          ctx.lineTo(grid[r][firstC].x, 435);
          ctx.closePath();
          ctx.fillStyle = "#000000";
          ctx.fill();

          // Stroke Ridge Contour
          ctx.beginPath();
          for (let c = firstC; c <= lastC; c++) {
            const p = grid[r][c];
            if (c === firstC) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
          }
          ctx.strokeStyle = `rgba(240, 245, 250, ${strokeAlpha})`;
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }
      }

      // Central Vertical Comb / Waterfall Lines
      const combStart = Math.floor(COLS * 0.46);
      const combEnd = Math.floor(COLS * 0.56);
      for (let c = combStart; c <= combEnd; c++) {
        let minPy = 999;
        let topX = 0;
        for (let r = 0; r < ROWS; r++) {
          if (grid[r][c].y < minPy && grid[r][c].elev > 30) {
            minPy = grid[r][c].y;
            topX = grid[r][c].x;
          }
        }
        if (minPy < 420) {
          ctx.beginPath();
          ctx.moveTo(topX, minPy);
          ctx.lineTo(topX, 428);
          ctx.strokeStyle = isAttacking
            ? "rgba(255, 75, 75, 0.45)"
            : isDefended
            ? "rgba(52, 211, 153, 0.45)"
            : "rgba(255, 255, 255, 0.38)";
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }

      // Stippled Point Cloud
      for (let r = 2; r < ROWS - 4; r++) {
        for (let c = 4; c < COLS - 4; c++) {
          const p = grid[r][c];
          if (p.elev > 14) {
            const prob = Math.min(0.85, (p.elev / 120) * 0.9);
            if (Math.sin(r * 13 + c * 37 + tick * 0.01) > 1.0 - prob) {
              const isPeakAlert = isAttacking && p.elev > 95;
              ctx.fillStyle = isPeakAlert ? "#ff3b3b" : isDefended && p.elev > 80 ? "#34d399" : "rgba(255, 255, 255, 0.85)";
              ctx.fillRect(p.x, p.y, 1.2, 1.2);
            }
          }
        }
      }

      // Secondary Rolling Hills
      ctx.beginPath();
      ctx.strokeStyle = "rgba(180, 185, 195, 0.45)";
      ctx.lineWidth = 0.9;
      for (let x = 60; x < width - 60; x += 3) {
        const nx = (x - width / 2) / 380;
        const h1 = Math.exp(-nx * nx * 4.0) * Math.sin(nx * 14 + tick * 0.02) * 32;
        const h2 = Math.exp(-Math.pow(nx + 0.4, 2) * 8.0) * 24;
        const h3 = Math.exp(-Math.pow(nx - 0.45, 2) * 10.0) * 28;
        const hy = 432 - Math.max(0, h1 + h2 + h3);
        if (x === 60) ctx.moveTo(x, hy);
        else ctx.lineTo(x, hy);
      }
      ctx.stroke();

      // -------------------------------------------------------------
      // TACTICAL TELEMETRY HUD OVERLAYS
      // -------------------------------------------------------------
      ctx.font = '10px "JetBrains Mono", "Courier New", monospace';
      ctx.textBaseline = "top";

      // Top Left
      ctx.strokeStyle = "#3f3f46";
      ctx.strokeRect(32, 22, 96, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText("0.11867167", 38, 26);
      ctx.fillStyle = "#71717a";
      ctx.fillText("* 2111 301", 136, 26);
      ctx.fillText("31.472... 31.1110", 32, 46);
      ctx.fillText("SPECTRAL TOPOGRAPHY MANIFOLD // CIC-IDS2017", 32, 60);
      const safeTau = activeTau ?? 0.88;
      const safeEps = epsilon ?? 0.05;
      const safeDiv = divergenceScore ?? 0.168;
      const safeLat = latencyMs ?? 1.42;

      ctx.fillText(`COORDINATES: [0x4E7A] L4_ATTN_ROLLOUT (JSD: ${safeDiv.toFixed(3)})`, 32, 74);

      // Top Right
      ctx.strokeStyle = isAttacking ? "#ff3b3b" : isDefended ? "#34d399" : "#3f3f46";
      ctx.strokeRect(width - 240, 22, 208, 18);
      ctx.fillStyle = isAttacking ? "#ff3b3b" : isDefended ? "#34d399" : "#ffffff";
      ctx.fillText(
        isDefended ? "DEFENDED // PGD-AT HARDENED" : "CLASSIFIED // SIGINT -8/7C6",
        width - 232,
        26
      );

      const currentElevM = Math.round(2840 * currentHeightScale);
      ctx.fillStyle = "#71717a";
      ctx.fillText(`ELEVATION: ${currentElevM}m | GRID: 48°11'N 12°04'E`, width - 240, 46);

      const statusText = currentHeightScale < 0.20
        ? "STATUS: [*] SENSOR STANDBY // IDLE"
        : isAttacking
        ? "STATUS: [!] ADVERSARIAL DISRUPTION"
        : isDefended
        ? "STATUS: [*] CONSENSUS MANIFOLD (DEFENDED)"
        : "STATUS: [*] BENIGN FLOW INGESTED";

      ctx.fillText(statusText, width - 240, 60);

      // Centered Standby Indicator
      if (currentHeightScale < 0.20) {
        ctx.fillStyle = "rgba(56, 189, 248, 0.9)";
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.textAlign = "center";
        ctx.fillText("STANDBY SENSOR MATRIX // WAITING FOR PACKET INGESTION", width / 2, 225);
        ctx.fillStyle = "rgba(161, 161, 170, 0.85)";
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillText("Click [INJECT BENIGN FLOW] or [INJECT ATTACK FLOW] above to generate 3D topology", width / 2, 245);
        ctx.textAlign = "start";
      }

      // Tabular Hex Register Block
      const hexRows = [
        ["Mäzenin...", "+0.072..."],
        ["10.1...", "-8/7C6"],
        ["350 185", "12%7513G7"],
        ["0191 138", isDefended ? "STABLE_AT" : "STABLE_BASE"],
        ["10553176", `L-INF ε=${safeEps.toFixed(3)}`],
        ["32KB.138", isAttacking ? "ATTN_HIJACK" : "ATTN_SYNCH"],
        ["1UB/L52", `TAU=${safeTau.toFixed(3)}`],
      ];
      for (let i = 0; i < hexRows.length; i++) {
        ctx.fillStyle = "#a1a1aa";
        ctx.fillText(hexRows[i][0], width - 240, 82 + i * 14);
        ctx.fillStyle = isAttacking && i === 5 ? "#ff3b3b" : isDefended && i === 3 ? "#34d399" : "#52525b";
        ctx.fillText(hexRows[i][1], width - 150, 82 + i * 14);
      }

      // Mid Left Telemetry
      ctx.font = 'bold 22px "JetBrains Mono", "Courier New", monospace';
      ctx.fillStyle = "#ffffff";
      ctx.fillText("+268", 32, 362);
      ctx.font = '9px "JetBrains Mono", "Courier New", monospace';
      ctx.fillStyle = "#71717a";
      ctx.fillText("ATTENTION DISTRIBUTION GRADIENT", 32, 388);
      ctx.fillText("FEATURE MANIFOLD: 77-D -> 3D TOPOLOGY", 32, 402);

      // Mid Center Annotations
      ctx.fillText("520 8332", width / 2 - 170, 396);
      ctx.fillText("G3170", width / 2 + 10, 396);
      ctx.fillText("9153128170", width / 2 + 180, 396);
      ctx.fillText("Moteur 56100238", width / 2 + 180, 410);

      // Mid Right Annotations
      ctx.fillText(`VECTOR: ${attackMethod} (ε=${safeEps.toFixed(3)})`, width - 240, 388);
      ctx.fillText(`LATENCY: ${safeLat.toFixed(2)}ms | DRAG MOUSE TO ROTATE`, width - 240, 402);

      // -------------------------------------------------------------
      // DIVIDER BAND: ZONE 1 / ZONE 2 BOUNDARY
      // -------------------------------------------------------------
      ctx.strokeStyle = "#27272a";
      ctx.beginPath();
      ctx.moveTo(24, 452);
      ctx.lineTo(width - 24, 452);
      ctx.stroke();

      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = "#ffffff";
      ctx.fillText("ZONE 2: MULTI-LAYERED TIME-SERIES ANALYSIS MATRIX // NETWORK FLUCTUATIONS", 32, 464);
      ctx.fillStyle = "#71717a";
      ctx.fillText(
        "LAYER 1: SPECTRAL NEEDLE DENSITY (LEFT)  |  LAYER 2: MULTI-TRACK OSCILLOGRAM (RIGHT)",
        Math.max(32, width - 620),
        464
      );

      // -------------------------------------------------------------
      // ZONE 2: BOTTOM MULTI-LAYERED TIME-SERIES MATRIX (Y: 485 to 740)
      // -------------------------------------------------------------
      const needleBaseY = 720;
      const needleMaxH = 205;
      const needleCount = 82;
      const needleRight = width * 0.44;
      const needleStep = (needleRight - 32) / needleCount;

      const needlePeaks: Array<{ x: number; y: number; val: number; isAttackNeedle: boolean }> = [];
      for (let i = 0; i < needleCount; i++) {
        const nx = i / needleCount;
        let val =
          Math.exp(-Math.pow((nx - 0.44) * 3.8, 2)) * 0.82 +
          Math.exp(-Math.pow((nx - 0.72) * 5.5, 2)) * 0.62 +
          Math.sin(i * 1.5) * 0.14 +
          Math.sin(i * 0.8 + tick * 0.05) * 0.06;

        const isAttackNeedle = isAttacking && i >= 32 && i <= 44;
        if (isAttackNeedle) {
          val = 0.94 + Math.sin(i * 3.0) * 0.05;
        }
        val = Math.max(0.04, Math.min(1.0, val));
        const nh = val * needleMaxH;
        const xpos = 32 + i * needleStep;
        const ypos = needleBaseY - nh;

        needlePeaks.push({ x: xpos, y: ypos, val, isAttackNeedle });

        ctx.beginPath();
        ctx.moveTo(xpos, needleBaseY);
        ctx.lineTo(xpos, ypos);
        ctx.strokeStyle = isAttackNeedle ? "#ff3b3b" : "rgba(215, 220, 225, 0.4)";
        ctx.lineWidth = 1.0;
        ctx.stroke();

        ctx.fillStyle = isAttackNeedle ? "#ff3b3b" : "#ffffff";
        ctx.fillRect(xpos - 1, ypos - 1, 2, 2);
      }

      // Envelope line
      ctx.beginPath();
      for (let i = 0; i < needlePeaks.length; i++) {
        if (i === 0) ctx.moveTo(needlePeaks[i].x, needlePeaks[i].y);
        else ctx.lineTo(needlePeaks[i].x, needlePeaks[i].y);
      }
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.1;
      ctx.stroke();

      // Threshold line
      const threshY = needleBaseY - safeTau * needleMaxH;
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(32, threshY);
      ctx.lineTo(needleRight, threshY);
      ctx.strokeStyle = "#ff3b3b";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ff3b3b";
      ctx.fillText(`TAU_THRESH: ${safeTau.toFixed(3)}`, 34, threshY - 12);

      // Subzone B: Multi-Track Network Oscillogram (Right)
      const oscLeft = width * 0.49;
      const oscRight = width - 32;
      const oscW = oscRight - oscLeft;

      const yLabels = ["+40", "+20", "0", "-20", "-40", "-60", "-80", "-100", "-120"];
      const oscTop = 505;
      const oscBottom = 720;
      const yStep = (oscBottom - oscTop) / (yLabels.length - 1);

      for (let i = 0; i < yLabels.length; i++) {
        const py = oscTop + i * yStep;
        ctx.fillStyle = "#52525b";
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText(yLabels[i], oscLeft - 32, py - 4);

        ctx.beginPath();
        ctx.setLineDash([2, 5]);
        ctx.moveTo(oscLeft, py);
        ctx.lineTo(oscRight, py);
        ctx.strokeStyle = "#18181b";
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Track 1: Clean Baseline Fluctuation
      ctx.beginPath();
      for (let x = 0; x < oscW; x += 4) {
        const tVal = (x / oscW) * 8.0 + tick * 0.04;
        const noise = Math.sin(tVal) * 14 + Math.sin(tVal * 2.3) * 8 + Math.cos(tVal * 0.7) * 6;
        const py = oscTop + yStep * 2 + noise;
        if (x === 0) ctx.moveTo(oscLeft + x, py);
        else ctx.lineTo(oscLeft + x, py);
      }
      ctx.strokeStyle = "rgba(100, 116, 139, 0.75)";
      ctx.lineWidth = 1.0;
      ctx.stroke();

      // Track 2: Adversarial Perturbation Signal
      ctx.beginPath();
      for (let x = 0; x < oscW; x += 4) {
        const tVal = (x / oscW) * 12.0 + tick * 0.06;
        const amp = isAttacking ? 28 : 10;
        const noise = Math.sin(tVal * 1.8) * amp + Math.cos(tVal * 4.2) * (amp * 0.4);
        const py = oscTop + yStep * 5 + noise;
        if (x === 0) ctx.moveTo(oscLeft + x, py);
        else ctx.lineTo(oscLeft + x, py);
      }
      ctx.strokeStyle = isAttacking ? "#ef4444" : "rgba(148, 163, 184, 0.5)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Track 3: Defended Consensus Filter
      ctx.beginPath();
      for (let x = 0; x < oscW; x += 4) {
        const tVal = (x / oscW) * 5.0 + tick * 0.02;
        const noise = Math.cos(tVal * 1.2) * 9 + Math.sin(tVal * 0.4) * 11;
        const py = oscTop + yStep * 7.2 + noise;
        if (x === 0) ctx.moveTo(oscLeft + x, py);
        else ctx.lineTo(oscLeft + x, py);
      }
      ctx.strokeStyle = isDefended ? "#10b981" : "rgba(71, 85, 105, 0.5)";
      ctx.lineWidth = 1.0;
      ctx.stroke();

      // Track Labels
      ctx.fillStyle = "#94a3b8";
      ctx.fillText("[TRK-01] INCOMING CLEAN PACKET STREAM", oscLeft + 10, oscTop + yStep * 2 - 10);
      ctx.fillStyle = isAttacking ? "#ef4444" : "#64748b";
      ctx.fillText("[TRK-02] PROBLEM-SPACE PERTURBATION DELTA", oscLeft + 10, oscTop + yStep * 5 - 10);
      ctx.fillStyle = isDefended ? "#10b981" : "#475569";
      ctx.fillText("[TRK-03] PGD-AT HARDENED RESIDUAL CONSENSUS", oscLeft + 10, oscTop + yStep * 7.2 - 10);

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("mousemove", onMouseMove);
    };
  }, [hasInjectedFlow, flowState, isAttacking, isDefended, attackMethod, epsilon, activeTau, cleanProb, advProb, divergenceScore, latencyMs]);

  return (
    <div className="relative w-full rounded border border-[#1f293d] bg-black shadow-2xl overflow-hidden">
      <canvas ref={canvasRef} className="block w-full h-[745px] cursor-grab active:cursor-grabbing" />
      <div className="absolute top-2 right-4 flex items-center gap-2 pointer-events-none">
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        <span className="text-[10px] font-mono tracking-wider text-zinc-400 uppercase">
          LIVE TACTICAL STREAM
        </span>
      </div>
    </div>
  );
};
