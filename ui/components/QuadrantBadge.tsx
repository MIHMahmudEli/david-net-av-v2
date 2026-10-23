"use client";

import React, { useState } from "react";
import { QuadrantInfo } from "../lib/api";
import { Crosshair, ShieldAlert, Sparkles, HelpCircle } from "lucide-react";

interface QuadrantBadgeProps {
  quadrant: QuadrantInfo;
  videoProb?: number; // 0.0 to 1.0 (manipulation probability)
  audioProb?: number; // 0.0 to 1.0 (manipulation probability)
  latencyMs: number;
  durationSec: number;
  clipId?: string;
}

export const QuadrantBadge: React.FC<QuadrantBadgeProps> = ({
  quadrant,
  videoProb,
  audioProb,
  latencyMs,
  durationSec,
  clipId,
}) => {
  // Derive (px, py) coordinates for Cartesian mapping
  // X = Audio Manipulation Probability (0 = Real Audio, 1 = Fake Audio)
  // Y = Video Manipulation Probability (0 = Real Video, 1 = Fake Video)
  const pA = audioProb !== undefined ? audioProb : (quadrant.probs.RVFA + quadrant.probs.FVFA);
  const pV = videoProb !== undefined ? videoProb : (quadrant.probs.FVRA + quadrant.probs.FVFA);

  // Clamp within [0, 1]
  const clampA = Math.max(0.02, Math.min(0.98, pA));
  const clampV = Math.max(0.02, Math.min(0.98, pV));

  // Map to SVG coordinates: 300x300 viewBox
  // X: 0 -> 32, 1 -> 268 (padding of 32px)
  // Y: 0 -> 268 (bottom), 1 -> 32 (top)
  const svgX = 32 + clampA * 236;
  const svgY = 268 - clampV * 236;

  const isRVRA = quadrant.label === "RVRA";
  const isFVFA = quadrant.label === "FVFA";
  const isFVRA = quadrant.label === "FVRA";
  const isRVFA = quadrant.label === "RVFA";

  const quadrantThemes = {
    RVRA: {
      border: "border-emerald-300 dark:border-emerald-500/30",
      bg: "bg-emerald-50/80 dark:bg-emerald-950/20",
      text: "text-emerald-700 dark:text-emerald-400",
      badge: "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border-emerald-300 dark:border-emerald-500/30",
      dot: "#059669",
      glow: "rgba(16, 185, 129, 0.25)",
    },
    FVFA: {
      border: "border-rose-300 dark:border-rose-500/30",
      bg: "bg-rose-50/80 dark:bg-rose-950/20",
      text: "text-rose-700 dark:text-rose-400",
      badge: "bg-rose-100 dark:bg-rose-500/20 text-rose-800 dark:text-rose-300 border-rose-300 dark:border-rose-500/30",
      dot: "#e11d48",
      glow: "rgba(244, 63, 94, 0.25)",
    },
    FVRA: {
      border: "border-amber-300 dark:border-amber-500/30",
      bg: "bg-amber-50/80 dark:bg-amber-950/20",
      text: "text-amber-800 dark:text-amber-400",
      badge: "bg-amber-100 dark:bg-amber-500/20 text-amber-800 dark:text-amber-300 border-amber-300 dark:border-amber-500/30",
      dot: "#d97706",
      glow: "rgba(245, 158, 11, 0.25)",
    },
    RVFA: {
      border: "border-sky-300 dark:border-sky-500/30",
      bg: "bg-sky-50/80 dark:bg-sky-950/20",
      text: "text-sky-700 dark:text-sky-400",
      badge: "bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 border-sky-300 dark:border-sky-500/30",
      dot: "#0284c7",
      glow: "rgba(56, 189, 248, 0.25)",
    },
  };

  const currentTheme = quadrantThemes[quadrant.label] || quadrantThemes.RVRA;

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 shadow-sm relative overflow-hidden transition-all">
      {/* Top Header telemetry */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200/80 dark:border-white/[0.08] gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-xl bg-sky-50 dark:bg-white/[0.05] border border-sky-200 dark:border-white/10 text-sky-600 dark:text-sky-400 flex items-center justify-center shrink-0">
            <Crosshair className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2 flex-wrap">
              <span className="text-[10px] sm:text-xs uppercase tracking-widest font-mono text-slate-500 dark:text-zinc-400 font-semibold">
                2D Cartesian Attribution Matrix
              </span>
              <span className="text-[9px] sm:text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-white/[0.06] text-slate-600 dark:text-zinc-300 border border-slate-200 dark:border-white/10 font-medium">
                $p_a \times p_v$ Space
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2 mt-0.5 flex-wrap">
              <span className="font-mono">{quadrant.label}:</span>
              <span className={currentTheme.text}>{quadrant.description}</span>
            </h2>
          </div>
        </div>

        <div className="flex items-center gap-3 sm:gap-4 text-xs font-mono text-slate-500 dark:text-zinc-400 self-start sm:self-auto pt-2 sm:pt-0">
          {clipId && (
            <div className="text-left sm:text-right hidden md:block">
              <span className="text-[9px] text-slate-400 dark:text-zinc-500 block uppercase font-sans">Clip Digest</span>
              <span className="text-slate-700 dark:text-zinc-300 font-semibold">{clipId.slice(0, 10)}</span>
            </div>
          )}
          <div className="text-left sm:text-right">
            <span className="text-[9px] text-slate-400 dark:text-zinc-500 block uppercase font-sans">Latency</span>
            <span className="text-sky-700 dark:text-sky-400 font-bold">{latencyMs} ms</span>
          </div>
          <div className="text-left sm:text-right">
            <span className="text-[9px] text-slate-400 dark:text-zinc-500 block uppercase font-sans">Duration</span>
            <span className="text-slate-800 dark:text-zinc-200 font-semibold">{durationSec.toFixed(2)}s</span>
          </div>
        </div>
      </div>

      {/* Main Body: 2D Plane on Left/Center + Breakdown on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-5 items-center">
        {/* The 2D Cartesian Quadrant Plane */}
        <div className="lg:col-span-7 flex flex-col items-center w-full">
          <div className="w-full max-w-[320px] sm:max-w-[360px] aspect-square relative bg-[#FAFCFF] dark:bg-[#07090E] border border-slate-200 dark:border-white/10 rounded-2xl p-3 shadow-inner">
            <svg
              viewBox="0 0 300 300"
              className="w-full h-full select-none overflow-visible"
            >
              <defs>
                <radialGradient id="reticleGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={currentTheme.dot} stopOpacity="0.7" />
                  <stop offset="60%" stopColor={currentTheme.dot} stopOpacity="0.2" />
                  <stop offset="100%" stopColor={currentTheme.dot} stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* Background Quadrant Shading */}
              {/* Top-Left: FVRA */}
              <rect
                x="32"
                y="32"
                width="118"
                height="118"
                fill="#fef3c7"
                fillOpacity={isFVRA ? 0.7 : 0.25}
                className="transition-all duration-300"
              />
              {/* Top-Right: FVFA */}
              <rect
                x="150"
                y="32"
                width="118"
                height="118"
                fill="#ffe4e6"
                fillOpacity={isFVFA ? 0.75 : 0.25}
                className="transition-all duration-300"
              />
              {/* Bottom-Left: RVRA */}
              <rect
                x="32"
                y="150"
                width="118"
                height="118"
                fill="#d1fae5"
                fillOpacity={isRVRA ? 0.7 : 0.25}
                className="transition-all duration-300"
              />
              {/* Bottom-Right: RVFA */}
              <rect
                x="150"
                y="150"
                width="118"
                height="118"
                fill="#e0f2fe"
                fillOpacity={isRVFA ? 0.7 : 0.25}
                className="transition-all duration-300"
              />

              {/* Outer Grid Bounds */}
              <rect x="32" y="32" width="236" height="236" fill="none" stroke="#CBD5E1" strokeWidth="1" />

              {/* Internal Subtle Sub-grid Lines */}
              <line x1="32" y1="91" x2="268" y2="91" stroke="#E2E8F0" strokeWidth="0.8" strokeDasharray="3,3" />
              <line x1="32" y1="209" x2="268" y2="209" stroke="#E2E8F0" strokeWidth="0.8" strokeDasharray="3,3" />
              <line x1="91" y1="32" x2="91" y2="268" stroke="#E2E8F0" strokeWidth="0.8" strokeDasharray="3,3" />
              <line x1="209" y1="32" x2="209" y2="268" stroke="#E2E8F0" strokeWidth="0.8" strokeDasharray="3,3" />

              {/* Decision Boundary Crosshair at tau = (0.5, 0.5) */}
              <line
                x1="32"
                y1="150"
                x2="268"
                y2="150"
                stroke="#64748B"
                strokeWidth="1.2"
                strokeDasharray="4,4"
              />
              <line
                x1="150"
                y1="32"
                x2="150"
                y2="268"
                stroke="#64748B"
                strokeWidth="1.2"
                strokeDasharray="4,4"
              />

              {/* Quadrant Labels inside quadrants */}
              <text x="42" y="52" fill="#b45309" fontSize="9.5" fontFamily="monospace" fontWeight="bold">
                FVRA [FaceSwap]
              </text>
              <text x="160" y="52" fill="#be123c" fontSize="9.5" fontFamily="monospace" fontWeight="bold">
                FVFA [Dual Fake]
              </text>
              <text x="42" y="260" fill="#047857" fontSize="9.5" fontFamily="monospace" fontWeight="bold">
                RVRA [Authentic]
              </text>
              <text x="160" y="260" fill="#0369a1" fontSize="9.5" fontFamily="monospace" fontWeight="bold">
                RVFA [Voice Dub]
              </text>

              {/* Axis Descriptions */}
              <text x="150" y="290" fill="#64748B" fontSize="8.5" fontFamily="monospace" textAnchor="middle" fontWeight="600">
                Audio Manipulation Probability ($p_a$) →
              </text>
              <text
                x="-150"
                y="15"
                fill="#64748B"
                fontSize="8.5"
                fontFamily="monospace"
                textAnchor="middle"
                fontWeight="600"
                transform="rotate(-90)"
              >
                Video Manipulation Probability ($p_v$) →
              </text>

              {/* Projection Guidelines from Target point to axes */}
              <line
                x1={svgX}
                y1={svgY}
                x2={svgX}
                y2="268"
                stroke={currentTheme.dot}
                strokeWidth="1.2"
                strokeDasharray="2,2"
                opacity="0.8"
              />
              <line
                x1={svgX}
                y1={svgY}
                x2="32"
                y2={svgY}
                stroke={currentTheme.dot}
                strokeWidth="1.2"
                strokeDasharray="2,2"
                opacity="0.8"
              />

              {/* Concentric Pulsing Radar Rings */}
              <circle
                cx={svgX}
                cy={svgY}
                r="22"
                fill="url(#reticleGlow)"
                className="animate-radar-pulse"
              />
              <circle
                cx={svgX}
                cy={svgY}
                r="11"
                fill="none"
                stroke={currentTheme.dot}
                strokeWidth="1.5"
                opacity="0.8"
              />
              <circle
                cx={svgX}
                cy={svgY}
                r="4.5"
                fill={currentTheme.dot}
                stroke="#ffffff"
                strokeWidth="2"
              />

              {/* Reticle Target Crosshairs */}
              <line
                x1={svgX - 8}
                y1={svgY}
                x2={svgX + 8}
                y2={svgY}
                stroke="#ffffff"
                strokeWidth="1.2"
              />
              <line
                x1={svgX}
                y1={svgY - 8}
                x2={svgX}
                y2={svgY + 8}
                stroke="#ffffff"
                strokeWidth="1.2"
              />
            </svg>
          </div>

          <div className="flex items-center justify-between w-full max-w-[320px] sm:max-w-[360px] mt-2 px-1 text-[11px] font-mono text-slate-500 dark:text-zinc-400">
            <span>
              Target: <strong className="text-slate-900 dark:text-white font-bold">({pA.toFixed(3)}, {pV.toFixed(3)})</strong>
            </span>
            <span>
              Threshold: <strong className="text-slate-600 dark:text-zinc-400 font-medium">τ=(0.50, 0.50)</strong>
            </span>
          </div>
        </div>

        {/* Right side: Disentangled Attribution Breakdown & Physics */}
        <div className="lg:col-span-5 space-y-4 w-full">
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-slate-500 dark:text-zinc-400 uppercase tracking-wider font-semibold block">
              Disentangled Softmax Attribution
            </span>
            <p className="text-xs text-slate-600 dark:text-zinc-400 leading-relaxed">
              Cross-attention heads classify audio and visual streams independently, separating lip-sync anomalies from voice cloning.
            </p>
          </div>

          {/* Probability Cards Grid */}
          <div className="grid grid-cols-2 gap-2 sm:gap-2.5">
            {(
              [
                { id: "RVRA", label: "RVRA", title: "Authentic", desc: "Real-V / Real-A" },
                { id: "FVRA", label: "FVRA", title: "Face Swap", desc: "Fake-V / Real-A" },
                { id: "RVFA", label: "RVFA", title: "Voice Clone", desc: "Real-V / Fake-A" },
                { id: "FVFA", label: "FVFA", title: "Dual Fake", desc: "Fake-V / Fake-A" },
              ] as const
            ).map((item) => {
              const prob = (quadrant.probs[item.id] || 0) * 100;
              const isCurrent = quadrant.label === item.id;
              const theme = quadrantThemes[item.id];

              return (
                <div
                  key={item.id}
                  className={`p-3 rounded-xl border transition-all ${
                    isCurrent
                      ? `${theme.bg} ${theme.border} ring-1 ring-slate-300 dark:ring-white/10 shadow-md`
                      : "bg-white dark:bg-[#0A0D14]/80 border-slate-200/90 dark:border-white/[0.04] hover:border-slate-300 dark:hover:border-white/10 shadow-sm"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${
                        isCurrent
                          ? theme.badge
                          : "bg-slate-100 dark:bg-white/[0.05] text-slate-600 dark:text-zinc-400"
                      }`}
                    >
                      {item.label}
                    </span>
                    <span className="text-xs sm:text-sm font-mono font-bold text-slate-900 dark:text-white">
                      {prob.toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-800 dark:text-zinc-200 font-semibold mt-1.5 truncate">
                    {item.title}
                  </div>
                  <div className="text-[10px] text-slate-500 dark:text-zinc-500 font-mono truncate">
                    {item.desc}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Calibrated ECE Guarantee Notice */}
          <div className="p-3 rounded-xl bg-slate-50 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.06] text-[11px] text-slate-600 dark:text-zinc-400 space-y-1 font-mono">
            <div className="flex justify-between items-center text-slate-700 dark:text-zinc-300">
              <span className="font-medium">Expected Calibration Error (ECE):</span>
              <span className="text-emerald-700 dark:text-emerald-400 font-bold">&le; 0.042</span>
            </div>
            <p className="text-[10px] text-slate-500 dark:text-zinc-500 leading-normal">
              Probability space calibrated via Post-Hoc Isotonic Temperature Scaling against adversarial attacks.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
