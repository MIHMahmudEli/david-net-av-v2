"use client";

import React from "react";
import { ModalityVerdict } from "../lib/api";
import { Video, Mic, ShieldCheck, AlertTriangle, HelpCircle } from "lucide-react";

interface VerdictCardProps {
  title: string;
  verdict: ModalityVerdict;
  modalityType: "video" | "audio";
}

export const VerdictCard: React.FC<VerdictCardProps> = ({
  title,
  verdict,
  modalityType,
}) => {
  const isFake = verdict.verdict === "fake";
  const isReal = verdict.verdict === "real";
  const isUnavailable = verdict.verdict === "unavailable";

  const percent = verdict.confidence ? Math.round(verdict.confidence * 1000) / 10 : 0;
  const rawProb = verdict.raw_prob !== undefined ? verdict.raw_prob : (isFake ? 0.95 : 0.05);

  const IconComponent = modalityType === "video" ? Video : Mic;

  return (
    <div
      className={`glass-panel rounded-2xl p-4 sm:p-5 shadow-sm relative overflow-hidden transition-all ${
        isFake
          ? "border-rose-200/90 dark:border-rose-500/30 bg-rose-50/40 dark:bg-rose-950/15"
          : isReal
          ? "border-emerald-200/90 dark:border-emerald-500/30 bg-emerald-50/40 dark:bg-emerald-950/15"
          : "border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#0A0D14]/80"
      }`}
    >
      {/* Top Banner */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-200/70 dark:border-white/[0.06]">
        <div className="flex items-center space-x-2.5">
          <div
            className={`p-2 rounded-xl ${
              isFake
                ? "bg-rose-100 dark:bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-500/20"
                : isReal
                ? "bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20"
                : "bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400 border border-slate-200 dark:border-zinc-700"
            }`}
          >
            <IconComponent className="w-4 h-4" />
          </div>
          <div>
            <span className="text-[10px] uppercase tracking-widest font-mono text-slate-500 dark:text-zinc-400 font-semibold block">
              Modality Stream
            </span>
            <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white tracking-tight">{title}</h3>
          </div>
        </div>

        <div className="flex items-center space-x-1.5">
          {isFake ? (
            <span className="inline-flex items-center space-x-1 text-[11px] sm:text-xs font-mono font-bold px-2 sm:px-2.5 py-1 rounded-md bg-rose-100 dark:bg-rose-500/15 text-rose-800 dark:text-rose-300 border border-rose-300 dark:border-rose-500/30 shadow-sm">
              <AlertTriangle className="w-3.5 h-3.5 mr-1 text-rose-600 dark:text-rose-400 shrink-0" />
              MANIPULATED
            </span>
          ) : isReal ? (
            <span className="inline-flex items-center space-x-1 text-[11px] sm:text-xs font-mono font-bold px-2 sm:px-2.5 py-1 rounded-md bg-emerald-100 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30 shadow-sm">
              <ShieldCheck className="w-3.5 h-3.5 mr-1 text-emerald-600 dark:text-emerald-400 shrink-0" />
              AUTHENTIC
            </span>
          ) : (
            <span className="inline-flex items-center space-x-1 text-[11px] sm:text-xs font-mono font-bold px-2 sm:px-2.5 py-1 rounded-md bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-400 border border-slate-200 dark:border-zinc-700">
              <HelpCircle className="w-3.5 h-3.5 mr-1 shrink-0" />
              UNAVAILABLE
            </span>
          )}
        </div>
      </div>

      {/* Confidence Metrics */}
      <div className="mt-3.5 space-y-3">
        <div className="flex items-baseline justify-between">
          <span className="text-[11px] sm:text-xs font-mono text-slate-500 dark:text-zinc-400 uppercase font-medium">
            Confidence Posterior:
          </span>
          <div className="flex items-baseline space-x-1">
            <span
              className={`text-2xl sm:text-3xl font-mono font-black tracking-tight ${
                isFake
                  ? "text-rose-600 dark:text-rose-400"
                  : isReal
                  ? "text-emerald-600 dark:text-emerald-400"
                  : "text-slate-400 dark:text-zinc-400"
              }`}
            >
              {isUnavailable ? "--" : `${percent.toFixed(1)}%`}
            </span>
          </div>
        </div>

        {/* Gauge bar */}
        <div className="w-full bg-slate-100 dark:bg-[#08090D] h-2.5 rounded-full overflow-hidden p-0.5 border border-slate-200 dark:border-white/[0.08]">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              isFake
                ? "bg-gradient-to-r from-rose-500 to-rose-400 shadow-sm"
                : isReal
                ? "bg-gradient-to-r from-emerald-500 to-emerald-400 shadow-sm"
                : "bg-slate-300 dark:bg-zinc-700"
            }`}
            style={{ width: `${Math.min(100, Math.max(4, percent))}%` }}
          />
        </div>

        {/* Telemetry Footnote */}
        <div className="grid grid-cols-2 pt-2 text-[10px] sm:text-[11px] font-mono text-slate-500 dark:text-zinc-400 border-t border-slate-200/60 dark:border-white/[0.04]">
          <div>
            <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Backbone Feature</span>
            <span className="text-slate-700 dark:text-zinc-300 font-semibold">
              {modalityType === "video" ? "VideoMAE ViT-B" : "WavLM Base+"}
            </span>
          </div>
          <div className="text-right">
            <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Sigmoid Logit</span>
            <span className="text-slate-800 dark:text-zinc-200 font-bold">
              {typeof rawProb === "number" ? rawProb.toFixed(4) : rawProb}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
