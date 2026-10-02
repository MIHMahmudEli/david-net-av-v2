"use client";

import React, { useState, useRef, useEffect } from "react";
import { Uploader } from "../components/Uploader";
import { VerdictCard } from "../components/VerdictCard";
import { QuadrantBadge } from "../components/QuadrantBadge";
import { DualTimeline } from "../components/DualTimeline";
import { SyncCurve } from "../components/SyncCurve";
import { Disclaimer } from "../components/Disclaimer";
import {
  PredictResponse,
  uploadMediaForPrediction,
  getApiBaseUrl,
  checkBackendHealth,
} from "../lib/api";
import {
  Shield,
  Download,
  Settings,
  Cpu,
  RefreshCw,
  Radar,
  Sparkles,
  Sun,
  Moon,
} from "lucide-react";

// Precomputed Curated Benchmarks for instant 1-click evaluation
const SAMPLE_BENCHMARKS: Record<"RVRA" | "FVRA" | "RVFA" | "FVFA", PredictResponse> = {
  RVRA: {
    clip_id: "BENCH_RVRA_GENUINE_001",
    duration_sec: 4.0,
    latency_ms: 164,
    modalities: { video: true, audio: true },
    model_version: "v2.4-EXP_011_s42",
    backbone_mode: "VideoMAE-Base + WavLM-Base+",
    disclaimer: "Calibrated multimodal inference estimate.",
    video: { verdict: "real", confidence: 0.985, raw_prob: 0.015 },
    audio: { verdict: "real", confidence: 0.978, raw_prob: 0.022 },
    quadrant: {
      label: "RVRA",
      description: "Pristine Authentic Capture",
      probs: { RVRA: 0.963, RVFA: 0.022, FVRA: 0.011, FVFA: 0.004 },
    },
    localization: { video: [], audio: [] },
    sync_curve: [
      0.65, 0.68, 0.72, 0.70, 0.75, 0.78, 0.74, 0.71, 0.76, 0.79,
      0.82, 0.80, 0.77, 0.75, 0.79, 0.81, 0.83, 0.80, 0.78, 0.76,
      0.79, 0.82, 0.80, 0.77, 0.75, 0.78, 0.81, 0.79, 0.76, 0.78,
      0.80, 0.82, 0.79, 0.77, 0.79, 0.81, 0.83, 0.80, 0.78, 0.76,
      0.79, 0.81, 0.83, 0.80, 0.78, 0.76, 0.79, 0.81, 0.82, 0.80,
    ],
  },
  FVRA: {
    clip_id: "BENCH_FVRA_FACESWAP_002",
    duration_sec: 4.0,
    latency_ms: 188,
    modalities: { video: true, audio: true },
    model_version: "v2.4-EXP_011_s42",
    backbone_mode: "VideoMAE-Base + WavLM-Base+",
    disclaimer: "Calibrated multimodal inference estimate.",
    video: { verdict: "fake", confidence: 0.964, raw_prob: 0.964 },
    audio: { verdict: "real", confidence: 0.942, raw_prob: 0.058 },
    quadrant: {
      label: "FVRA",
      description: "Facial Synthesis / Re-enactment",
      probs: { RVRA: 0.038, RVFA: 0.012, FVRA: 0.915, FVFA: 0.035 },
    },
    localization: {
      video: [{ start: 0.8, end: 3.1, score: 0.964 }],
      audio: [],
    },
    sync_curve: [
      0.24, 0.18, 0.05, -0.15, -0.32, -0.44, -0.38, -0.22, -0.12, 0.04,
      0.12, 0.18, 0.22, 0.15, 0.02, -0.18, -0.35, -0.42, -0.31, -0.14,
      -0.02, 0.08, 0.15, 0.20, 0.12, -0.05, -0.24, -0.39, -0.28, -0.11,
      0.02, 0.14, 0.19, 0.12, -0.04, -0.22, -0.36, -0.29, -0.10, 0.05,
      0.15, 0.21, 0.16, 0.04, -0.15, -0.32, -0.25, -0.08, 0.09, 0.18,
    ],
  },
  RVFA: {
    clip_id: "BENCH_RVFA_VOICEDUB_003",
    duration_sec: 4.0,
    latency_ms: 172,
    modalities: { video: true, audio: true },
    model_version: "v2.4-EXP_011_s42",
    backbone_mode: "VideoMAE-Base + WavLM-Base+",
    disclaimer: "Calibrated multimodal inference estimate.",
    video: { verdict: "real", confidence: 0.948, raw_prob: 0.052 },
    audio: { verdict: "fake", confidence: 0.972, raw_prob: 0.972 },
    quadrant: {
      label: "RVFA",
      description: "Voice Clone / Neural TTS Dub",
      probs: { RVRA: 0.029, RVFA: 0.923, FVRA: 0.015, FVFA: 0.033 },
    },
    localization: {
      video: [],
      audio: [{ start: 1.1, end: 3.5, score: 0.972 }],
    },
    sync_curve: [
      -0.38, -0.45, -0.52, -0.58, -0.65, -0.60, -0.55, -0.48, -0.54, -0.62,
      -0.68, -0.72, -0.65, -0.58, -0.52, -0.60, -0.67, -0.71, -0.64, -0.56,
      -0.49, -0.55, -0.63, -0.70, -0.66, -0.59, -0.51, -0.46, -0.53, -0.61,
      -0.67, -0.70, -0.63, -0.55, -0.48, -0.56, -0.64, -0.69, -0.62, -0.54,
      -0.47, -0.53, -0.61, -0.68, -0.64, -0.57, -0.49, -0.44, -0.51, -0.59,
    ],
  },
  FVFA: {
    clip_id: "BENCH_FVFA_DUALSYNTH_004",
    duration_sec: 4.0,
    latency_ms: 215,
    modalities: { video: true, audio: true },
    model_version: "v2.4-EXP_011_s42",
    backbone_mode: "VideoMAE-Base + WavLM-Base+",
    disclaimer: "Calibrated multimodal inference estimate.",
    video: { verdict: "fake", confidence: 0.991, raw_prob: 0.991 },
    audio: { verdict: "fake", confidence: 0.984, raw_prob: 0.984 },
    quadrant: {
      label: "FVFA",
      description: "Fully Synthetic Deepfake",
      probs: { RVRA: 0.005, RVFA: 0.021, FVRA: 0.024, FVFA: 0.95 },
    },
    localization: {
      video: [{ start: 0.3, end: 3.9, score: 0.991 }],
      audio: [{ start: 0.5, end: 3.6, score: 0.984 }],
    },
    sync_curve: [
      -0.65, -0.72, -0.78, -0.84, -0.89, -0.85, -0.79, -0.73, -0.80, -0.86,
      -0.91, -0.94, -0.88, -0.82, -0.76, -0.83, -0.89, -0.93, -0.87, -0.80,
      -0.74, -0.81, -0.87, -0.92, -0.88, -0.81, -0.75, -0.70, -0.77, -0.84,
      -0.90, -0.93, -0.86, -0.80, -0.74, -0.81, -0.88, -0.92, -0.85, -0.78,
      -0.72, -0.79, -0.86, -0.91, -0.87, -0.80, -0.73, -0.68, -0.75, -0.82,
    ],
  },
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [explain, setExplain] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Playhead & video synchronizer
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Backend state
  const [backendUrl, setBackendUrl] = useState<string>("");
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");
  const [modelVersion, setModelVersion] = useState<string>("");
  const [backboneMode, setBackboneMode] = useState<string>("");

  // Theme mode state (persisted to localStorage)
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("david_theme");
      if (saved === "dark") {
        setTheme("dark");
        document.documentElement.classList.add("dark");
      } else {
        setTheme("light");
        document.documentElement.classList.remove("dark");
      }
    } catch (_) {}
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    try {
      localStorage.setItem("david_theme", nextTheme);
      if (nextTheme === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    } catch (_) {}
  };

  useEffect(() => {
    const base = getApiBaseUrl();
    setBackendUrl(base);

    checkBackendHealth()
      .then((data) => {
        setBackendStatus("online");
        if (data.model_version) setModelVersion(data.model_version);
        if (data.backbone_mode) setBackboneMode(data.backbone_mode);
      })
      .catch(() => {
        setBackendStatus("offline");
      });
  }, []);

  // Synchronize native video player events
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };
    const handleDurationChange = () => {
      setVideoDuration(video.duration);
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("durationchange", handleDurationChange);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);

    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("durationchange", handleDurationChange);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
    };
  }, [selectedFile]);

  const handleTogglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  };

  const handleSeek = (timeSec: number) => {
    setCurrentTime(timeSec);
    if (videoRef.current) {
      videoRef.current.currentTime = timeSec;
    }
  };

  const handleConfigureBackend = () => {
    const current = backendUrl;
    const input = prompt("Enter DAVID-Net Inference Backend API URL:", current);
    if (input !== null) {
      const clean = input.trim().replace(/\/+$/, "");
      if (clean) {
        localStorage.setItem("david_api_base", clean);
        setBackendUrl(clean);
      } else {
        localStorage.removeItem("david_api_base");
        setBackendUrl(getApiBaseUrl());
      }
      setBackendStatus("checking");
      checkBackendHealth()
        .then((data) => {
          setBackendStatus("online");
          if (data.backbone_mode) setBackboneMode(data.backbone_mode);
        })
        .catch(() => setBackendStatus("offline"));
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const data = await uploadMediaForPrediction(selectedFile, explain);
      setResult(data);
      if (data.backbone_mode) setBackboneMode(data.backbone_mode);
      if (data.duration_sec && !videoDuration) {
        setVideoDuration(data.duration_sec);
      }
    } catch (err: any) {
      setError(err.message || "Failed to analyze media file.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSamplePreset = (presetId: "RVRA" | "FVRA" | "RVFA" | "FVFA") => {
    const sample = SAMPLE_BENCHMARKS[presetId];
    setResult(sample);
    setVideoDuration(sample.duration_sec);
    setCurrentTime(0);
    setError(null);
  };

  const handleExportJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `davidnet_forensic_report_${result.clip_id || "evidence"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getHostname = (url: string) => {
    try {
      return new URL(url).hostname;
    } catch {
      return url || "default";
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#F8FAFC] dark:bg-[#08090D] text-slate-800 dark:text-zinc-100 selection:bg-sky-500 selection:text-white relative overflow-x-hidden transition-colors duration-200">
      {/* Dynamic Ambient Background Meshes for White/Dark Theme */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute -top-[20%] left-1/2 -translate-x-1/2 w-[900px] h-[550px] bg-gradient-to-b from-sky-200/35 via-indigo-100/20 to-transparent dark:from-sky-500/10 dark:via-indigo-500/5 blur-[140px] rounded-full" />
        <div className="absolute top-[30%] -right-[15%] w-[600px] h-[600px] bg-cyan-100/30 dark:bg-cyan-500/5 blur-[150px] rounded-full" />
        <div
          className="absolute inset-0 opacity-[0.35] dark:opacity-[0.035]"
          style={{
            backgroundImage:
              theme === "dark"
                ? "radial-gradient(rgba(255, 255, 255, 0.8) 1px, transparent 1px)"
                : "radial-gradient(rgba(148, 163, 184, 0.35) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />
      </div>

      {/* Top Telemetry HUD Header */}
      <header className="border-b border-slate-200/80 dark:border-white/[0.08] bg-white/80 dark:bg-[#0A0D14]/80 backdrop-blur-xl sticky top-0 z-50 shadow-sm transition-colors duration-200">
        <div className="max-w-[1560px] mx-auto px-3.5 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center space-x-2.5 sm:space-x-3 overflow-hidden">
            <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-black text-white shadow-md shadow-sky-500/20 shrink-0">
              <Shield className="w-4 h-4 sm:w-5 sm:h-5 fill-current" />
            </div>
            <div className="truncate">
              <div className="flex items-center space-x-2">
                <span className="text-sm sm:text-base font-black tracking-tight text-slate-900 dark:text-white font-mono">
                  DAVID-Net
                </span>
                <span
                  onClick={handleConfigureBackend}
                  title="Click to configure API Base URL"
                  className="text-[9px] sm:text-[10px] font-mono tracking-wider text-sky-700 dark:text-sky-400 bg-sky-50 dark:bg-sky-950/60 border border-sky-200 dark:border-sky-800/60 px-1.5 sm:px-2 py-0.5 rounded-full shrink-0 font-semibold cursor-pointer hover:border-sky-400 transition"
                >
                  v2.4 WORKBENCH
                </span>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-zinc-400 font-mono hidden md:block">
                Disentangled Audio-Visual Cross-Modal Attribution
              </p>
            </div>
          </div>

          {/* Right Controls: Theme Toggle Switch */}
          <div className="flex items-center space-x-2 sm:space-x-3 shrink-0">
            <button
              type="button"
              onClick={toggleTheme}
              className="flex items-center space-x-2 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-white/10 bg-slate-100/90 dark:bg-white/[0.06] hover:bg-slate-200/90 dark:hover:bg-white/[0.12] text-slate-700 dark:text-zinc-200 transition-all font-mono text-xs shadow-sm active:scale-95 group"
              title={theme === "dark" ? "Switch to White Mode" : "Switch to Dark Mode"}
            >
              {theme === "dark" ? (
                <>
                  <Sun className="w-4 h-4 text-amber-400 group-hover:rotate-45 transition-transform" />
                  <span className="font-semibold text-zinc-200 hidden sm:inline">White Mode</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4 text-sky-700 group-hover:-rotate-12 transition-transform" />
                  <span className="font-semibold text-slate-800 hidden sm:inline">Dark Mode</span>
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Cockpit Layout */}
      <main className="max-w-[1560px] mx-auto px-3.5 sm:px-6 lg:px-8 py-5 sm:py-8 w-full flex-1 z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
          {/* ============================================================== */}
          {/* LEFT COLUMN: Media Ingestion, Video Player & Analysis Controls */}
          {/* ============================================================== */}
          <div className="lg:col-span-5 space-y-5 sm:space-y-6">
            {/* Ingestion Cockpit Card */}
            <div className="glass-panel rounded-2xl p-4 sm:p-6 shadow-xl space-y-4 sm:space-y-5">
              <div className="flex items-center justify-between pb-3 border-b border-slate-200/80 dark:border-white/[0.08]">
                <div className="flex items-center space-x-2">
                  <Radar className="w-4 h-4 text-sky-600 dark:text-sky-400" />
                  <h2 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider font-mono">
                    Media Ingestion & Player
                  </h2>
                </div>
                <span className="text-[10px] font-mono text-slate-500 dark:text-zinc-400 font-medium">
                  Multimodal Demux
                </span>
              </div>

              {/* Uploader with Integrated Player */}
              <Uploader
                selectedFile={selectedFile}
                onFileSelect={(file) => {
                  setSelectedFile(file);
                  setResult(null);
                  setError(null);
                }}
                onClear={() => {
                  setSelectedFile(null);
                  setResult(null);
                  setCurrentTime(0);
                  setVideoDuration(0);
                }}
                isLoading={loading}
                onLoadSamplePreset={handleLoadSamplePreset}
                videoRef={videoRef}
                currentTime={currentTime}
                duration={videoDuration || (result ? result.duration_sec : 0)}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                onSeek={handleSeek}
              />

              {/* Inference Config & Actions */}
              <div className="pt-2 border-t border-slate-200/80 dark:border-white/[0.08] space-y-3.5 sm:space-y-4">
                <label className="flex items-center space-x-2.5 text-xs text-slate-600 dark:text-zinc-400 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={explain}
                    onChange={(e) => setExplain(e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 dark:border-zinc-700 text-sky-600 focus:ring-sky-500 bg-white dark:bg-zinc-900"
                  />
                  <span className="font-mono text-[11px] text-slate-700 dark:text-zinc-300 font-medium">
                    Extract Cross-Modal Attention Saliency (Av→a, Aa→v)
                  </span>
                </label>

                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={!selectedFile || loading}
                  className="w-full py-3 sm:py-3.5 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-600 hover:to-indigo-700 disabled:from-slate-200 dark:disabled:from-zinc-800 disabled:to-slate-200 dark:disabled:to-zinc-800 disabled:text-slate-400 dark:disabled:text-zinc-600 font-mono font-bold text-xs sm:text-sm text-white shadow-lg shadow-sky-500/20 transition-all flex items-center justify-center space-x-2.5 active:scale-[0.99]"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Computing Cross-Modal Inference...</span>
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" />
                      <span>Execute Forensic Verification</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/60 rounded-xl p-4 text-xs font-mono text-rose-800 dark:text-rose-200 space-y-1 shadow-sm">
                <div className="font-bold uppercase tracking-wider text-rose-700 dark:text-rose-400">
                  Forensic Inference Error:
                </div>
                <div>{error}</div>
              </div>
            )}

            {/* System Architectural Specs Callout */}
            <div className="glass-panel rounded-2xl p-4 sm:p-5 text-xs font-mono text-slate-600 dark:text-zinc-400 space-y-3 shadow-sm">
              <div className="flex items-center justify-between text-slate-800 dark:text-zinc-200 pb-2 border-b border-slate-200/80 dark:border-white/[0.08]">
                <span className="uppercase tracking-wider text-[10px] font-semibold text-slate-500 dark:text-zinc-400">
                  Pipeline Architecture
                </span>
                <span className="text-sky-700 dark:text-sky-400 font-bold">DAVID-Net 2.0</span>
              </div>
              <div className="grid grid-cols-2 gap-2.5 sm:gap-3 text-[10px] sm:text-[11px]">
                <div>
                  <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Visual Backbone</span>
                  <span className="text-slate-800 dark:text-zinc-300 font-semibold">VideoMAE (ViT-Base)</span>
                </div>
                <div>
                  <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Audio Backbone</span>
                  <span className="text-slate-800 dark:text-zinc-300 font-semibold">WavLM (Base-Plus)</span>
                </div>
                <div>
                  <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Cross-Attention</span>
                  <span className="text-slate-800 dark:text-zinc-300 font-semibold">Disentangled Pre-LN</span>
                </div>
                <div>
                  <span className="text-slate-400 dark:text-zinc-500 block text-[9px] uppercase font-sans">Attribution</span>
                  <span className="text-slate-800 dark:text-zinc-300 font-semibold">4-Quadrant Softmax</span>
                </div>
              </div>
            </div>
          </div>

          {/* ============================================================== */}
          {/* RIGHT COLUMN: Intelligence Dossier & Telemetry Analytics        */}
          {/* ============================================================== */}
          <div className="lg:col-span-7 space-y-5 sm:space-y-6">
            {result ? (
              <div className="space-y-5 sm:space-y-6 animate-fadeIn">
                {/* Result Dossier Header with Export JSON Action */}
                <div className="flex items-center justify-between px-1">
                  <div>
                    <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider font-mono">
                      Forensic Intelligence Dossier
                    </h3>
                    <p className="text-[10px] text-slate-500 dark:text-zinc-400 font-mono">
                      Calibrated Multimodal Attributions &amp; Cross-Modal Telemetry
                    </p>
                  </div>
                  <button
                    onClick={handleExportJson}
                    className="inline-flex items-center space-x-1.5 text-xs font-mono text-slate-700 dark:text-zinc-200 hover:text-slate-900 dark:hover:text-white bg-white dark:bg-white/[0.06] hover:bg-slate-50 dark:hover:bg-white/[0.12] border border-slate-200 dark:border-white/10 px-3 py-1.5 rounded-xl shadow-sm transition active:scale-95"
                    title="Download Serialized RFC-8259 Forensic Report"
                  >
                    <Download className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
                    <span className="font-semibold">Export JSON</span>
                  </button>
                </div>

                {/* 1. 2D Cartesian Quadrant Attribution Plane */}
                {result.quadrant && (
                  <QuadrantBadge
                    quadrant={result.quadrant}
                    videoProb={result.video?.raw_prob}
                    audioProb={result.audio?.raw_prob}
                    latencyMs={result.latency_ms}
                    durationSec={result.duration_sec}
                    clipId={result.clip_id}
                  />
                )}

                {/* 2. Disentangled Modality Authenticity Gauges */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 sm:gap-4">
                  <VerdictCard
                    title="Video Stream Authenticity"
                    verdict={result.video}
                    modalityType="video"
                  />
                  <VerdictCard
                    title="Audio Stream Authenticity"
                    verdict={result.audio}
                    modalityType="audio"
                  />
                </div>

                {/* 3. Temporal Forgery Localization Multi-Track */}
                <DualTimeline
                  videoIntervals={result.localization.video}
                  audioIntervals={result.localization.audio}
                  durationSec={result.duration_sec}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                />

                {/* 4. Cross-Modal Synchronization Agreement Curve */}
                <SyncCurve
                  syncCurve={result.sync_curve}
                  durationSec={result.duration_sec}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                />

                {/* 5. Responsible Use & Decision Support Guarantee */}
                <Disclaimer />
              </div>
            ) : (
              /* Awaiting Ingestion Standby State */
              <div className="glass-panel rounded-2xl p-6 sm:p-12 text-center space-y-6 shadow-xl">
                <div className="w-14 h-14 sm:w-16 sm:h-16 mx-auto rounded-2xl bg-sky-50 dark:bg-white/[0.03] border border-sky-200/80 dark:border-white/[0.08] flex items-center justify-center text-sky-600 dark:text-sky-400 shadow-sm">
                  <Radar className="w-7 h-7 sm:w-8 sm:h-8 animate-pulse" />
                </div>
                <div className="max-w-md mx-auto space-y-2">
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white tracking-tight">
                    Forensic Intelligence Dossier Standby
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-zinc-400 leading-relaxed font-mono">
                    Select an audio-visual media clip on the left, or tap one of
                    the benchmark presets (RVRA, FVRA, RVFA, FVFA) to execute
                    instant multimodal verification.
                  </p>
                </div>

                {/* Schematic Blueprint Feature Matrix */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3 text-left pt-6 border-t border-slate-200/80 dark:border-white/[0.08]">
                  <div className="p-3 rounded-xl bg-slate-50/80 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.04]">
                    <span className="text-[10px] font-mono text-sky-700 dark:text-sky-400 uppercase block font-bold">
                      Feature 01
                    </span>
                    <strong className="text-xs text-slate-900 dark:text-white block mt-0.5">
                      2D Cartesian
                    </strong>
                    <span className="text-[10px] text-slate-500 dark:text-zinc-400">
                      $p_a \times p_v$ coordinate radar
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50/80 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.04]">
                    <span className="text-[10px] font-mono text-sky-700 dark:text-sky-400 uppercase block font-bold">
                      Feature 02
                    </span>
                    <strong className="text-xs text-slate-900 dark:text-white block mt-0.5">
                      Disentanglement
                    </strong>
                    <span className="text-[10px] text-slate-500 dark:text-zinc-400">
                      Independent per-stream logits
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50/80 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.04]">
                    <span className="text-[10px] font-mono text-sky-700 dark:text-sky-400 uppercase block font-bold">
                      Feature 03
                    </span>
                    <strong className="text-xs text-slate-900 dark:text-white block mt-0.5">
                      Dual MLP Bounds
                    </strong>
                    <span className="text-[10px] text-slate-500 dark:text-zinc-400">
                      Frame-dense interval cuts
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50/80 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.04]">
                    <span className="text-[10px] font-mono text-sky-700 dark:text-sky-400 uppercase block font-bold">
                      Feature 04
                    </span>
                    <strong className="text-xs text-slate-900 dark:text-white block mt-0.5">
                      Sync Curve ($s_t$)
                    </strong>
                    <span className="text-[10px] text-slate-500 dark:text-zinc-400">
                      Phoneme-viseme InfoNCE
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Forensic Telemetry Footer */}
      <footer className="border-t border-slate-200/80 dark:border-white/[0.08] py-4 sm:py-5 px-3.5 sm:px-8 text-xs font-mono text-slate-500 dark:text-zinc-400 flex flex-col sm:flex-row items-center justify-between gap-2.5 sm:gap-3 bg-white/80 dark:bg-[#0A0D14]/80 backdrop-blur-md z-10 transition-colors duration-200">
        <div className="text-center sm:text-left text-[11px] sm:text-xs">
          DAVID-Net &bull; Disentangled Audio-Visual Deepfake Attribution &bull; AIUB Thesis
        </div>
        <div className="flex items-center space-x-3 text-[10px] sm:text-[11px] text-slate-500 dark:text-zinc-400">
          <span>RFC-8259 Compliant</span>
          <span>&bull;</span>
          <span>Zero-Retention RAM Enclave</span>
        </div>
      </footer>
    </div>
  );
}
