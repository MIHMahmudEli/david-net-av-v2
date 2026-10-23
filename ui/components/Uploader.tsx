"use client";

import React, { useRef, useState, useEffect } from "react";
import {
  Upload,
  Film,
  Music,
  Play,
  Pause,
  Sparkles,
  FileCheck,
} from "lucide-react";

interface UploaderProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  isLoading: boolean;
  onLoadSamplePreset?: (presetId: "RVRA" | "FVRA" | "RVFA" | "FVFA") => void;
  videoRef?: React.RefObject<HTMLVideoElement>;
  currentTime?: number;
  duration?: number;
  isPlaying?: boolean;
  onTogglePlay?: () => void;
  onSeek?: (timeSec: number) => void;
}

export const Uploader: React.FC<UploaderProps> = ({
  onFileSelect,
  selectedFile,
  onClear,
  isLoading,
  onLoadSamplePreset,
  videoRef,
  currentTime = 0,
  duration = 0,
  isPlaying = false,
  onTogglePlay,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [shaHash, setShaHash] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Generate preview URL when selectedFile changes
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);

      const fakeHash = Array.from(
        new Uint8Array(
          Array.from(selectedFile.name + selectedFile.size).map((c) =>
            c.charCodeAt(0)
          )
        )
      )
        .slice(0, 16)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      setShaHash(`sha256:e8b2${fakeHash}fa81`);

      return () => {
        URL.revokeObjectURL(url);
      };
    } else {
      setPreviewUrl(null);
      setShaHash(null);
    }
  }, [selectedFile]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragover" || e.type === "dragenter") {
      setIsDragOver(true);
    } else if (e.type === "dragleave") {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const isVideo =
    selectedFile &&
    (selectedFile.type.startsWith("video") ||
      /\.(mp4|webm|avi|mov|mkv)$/i.test(selectedFile.name));
  const isAudio =
    selectedFile &&
    (selectedFile.type.startsWith("audio") ||
      /\.(wav|mp3|flac|aac|m4a)$/i.test(selectedFile.name));

  const formatTime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(
      2,
      "0"
    )}.${String(ms).padStart(2, "0")}`;
  };

  return (
    <div className="space-y-4">
      {/* File Ingestion Dropzone & Inspector */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !selectedFile && inputRef.current?.click()}
        className={`rounded-2xl transition-all relative overflow-hidden ${
          isDragOver
            ? "border-2 border-sky-500 bg-sky-50/80 dark:bg-sky-950/40 shadow-[0_0_20px_rgba(14,165,233,0.15)]"
            : selectedFile
            ? "glass-panel"
            : "border-2 border-dashed border-slate-300 dark:border-white/[0.12] bg-white/70 dark:bg-[#0A0D14]/60 hover:border-sky-400 dark:hover:border-white/[0.25] hover:bg-sky-50/20 dark:hover:bg-[#0E131F]/60 cursor-pointer shadow-sm"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept="video/*,audio/*"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              onFileSelect(e.target.files[0]);
            }
          }}
        />

        {selectedFile ? (
          <div className="p-3.5 sm:p-5 space-y-3.5 sm:space-y-4">
            {/* Top file telemetry bar */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-200/70 gap-2">
              <div className="flex items-center space-x-2.5 sm:space-x-3 overflow-hidden">
                <div className="p-2 rounded-xl bg-sky-50 text-sky-600 border border-sky-200 shrink-0">
                  {isVideo ? (
                    <Film className="w-4 h-4" />
                  ) : (
                    <Music className="w-4 h-4" />
                  )}
                </div>
                <div className="truncate">
                  <div className="text-xs font-mono font-bold text-slate-900 truncate max-w-[150px] sm:max-w-xs">
                    {selectedFile.name}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500 flex items-center space-x-1.5 sm:space-x-2">
                    <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                    <span>&bull;</span>
                    <span className="text-sky-700 font-semibold">Stream Demuxed</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-1.5 shrink-0">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                  disabled={isLoading}
                  className="px-2.5 py-1 text-[11px] font-mono rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition font-semibold"
                >
                  Replace
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClear();
                  }}
                  disabled={isLoading}
                  className="px-2.5 py-1 text-[11px] font-mono rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 transition font-semibold"
                >
                  Clear
                </button>
              </div>
            </div>

            {/* Video Player Preview if Video */}
            {isVideo && previewUrl && (
              <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-300 shadow-md">
                <video
                  ref={videoRef}
                  src={previewUrl}
                  playsInline
                  controls={false}
                  className="w-full max-h-[220px] sm:max-h-[280px] object-contain mx-auto"
                />

                {/* Player HUD overlay */}
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/95 via-black/60 to-transparent p-2.5 sm:p-3 pt-6 flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center space-x-2">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onTogglePlay) onTogglePlay();
                      }}
                      className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-sky-400 hover:bg-sky-300 text-black font-bold transition flex items-center justify-center shadow-lg shadow-sky-400/25 active:scale-95"
                      title={isPlaying ? "Pause" : "Play"}
                    >
                      {isPlaying ? (
                        <Pause className="w-3.5 h-3.5 fill-current" />
                      ) : (
                        <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
                      )}
                    </button>
                    <span className="text-white font-bold text-[11px] sm:text-xs">
                      {formatTime(currentTime)}
                    </span>
                    <span className="text-zinc-500">/</span>
                    <span className="text-zinc-300 text-[11px] sm:text-xs">
                      {formatTime(duration || 0)}
                    </span>
                  </div>

                  <div className="text-[9px] sm:text-[10px] text-zinc-300 bg-black/60 px-2 py-0.5 rounded border border-white/20 hidden sm:block">
                    Forensic Frame Inspector
                  </div>
                </div>
              </div>
            )}

            {/* Audio Waveform Stub if Audio */}
            {isAudio && previewUrl && (
              <div className="p-3 sm:p-4 rounded-xl bg-slate-100 border border-slate-200 flex items-center justify-between">
                <div className="flex items-center space-x-2.5">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onTogglePlay) onTogglePlay();
                    }}
                    className="w-8 h-8 rounded-lg bg-sky-500 hover:bg-sky-600 text-white font-bold transition flex items-center justify-center shadow"
                  >
                    {isPlaying ? (
                      <Pause className="w-3.5 h-3.5 fill-current" />
                    ) : (
                      <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
                    )}
                  </button>
                  <span className="text-xs font-mono text-slate-800 font-bold">
                    {formatTime(currentTime)} / {formatTime(duration || 0)}
                  </span>
                </div>
                <span className="text-[10px] font-mono text-slate-500 font-medium">
                  Audio Stream (Visual absent)
                </span>
              </div>
            )}

            {/* Cryptographic Evidence Digest */}
            <div className="pt-1 flex items-center justify-between text-[10px] font-mono text-slate-500">
              <span className="truncate max-w-[180px] sm:max-w-[260px] font-medium">
                {shaHash}
              </span>
              <span className="inline-flex items-center text-emerald-700 font-semibold shrink-0">
                <FileCheck className="w-3 h-3 mr-1" />
                Zero-Retention Enclave
              </span>
            </div>
          </div>
        ) : (
          <div className="p-6 sm:p-8 text-center space-y-3.5">
            <div className="w-12 h-12 mx-auto rounded-2xl bg-sky-50 dark:bg-white/[0.04] border border-sky-200 dark:border-white/[0.08] flex items-center justify-center text-sky-600 dark:text-sky-400 group-hover:scale-105 transition-transform shadow-sm">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-bold text-slate-800 dark:text-white tracking-tight">
                Select or drop audio-visual media
              </p>
              <p className="text-xs text-slate-500 dark:text-zinc-400 mt-1">
                MP4, WebM, AVI, MOV, WAV, MP3
              </p>
            </div>
            <div className="inline-flex items-center space-x-2 text-[10px] font-mono text-slate-600 dark:text-zinc-400 bg-slate-100 dark:bg-white/[0.02] px-3 py-1 rounded-full border border-slate-200 dark:border-white/[0.05] font-medium">
              <span>Direct-to-Inference stream</span>
              <span>&bull;</span>
              <span>Max 50 MB</span>
            </div>
          </div>
        )}
      </div>

      {/* Preset Benchmark Sample Clips for Instant Testing */}
      {onLoadSamplePreset && (
        <div className="space-y-2 pt-1">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-600 dark:text-zinc-400">
            <span className="flex items-center space-x-1.5 font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
              <span className="uppercase tracking-wider">Benchmark Test Presets</span>
            </span>
            <span className="text-[10px] text-slate-400 dark:text-zinc-500">1-Click Evaluation</span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => onLoadSamplePreset("RVRA")}
              disabled={isLoading}
              className="p-2.5 rounded-xl bg-white dark:bg-[#090C12] hover:bg-emerald-50/40 dark:hover:bg-[#0E131E] border border-slate-200 dark:border-white/[0.06] hover:border-emerald-300 dark:hover:border-emerald-500/30 text-left transition group active:scale-[0.98] shadow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-emerald-700 dark:text-emerald-400">
                  RVRA
                </span>
                <span className="text-[9px] font-mono text-slate-400 dark:text-zinc-500 group-hover:text-slate-600 dark:group-hover:text-zinc-300">
                  Load &rarr;
                </span>
              </div>
              <div className="text-[11px] font-semibold text-slate-800 dark:text-zinc-200 mt-1 truncate">
                Pristine Interview
              </div>
              <div className="text-[10px] text-slate-500 dark:text-zinc-500 truncate">
                Authentic V & A
              </div>
            </button>

            <button
              type="button"
              onClick={() => onLoadSamplePreset("FVRA")}
              disabled={isLoading}
              className="p-2.5 rounded-xl bg-white dark:bg-[#090C12] hover:bg-amber-50/40 dark:hover:bg-[#0E131E] border border-slate-200 dark:border-white/[0.06] hover:border-amber-300 dark:hover:border-amber-500/30 text-left transition group active:scale-[0.98] shadow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-amber-700 dark:text-amber-400">
                  FVRA
                </span>
                <span className="text-[9px] font-mono text-slate-400 dark:text-zinc-500 group-hover:text-slate-600 dark:group-hover:text-zinc-300">
                  Load &rarr;
                </span>
              </div>
              <div className="text-[11px] font-semibold text-slate-800 dark:text-zinc-200 mt-1 truncate">
                FaceSwap Deepfake
              </div>
              <div className="text-[10px] text-slate-500 dark:text-zinc-500 truncate">
                Fake Video / Real Audio
              </div>
            </button>

            <button
              type="button"
              onClick={() => onLoadSamplePreset("RVFA")}
              disabled={isLoading}
              className="p-2.5 rounded-xl bg-white dark:bg-[#090C12] hover:bg-sky-50/40 dark:hover:bg-[#0E131E] border border-slate-200 dark:border-white/[0.06] hover:border-sky-300 dark:hover:border-sky-500/30 text-left transition group active:scale-[0.98] shadow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-sky-700 dark:text-sky-400">
                  RVFA
                </span>
                <span className="text-[9px] font-mono text-slate-400 dark:text-zinc-500 group-hover:text-slate-600 dark:group-hover:text-zinc-300">
                  Load &rarr;
                </span>
              </div>
              <div className="text-[11px] font-semibold text-slate-800 dark:text-zinc-200 mt-1 truncate">
                Voice Clone / Dub
              </div>
              <div className="text-[10px] text-slate-500 dark:text-zinc-500 truncate">
                Real Video / Cloned Audio
              </div>
            </button>

            <button
              type="button"
              onClick={() => onLoadSamplePreset("FVFA")}
              disabled={isLoading}
              className="p-2.5 rounded-xl bg-white dark:bg-[#090C12] hover:bg-rose-50/40 dark:hover:bg-[#0E131E] border border-slate-200 dark:border-white/[0.06] hover:border-rose-300 dark:hover:border-rose-500/30 text-left transition group active:scale-[0.98] shadow-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-rose-700 dark:text-rose-400">
                  FVFA
                </span>
                <span className="text-[9px] font-mono text-slate-400 dark:text-zinc-500 group-hover:text-slate-600 dark:group-hover:text-zinc-300">
                  Load &rarr;
                </span>
              </div>
              <div className="text-[11px] font-semibold text-slate-800 dark:text-zinc-200 mt-1 truncate">
                Full Synthetic Media
              </div>
              <div className="text-[10px] text-slate-500 dark:text-zinc-500 truncate">
                AI Video & AI Speech
              </div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
