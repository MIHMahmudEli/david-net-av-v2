/**
 * Typed client & models for DAVID-Net Forensic Microservice.
 * Corresponds to docs/05_deployment.md & Chapter 2, Section 2.1.5.
 */

export type ModalityVerdict = {
  verdict: "real" | "fake" | "unavailable";
  confidence: number | null;
  raw_prob?: number;
};

export type QuadrantInfo = {
  label: "RVRA" | "RVFA" | "FVRA" | "FVFA";
  description: string;
  probs: {
    RVRA: number;
    RVFA: number;
    FVRA: number;
    FVFA: number;
  };
};

export type TemporalInterval = {
  start: number;
  end: number;
  score: number;
};

export type PredictResponse = {
  clip_id: string;
  duration_sec: number;
  modalities: {
    video: boolean;
    audio: boolean;
  };
  video: ModalityVerdict;
  audio: ModalityVerdict;
  quadrant: QuadrantInfo | null;
  localization: {
    video: TemporalInterval[];
    audio: TemporalInterval[];
  };
  sync_curve: number[];
  disclaimer: string;
  model_version: string;
  backbone_mode?: string;
  latency_ms: number;
  explain?: {
    note?: string;
    token_saliency_v?: number;
    token_saliency_a?: number;
  };
};

export type HealthResponse = {
  status: string;
  model_version?: string;
  backbone_mode?: string;
  device?: string;
  max_upload_bytes?: number;
};

export const DEFAULT_API_URL = "https://davidnet-api.onrender.com";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const custom = localStorage.getItem("david_api_base");
    if (custom) return custom.replace(/\/+$/, "");
  }
  const envUrl = process.env.NEXT_PUBLIC_API_URL || process.env.API_URL;
  if (envUrl) return envUrl.replace(/\/+$/, "");

  return DEFAULT_API_URL;
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/health`);
  if (!res.ok) throw new Error(`Health check error: ${res.status}`);
  return await res.json();
}

export async function uploadMediaForPrediction(
  file: File,
  explain: boolean = false
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const isAudio =
    file.type.startsWith("audio") ||
    /\.(wav|mp3|flac|aac|m4a)$/i.test(file.name);
  
  // Directly upload to backend to bypass Vercel's 4.5 MB Serverless Function payload limit
  const base = getApiBaseUrl();
  const endpoint = isAudio ? `${base}/predict-audio` : `${base}/predict`;

  const res = await fetch(`${endpoint}?explain=${explain}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let errMsg = `Inference server error (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errMsg = errJson.detail;
    } catch {}
    throw new Error(errMsg);
  }

  return await res.json();
}
