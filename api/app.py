"""FastAPI service and Web Dashboard for DAVID-Net Deepfake Detection.

Features:
- GET  /              -> Interactive Web UI (Upload media, view 4-quadrant attribution,
                         dual timelines, sync curve, and confidence gauges).
- GET  /health        -> Health check & engine status.
- GET  /version       -> Model version, parameters, and architecture details.
- POST /predict       -> Multipart video upload for multimodal forgery detection.
- POST /predict-audio -> Multipart audio upload for standalone speech deepfake detection.

Run locally:
  uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(
    title="DAVID-Net Deepfake Detection Microservice",
    description="Multimodal Disentangled Deepfake Attribution & Temporal Localization API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50 MB
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from api.inference import DavidNetInference
        ckpt = os.environ.get("DAVID_CHECKPOINT", "checkpoints/david_net_best.safetensors")
        cfg = os.environ.get("DAVID_CONFIG", "configs/david_net.yaml")
        _engine = DavidNetInference(checkpoint=ckpt, config=cfg)
    return _engine


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi import Response
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the interactive web interface."""
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DAVID-Net API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>"


@app.get("/health")
def health():
    """Liveness probe that returns immediately (<1ms) to keep cloud health checks green."""
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "model_version": _engine.version if _engine else "DAVID-Net-v1.0 (Ready)",
        "backbone_mode": _engine.backbone_mode if _engine else "Dynamic Auto-Select",
        "device": str(_engine.device) if _engine else "cpu",
        "max_upload_bytes": MAX_BYTES,
    }


@app.get("/version")
def version():
    """Model version and architecture parameters."""
    engine = get_engine()
    total_params = sum(p.numel() for p in engine.model.parameters())
    return {
        "model_version": engine.version,
        "backbone_mode": engine.backbone_mode,
        "architecture": "DAVID-Net (Disentangled Cross-Attention)",
        "d_model": engine.cfg_dict["model"].get("d_model", 768),
        "n_heads": engine.cfg_dict["model"].get("n_heads", 8),
        "n_fusion_layers": engine.cfg_dict["model"].get("n_fusion_layers", 4),
        "total_parameters": total_params,
        "device": str(engine.device),
    }


async def _save_upload(file: UploadFile, default_suffix: str) -> str:
    suffix = os.path.splitext(file.filename or f"clip{default_suffix}")[1] or default_suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total_bytes = 0
    # Stream in 64KB chunks directly to disk to prevent RAM spikes on large media uploads
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_BYTES:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_BYTES // (1024*1024)} MB limit.")
        tmp.write(chunk)
    tmp.close()
    return tmp.name


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = Query(False)):
    """Full multimodal video clip analysis.
    
    Handles silent or missing audio via availability null-token routing.
    """
    if file.content_type is None or not (file.content_type.startswith("video") or file.filename.lower().endswith(('.mp4', '.avi', '.webm', '.mov', '.mkv'))):
        raise HTTPException(status_code=415, detail="Please upload a supported video file (MP4, AVI, WebM, MOV).")

    path = await _save_upload(file, ".mp4")
    try:
        has_audio = True
        try:
            from src.data.preprocess import probe_media
            has_audio = probe_media(path).get("has_audio", True)
        except Exception:
            pass  # Fall back to attempting full multimodal extraction

        result = get_engine().predict(path, explain=explain, has_video=True, has_audio=has_audio)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic inference failed: {e}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        import gc
        gc.collect()


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...), explain: bool = Query(False)):
    """Standalone audio track / voice deepfake analysis."""
    if file.content_type is None or not (file.content_type.startswith("audio") or file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.m4a'))):
        raise HTTPException(status_code=415, detail="Please upload a supported audio file (WAV, MP3, FLAC, AAC).")

    path = await _save_upload(file, ".wav")
    try:
        result = get_engine().predict_audio(path, explain=explain)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio forensic inference failed: {e}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        import gc
        gc.collect()
