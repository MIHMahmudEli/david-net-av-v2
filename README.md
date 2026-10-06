# DAVID-Net: Disentangled Audio-Visual Forensics for Deepfake Attribution and Localization

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Artifacts%20%26%20Weights-orange)](https://huggingface.co/MIHMahmudEli/david-net-av-v2)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Frontend-Next.js%2014-black.svg)](https://nextjs.org/)

Official open-source research implementation and deployment suite for **DAVID-Net** (*Disentangled Audio-Visual Deepfake Network*), a multimodal forensic framework designed for fine-grained authenticity attribution, cross-modal synchronization verification, and temporal manipulation localization in synthetic media.

Pretrained model weights, evaluation logs, split manifests, and raw experiment metrics are hosted on [Hugging Face Hub (`MIHMahmudEli/david-net-av-v2`)](https://huggingface.co/MIHMahmudEli/david-net-av-v2).

---

## Overview

Conventional deepfake detectors collapse multimodal forensic evidence into a monolithic binary label ("real" vs. "fake"). This reduction discards critical attribution details: it fails to determine *which* modality was manipulated (visual, acoustic, or both) and *when* the alteration took place.

Because modern generative pipelines produce facial reenactment/video synthesis and voice cloning independently, real-world media can fall into any of **four authenticity quadrants**:

| Quadrant | Video | Audio | Real-World Scenario |
| :---: | :---: | :---: | :--- |
| **RVRA** | Real | Real | Authentic recording |
| **RVFA** | Real | **Fake** | Voice cloning / neural speech synthesis dubbed over pristine footage |
| **FVRA** | **Fake** | Real | Face swap / facial reenactment combined with authentic speech |
| **FVFA** | **Fake** | **Fake** | Fully synthetic media (synthetic avatar + cloned speech) |

### Key Architectural Pillars

1. **Four-Quadrant Attribution**: Disentangles visual and acoustic decision heads, producing calibrated individual modality probabilities alongside a joint four-class authenticity quadrant distribution.
2. **Subspace Disentanglement**: Projects features into modality-specific artifact subspaces and a dedicated cross-modal synchronization subspace with an explicit orthogonality constraint. This prevents cross-modal shortcuts and enables robust detection of internally consistent double-fakes (**FVFA**).
3. **Missing-Modality Fault Tolerance**: Employs dedicated null-token projection pathways that ensure stable unimodal performance when an audio or video stream is missing or corrupted.
4. **Temporal Action Localization (TAL)**: Generates frame-level visual anomaly scores and chunk-level acoustic manipulation timelines for temporal forgery boundary detection.
5. **Production Deployment**: Includes a high-throughput FastAPI inference microservice and an interactive Next.js 14 verification interface.

---

## Empirical Verification (5-Seed Campaign)

The experimental findings reflect a 5-seed benchmark evaluation ($\mathcal{S} \in \{7, 42, 123, 456, 2024\}$) under strict identity-disjoint partitioning (0.0% train/test subject identity overlap):

* **In-Domain Performance (FakeAVCeleb)**:
  * **Clip AUC**: $0.976 \pm 0.006$
  * **Video AUC**: $0.977 \pm 0.003$
  * **Audio AUC**: $0.999 \pm 0.002$
  * **Quadrant Macro-$F_1$**: $91.0 \pm 1.9\%$
  * **Calibration (ECE, 15 bins)**: $0.032$ (visual), $0.024$ (acoustic)
* **Zero-Shot Cross-Dataset Transfer**:
  * **ASVspoof 2019 LA**: $0.839 \pm 0.019$ AUC
  * **In-the-Wild Audio**: $0.691 \pm 0.021$ AUC
  * **Celeb-DF v2**: $0.672 \pm 0.014$ AUC
* **Cross-Generator Generalization (LOGO)**:
  * $\ge 90.3\%$ Clip AUC across all withheld generator families under Leave-One-Generator-Out evaluation.
* **Missing-Modality Resilience**:
  * Audio muted (null audio token): $0.976 \pm 0.005$ Video AUC
  * Video blanked (null video token): $0.999 \pm 0.001$ Audio AUC

Full machine-readable results, seed-level outputs, DeLong test statistics, and configuration files are archived on [Hugging Face](https://huggingface.co/MIHMahmudEli/david-net-av-v2).

---

## Repository Structure

```text
├── api/                  # FastAPI inference microservice & standalone web endpoints
│   ├── app.py            # FastAPI application routing & health checks
│   ├── inference.py      # Preprocessing & inference pipeline
│   ├── requirements.txt  # API service dependencies
│   └── templates/        # Lightweight server-rendered interface
├── configs/              # Reproducible experiment and architecture configurations
│   ├── david_net.yaml    # Full multimodal model architecture config
│   ├── david_net_kaggle.yaml # Pinned GPU environment training config
│   ├── david_net_lite.yaml   # Lightweight distilled deployment config
│   ├── qacp.yaml         # Stage-0 contrastive pretraining config
│   └── CODE_REVISION.json # Cryptographic git revision pin map
├── dataset_recon/        # Standard manifest schemas for benchmark corpora
├── kaggle/               # Automated batch kernel builders and pipelines
├── scripts/              # Reproduction, evaluation, and benchmark scripts
│   ├── aggregate_results.py      # Statistical aggregation & bootstrap confidence
│   ├── build_manifests.py        # Dataset partition & manifest generator
│   ├── eval_robustness_sweep.py  # Parametric perturbation robustness harness
│   └── publish_hf_artifacts.py   # Hugging Face Hub synchronization utility
├── src/                  # Core library source code
│   ├── models/           # DAVID-Net architecture, encoders, fusion, & sync modules
│   ├── data/             # Preprocessing, face extraction, audio decoding, & augmentation
│   ├── training/         # Multi-task loss functions, QACP pretrainer, & trainer
│   ├── eval/             # Forensic metrics, quadrant attribution, TAL, & robustness
│   ├── orchestrator/     # Distributed multi-worker job scheduling & database
│   ├── pipeline/         # End-to-end execution, checksum manifests, & hub utilities
│   └── utils/            # Seeding, watchdog, configuration, & storage helpers
├── tests/                # Automated unit and integration test suite (182+ tests)
├── ui/                   # Next.js 14 forensic verification web dashboard
│   ├── app/              # App router & views
│   ├── components/       # Dual timeline, waveform sync, and verdict displays
│   └── package.json      # UI dependencies
├── docker-compose.yml    # Full-stack container orchestration
├── Dockerfile            # Container definition for model serving
├── requirements.txt      # Core Python dependencies
└── LICENSE               # MIT License
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/MIHMahmudEli/david-net-av-v2.git
cd david-net-av-v2

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify Installation (Test Suite)

Run the automated test suite to confirm model forward passes, loss computations, and pipeline components:

```bash
python -m pytest tests/
```

Expected result: `182 passed, 2 skipped`.

---

## Usage

### Python Inference API

```python
import torch
from src.models.david_net import DavidNet, DavidNetConfig

# Initialize model architecture
config = DavidNetConfig()
model = DavidNet(config)
model.eval()

# Synthetic input tensors: 16 video frames (3x224x224) and 1-second audio (16kHz)
video_frames = torch.randn(1, 16, 3, 224, 224)
audio_waveform = torch.randn(1, 16000)

with torch.no_grad():
    outputs = model(video_frames, audio_waveform)

print("Video Authenticity Probability:", outputs["p_video"].item())
print("Audio Authenticity Probability:", outputs["p_audio"].item())
print("Quadrant Probabilities [RVRA, RVFA, FVRA, FVFA]:", outputs["quadrant_probs"].tolist())
```

### Downloading Pretrained Weights

Pretrained safetensors weights (`david_net_best.safetensors`, ~341 MB) can be loaded directly from Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

weights_path = hf_hub_download(
    repo_id="MIHMahmudEli/david-net-av-v2",
    filename="models/david_net_best.safetensors"
)
state_dict = load_file(weights_path)
model.load_state_dict(state_dict, strict=False)
```

### Running the API Microservice

Launch the FastAPI backend locally:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:7860/docs`.

### Running the Interactive Web UI

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:3000` to interact with the forensic verification dashboard.

### Full-Stack Docker Deployment

Run both the API backend and frontend interface via Docker Compose:

```bash
docker-compose up --build
```

---

## Training and Evaluation

### Stage 0: Quadrant-Aware Contrastive Pretraining (QACP)

```bash
python -m src.training.pretrain_qacp --config configs/qacp.yaml
```

### Stage 1: Supervised Multimodal Training

```bash
python -m src.training.train --config configs/david_net.yaml
```

### Benchmark Evaluation

Evaluate on a manifest split (reporting Clip AUC, Modality AUCs, and Quadrant Macro-$F_1$):

```bash
python -m src.eval.evaluate --config configs/david_net.yaml --manifest src/data/splits/train.jsonl
```

---

## Ethics & Responsible AI Statement

DAVID-Net is designed strictly for defensive digital forensics, media verification, journalism, and content authenticity verification. The system contains no generative or media synthesis capabilities. Operating outputs provide calibrated probabilistic evidence with explicit evidentiary disclaimers to support human forensic practitioners rather than issuing autonomous legal or forensic determinations.

---

## License

This project is licensed under the [MIT License](LICENSE).
