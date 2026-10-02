# DAVID-Net: Disentangled Audio-Visual Forensics for Deepfake Attribution and Localization

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Frontend-Next.js%2014-black.svg)](https://nextjs.org/)

Official open-source PyTorch implementation and deployment suite for **DAVID-Net** (*Disentangled Audio-Visual Deepfake Network*), a multimodal forensic framework designed for fine-grained authenticity attribution, cross-modal synchronization verification, and temporal forgery localization in synthetic media.

---

## 📌 Overview

Forensic analysis of synthetic media faces a foundational challenge: conventional detectors collapse multimodal evidence into a binary "real or fake" label. This reduction discards critical attribution details—failing to identify *which* modality was altered (visual, acoustic, or both) and *when* the manipulation occurred.

Since modern generative pipelines synthesize facial reenactment/video and cloned speech independently, media can fall into any of **four authenticity quadrants**:

| Quadrant | Video | Audio | Real-World Scenario |
| :---: | :---: | :---: | :--- |
| **RVRA** | Real | Real | Authentic recording |
| **RVFA** | Real | **Fake** | Voice cloning / neural speech synthesis dubbed over pristine footage |
| **FVRA** | **Fake** | Real | Face swap / facial reenactment combined with authentic speech |
| **FVFA** | **Fake** | **Fake** | Fully synthetic media (synthetic avatar + cloned speech) |

### Key Innovations

1. **Four-Quadrant Attribution**: Evaluates visual and acoustic authenticity separately and assigns clips to one of the four authenticity quadrants without sacrificing unimodal sensitivity.
2. **Subspace Disentanglement**: Separates modality-specific artifact representations from cross-modal synchronization representations. This enables robust detection of internally consistent double-fakes (**FVFA**) that deceive synchronization-only models.
3. **Quadrant-Aware Contrastive Pretraining (QACP)**: A self-supervised pretraining scheme that generates all four authenticity quadrants from pristine recordings alone using self-blended video and vocoder copy-synthesis speech, enforcing generalization across unseen generator families.
4. **Temporal & Spatial Localization**: Generates frame-level visual anomaly heatmaps and speech chunk manipulation timelines for fine-grained forensic analysis.
5. **Production Deployment**: Includes a high-throughput FastAPI inference microservice and an interactive Next.js 14 web client.

---

## 🔬 Empirical Highlights

Extensively evaluated on **FakeAVCeleb** and validated via zero-shot cross-dataset evaluation across six independent benchmarks (**ASVspoof 2019 LA**, **In-the-Wild Audio**, **Celeb-DF v2**, **DFDC-10**, **DeepFakeTIMIT**, and **WaveFake**):

* **In-Domain Performance (FakeAVCeleb)**:
  * **Clip AUC**: $0.976 \pm 0.006$
  * **Video AUC**: $0.977 \pm 0.003$
  * **Audio AUC**: $0.999 \pm 0.002$
  * **Quadrant Macro-$F_1$**: $91.0 \pm 1.9\%$
* **Zero-Shot Generalization**:
  * $+18.5\%$ AUC on **ASVspoof 2019 LA** over specialized baselines ($0.833$ vs. $0.648$)
  * $+14.0\%$ AUC on **In-the-Wild Audio** ($0.703$ vs. $0.563$)
  * $+8.3\%$ AUC on **Celeb-DF v2** ($0.672$ vs. $0.589$)
* **Cross-Generator Robustness (LOGO)**:
  * $\ge 90.5\%$ Clip AUC (mean $0.919$) across all withheld generator families under Leave-One-Generator-Out evaluation.

---

## 📁 Repository Structure

```
├── api/                  # FastAPI inference service & standalone web interface
│   ├── app.py            # Microservice endpoints & health checks
│   ├── inference.py      # Preprocessing & inference pipeline
│   ├── requirements.txt  # API dependencies
│   └── templates/        # Built-in lightweight HTML preview interface
├── configs/              # Reproducible experiment configurations
│   ├── david_net.yaml    # Full multimodal model config
│   ├── david_net_lite.yaml # Lightweight distilled model config
│   └── qacp.yaml         # QACP self-supervised pretraining config
├── dataset_recon/        # Standard manifest schemas for benchmark corpora
├── scripts/              # Reproduction, evaluation, and benchmark scripts
│   ├── run_experiments.py # Multi-seed experiment executor
│   ├── aggregate_results.py # Statistical aggregation & bootstrap confidence
│   └── build_manifests.py # Dataset partition & manifest generator
├── src/                  # Core library source code
│   ├── models/           # DAVID-Net architecture, encoders, fusion, & sync modules
│   ├── data/             # Preprocessing, face extraction, audio decoding, & QACP
│   ├── training/         # Multi-task loss functions, QACP pretrainer, & trainer
│   ├── eval/             # Forensic metrics, quadrant attribution, & ROC/DeLong
│   ├── pipeline/         # End-to-end training and evaluation orchestration
│   └── utils/            # Seeding, coordination, and configuration loaders
├── tests/                # Automated unit and integration smoke tests
├── ui/                   # Production Next.js 14 forensic dashboard
│   ├── app/              # App router & views
│   ├── components/       # Timeline gauges, waveform displays, & heatmaps
│   └── package.json      # UI dependencies
├── docker-compose.yml    # Full-stack container orchestration
├── Dockerfile            # Container definition for model serving
├── requirements.txt      # Core Python dependencies
└── LICENSE               # MIT License
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/MIHMahmudEli/david-net-av-v2.git
cd david-net-av-v2

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify Installation (Smoke Test)

Run the automated test suite to confirm model forward/backward passes and loss computations:

```bash
pytest tests/
```

Or run an end-to-end dry run:

```bash
python -m src.models.david_net
```

---

## 💻 Usage

### Python Inference API

```python
import torch
from src.models.david_net import DavidNet
from src.utils.config import load_config

# Load configuration and model architecture
config = load_config("configs/david_net.yaml")
model = DavidNet.from_config(config)
model.eval()

# Dummy input tensors (Batch=1, Frames=16, Channels=3, H=224, W=224; Audio=16000 samples)
video_frames = torch.randn(1, 16, 3, 224, 224)
audio_waveform = torch.randn(1, 16000)

with torch.no_grad():
    outputs = model(video_frames, audio_waveform)

print("Video Authenticity Score:", outputs["p_video"].item())
print("Audio Authenticity Score:", outputs["p_audio"].item())
print("Quadrant Probabilities (RVRA, RVFA, FVRA, FVFA):", outputs["quadrant_probs"].tolist())
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

## 🔬 Training and Evaluation

### Stage 0: Quadrant-Aware Contrastive Pretraining (QACP)

Train the representation space using self-supervised proxy forgeries constructed from clean data:

```bash
python -m src.training.pretrain_qacp --config configs/qacp.yaml
```

### Stage 1: Supervised Multi-Task Training

Train the dual-branch transformer with cross-modal sync and orthogonal disentanglement:

```bash
python -m src.training.train --config configs/david_net.yaml
```

### Benchmark Evaluation

Evaluate on a manifest split (reporting Clip AUC, Modality AUCs, and Quadrant Macro-$F_1$):

```bash
python -m src.eval.evaluate --config configs/david_net.yaml --manifest src/data/splits/train.jsonl
```

---

## ⚖️ Ethics & Responsible AI Statement

DAVID-Net is developed for defensive digital forensics, journalism, fact-checking, and content moderation to mitigate the misuse of synthetic media. Model outputs provide calibrated decision support with associated uncertainty estimates; forensic practitioners should use these metrics in conjunction with broader contextual analysis.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
