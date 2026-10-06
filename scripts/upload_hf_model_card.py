"""Upload professional model card to Hugging Face Hub repository MIHMahmudEli/david-net-av-v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi
from src.scheduler.config import hf_token

MODEL_CARD_CONTENT = """---
license: mit
pipeline_tag: video-classification
library_name: pytorch
tags:
- deepfake-detection
- audio-visual
- multimodal
- digital-forensics
- safetensors
- video-classification
- audio-classification
- fakeavceleb
---

# DAVID-Net: Disentangled Audio-Visual Forensics for Deepfake Attribution and Localization

Official pretrained model weights, execution configurations, dataset manifests, and benchmark evaluation outputs for **DAVID-Net** (*Disentangled Audio-Visual Deepfake Network*).

* **Source Code Repository:** [GitHub: MIHMahmudEli/david-net-av-v2](https://github.com/MIHMahmudEli/david-net-av-v2)
* **License:** MIT License
* **Format:** SafeTensors (`models/david_net_best.safetensors`, ~341 MB)

---

## 📌 Framework Summary

Conventional deepfake detectors collapse multimodal evidence into a single binary output ("real" vs. "fake"). This reduction discards critical attribution details—failing to identify *which* modality was altered (visual, acoustic, or both) and *when* the manipulation occurred.

DAVID-Net formulates synthetic media forensics across **four authenticity quadrants**:

| Quadrant | Video | Audio | Forensic Interpretation |
| :---: | :---: | :---: | :--- |
| **RVRA** | Real | Real | Authentic recording |
| **RVFA** | Real | **Fake** | Voice cloning / neural speech synthesis over authentic footage |
| **FVRA** | **Fake** | Real | Face swap / facial reenactment combined with authentic audio |
| **FVFA** | **Fake** | **Fake** | Fully synthetic media (synthetic avatar + cloned speech) |

### Core Architectural Features

1. **Subspace Disentanglement**: Separates modality-specific artifact features from cross-modal synchronization features using an explicit orthogonality loss. This enables detection of internally synchronized dual-stream fakes (**FVFA**) that deceive synchrony-only detectors.
2. **Four-Quadrant Attribution**: Disentangled decision heads produce individual modality authenticity probabilities ($p_v, p_a$) and a joint four-quadrant distribution.
3. **Missing-Modality Fault Tolerance**: Dedicated null-token projection pathways preserve unimodal detection accuracy when an audio or video stream is severed ($0.976$ Video AUC with audio muted; $0.999$ Audio AUC with video blanked).
4. **Temporal Action Localization (TAL)**: Fine-grained frame-level visual anomaly scoring and speech segment manipulation boundary localization.

---

## 📁 Repository Contents

```text
├── models/
│   ├── config.json                     # Architecture configuration dictionary
│   └── david_net_best.safetensors      # Validated full model weights (~341 MB, SafeTensors)
├── results/
│   ├── main/
│   │   ├── raw_results.csv             # Full raw metrics (105 evaluated / 135 total runs, 5 seeds)
│   │   ├── summary.json                # Summary metrics, DeLong test statistics, 95% bootstrap CIs
│   │   ├── main_results.csv            # In-domain FakeAVCeleb benchmark results
│   │   └── per_class_results.csv       # Four-quadrant per-class precision, recall, F1
│   ├── ablations/
│   │   └── ablation_results.csv        # Systematic ablation study across 11 architectural variants
│   ├── cross_dataset/
│   │   └── cross_dataset_results.csv   # Zero-shot evaluations across six external benchmark datasets
│   ├── logo/
│   │   └── logo_results.csv            # Leave-One-Generator-Out cross-generator transfer metrics
│   ├── subject_disjoint/
│   │   └── split_leakage_audit.csv     # Cryptographic audit confirming 0.0% identity leakage
│   ├── missing_modality/
│   │   └── missing_modality_results.csv# Fault tolerance under audio-muted & video-blanked conditions
│   ├── calibration/
│   │   └── calibration_metrics.csv     # 15-bin Expected Calibration Error (ECE)
│   ├── demographics/
│   │   └── fairness_audit.csv          # Subgroup demographic error rates and sample counts
│   └── statistics/
│       ├── statistical_audit.csv       # Pairwise Wilcoxon & DeLong test p-values (Holm-corrected)
│       └── protocol_comparison.csv     # Strict disjoint protocol vs legacy random split
├── scripts/
│   ├── evaluation/                     # Full evaluation engines (evaluate, metrics, localization, etc.)
│   └── analysis/                       # Results aggregation & robustness sweep harnesses
├── configs/
│   └── experiments/                    # Training and architecture YAML configurations
└── metadata/
    ├── datasets/                       # Benchmark dataset reconciliation manifests & SPLITS_SHA256.json
    └── experiments/                    # Hyperparameters and git revision provenance map
```

---

## 🔬 Benchmark Evaluation Results (5-Seed Campaign)

All results are verified across five random seeds ($\mathcal{S} \in \{7, 42, 123, 456, 2024\}$) under strict identity-disjoint evaluation (0.0% subject overlap):

| Evaluation Benchmark | Video AUC | Audio AUC | Clip AUC | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **FakeAVCeleb (In-Domain)** | **0.977 ± 0.003** | **0.999 ± 0.002** | **0.976 ± 0.006** | Quadrant Acc: 91.0%, RVRA Recall: 82.4% |
| **ASVspoof 2019 LA** | — | **0.839 ± 0.019** | **0.839 ± 0.019** | Audio-only zero-shot transfer |
| **In-the-Wild Audio** | — | **0.691 ± 0.021** | **0.691 ± 0.021** | Audio-only zero-shot transfer |
| **Celeb-DF v2** | **0.672 ± 0.014** | — | **0.672 ± 0.014** | Visual-only zero-shot transfer |
| **DFDC-10** | **0.647 ± 0.020** | **0.552 ± 0.015** | **0.647 ± 0.020** | Multimodal zero-shot transfer |
| **DeepFakeTIMIT** | **0.575 ± 0.021** | — | **0.575 ± 0.021** | Visual-only zero-shot transfer |
| **WaveFake** | — | — | **0.113 ± 0.025** | Unmatched threshold transfer (DR) |

---

## 💻 Quickstart: Loading Pretrained Weights

```python
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 1. Download configuration and weights from Hugging Face
repo_id = "MIHMahmudEli/david-net-av-v2"
config_path = hf_hub_download(repo_id=repo_id, filename="models/config.json")
weights_path = hf_hub_download(repo_id=repo_id, filename="models/david_net_best.safetensors")

with open(config_path, "r") as f:
    config = json.load(f)

# 2. Load safe tensors state dict
state_dict = load_file(weights_path)
print(f"Loaded {len(state_dict)} tensor keys successfully from {weights_path}.")

# 3. Model forward pass (see GitHub repository for complete DavidNet class definition)
# from src.models.david_net import DavidNet, DavidNetConfig
# model = DavidNet(DavidNetConfig(**config))
# model.load_state_dict(state_dict, strict=False)
# model.eval()
```

---

## ⚖️ Ethical & Responsible Use Statement

The models and forensic artifacts in this repository are developed strictly for defensive media authenticity verification, journalism, content moderation, and digital forensics research. The system contains no generative or synthesis capabilities. Operating predictions provide calibrated probabilistic evidence with explicit evidentiary disclaimers to assist human analysts.
"""

def main():
    token = hf_token()
    if not token:
        print("ERROR: Hugging Face token not found in .env")
        sys.exit(1)
        
    api = HfApi(token=token)
    repo_id = "MIHMahmudEli/david-net-av-v2"
    repo_type = "model"
    
    print(f"Uploading professional model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="docs: add professional model card and artifact documentation",
    )
    print("Model card uploaded successfully!")

if __name__ == "__main__":
    main()
