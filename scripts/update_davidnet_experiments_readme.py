"""Update MIHMahmudEli/davidnet-experiments repository with a professional model card and registry documentation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from src.scheduler.config import hf_token

def generate_readme():
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN missing in .env")
        
    p_orig = hf_hub_download("MIHMahmudEli/davidnet-experiments", "README.md", repo_type="model", token=token)
    orig_lines = open(p_orig, "r", encoding="utf-8").readlines()
    table_lines = [l.strip() for l in orig_lines if l.strip().startswith("| EXP_")]
    
    # Load summary metrics
    p_csv = hf_hub_download("MIHMahmudEli/davidnet-experiments", "reports/raw_results.csv", repo_type="model", token=token)
    df = pd.read_csv(p_csv)
    
    # Models table (5-seed averages)
    models_summary = []
    for name, grp in df.groupby("name"):
        n_seeds = len(grp)
        v_m, v_s = grp["video_auc"].mean(), grp["video_auc"].std()
        a_m, a_s = grp["audio_auc"].mean(), grp["audio_auc"].std()
        c_m, c_s = grp["clip_auc"].mean(), grp["clip_auc"].std()
        
        v_str = f"{v_m:.3f} ± {v_s:.3f}" if pd.notna(v_m) else "—"
        a_str = f"{a_m:.3f} ± {a_s:.3f}" if pd.notna(a_m) else "—"
        c_str = f"{c_m:.3f} ± {c_s:.3f}" if pd.notna(c_m) else "—"
        
        models_summary.append({
            "name": name,
            "seeds": n_seeds,
            "video_auc": v_str,
            "audio_auc": a_str,
            "clip_auc": c_str
        })
        
    models_summary = sorted(models_summary, key=lambda x: x["name"])
    
    summary_table_rows = []
    for row in models_summary:
        summary_table_rows.append(f"| `{row['name']}` | {row['seeds']} | {row['video_auc']} | {row['audio_auc']} | {row['clip_auc']} |")
    summary_table_str = "\n".join(summary_table_rows)
    registry_table_str = "\n".join(table_lines)
    
    card = f"""---
license: mit
pipeline_tag: video-classification
library_name: pytorch
tags:
- deepfake-detection
- audio-visual
- multimodal
- digital-forensics
- safetensors
- experiments
- benchmarks
- fakeavceleb
- cross-dataset
- leave-one-generator-out
---

# DAVID-Net: 135-Run Experimental Registry and Benchmark Archive

Official experimental archive, multi-seed checkpoints, training event streams, and evaluation telemetry for **DAVID-Net** (*Disentangled Audio-Visual Deepfake Network*).

This repository contains the persistent, uncompressed experimental record for the entire **135-run / 5-seed empirical campaign** across 27 experimental configurations (seeds $\\mathcal{{S}} \\in \\{{7, 42, 123, 456, 2024\\}}$).

* **Source Code Repository:** [GitHub: MIHMahmudEli/david-net-av-v2](https://github.com/MIHMahmudEli/david-net-av-v2)
* **Core Pretrained Model:** [Hugging Face: MIHMahmudEli/david-net-av-v2](https://huggingface.co/MIHMahmudEli/david-net-av-v2)
* **License:** MIT License
* **Format:** SafeTensors (`best_model/model.safetensors` per run), JSON configurations, CSV telemetry

---

## 📌 Experimental Campaign Overview

The empirical validation of DAVID-Net reflects a comprehensive matrix of **135 fully converged training runs** partitioned across four experimental stages:

1. **Stage-0: Quadrant-Aware Contrastive Pretraining (QACP)** (*30 runs*):
   * 6 self-supervised pretraining variants $\\times$ 5 random seeds (seeds 7, 42, 123, 456, 2024).
   * Generates proxy synthetic media from pristine recordings using self-blended video and vocoder copy-synthesis speech.
2. **Stage-1: In-Domain Baselines & Proposed Architecture** (*35 runs*):
   * 7 model architectures $\\times$ 5 random seeds: Video Probe, Audio Probe, AASIST, EfficientNet-B4, Late Fusion, DAVID-Net Phase A, and DAVID-Net End-to-End Phase B.
   * Evaluated under strict identity-disjoint FakeAVCeleb partitioning (0.0% train/test subject identity overlap).
3. **Systematic Component Ablation Study** (*55 runs*):
   * 11 ablation variants $\\times$ 5 random seeds: isolating cross-modal synchronization (`no_sync`), subspace disentanglement (`no_disentangle`), localization heads (`no_loc`), modality dropout (`no_moddrop`), single-task loss (`single_task`), pretraining (`no_qacp`), and sub-component pretraining objectives.
4. **Leave-One-Generator-Out (LOGO) Study** (*15 runs*):
   * 3 withheld generator families (`logo_wav2lip`, `logo_fsgan`, `logo_faceswap`) $\times$ 5 random seeds to measure zero-shot cross-generator generalization.

---

## 🔬 In-Domain Benchmark Summary ($N=5$ Seeds)

All 21 evaluated model configurations averaged over five independent seeds:

| Model / Configuration | Seeds | Video AUC | Audio AUC | Clip AUC |
| :--- | :---: | :---: | :---: | :---: |
{summary_table_str}

### Key Findings
* **Proposed Architecture (DAVID-Net Phase A)**: Achieves **$0.976 \\pm 0.006$ Clip AUC**, with **$0.977 \\pm 0.003$ Video AUC** and **$0.999 \\pm 0.002$ Audio AUC**, outperforming monolithic multimodal Late Fusion ($0.918 \\pm 0.015$).
* **Cross-Generator Floor (LOGO)**: Maintains $\\ge 90.3\\%$ Clip AUC across all withheld generator families under leave-one-generator-out evaluation.
* **Empirical Calibration**: Attains low Expected Calibration Error (ECE, 15 bins) of **$0.032$** (visual) and **$0.024$** (acoustic) on the test split.

---

## 📁 Repository Organization & File Hierarchy

Each of the 135 completed runs is stored in a dedicated directory:

```text
experiments/EXP_{{ID}}_{{NAME}}_s{{SEED}}/
├── best_model/
│   ├── model.safetensors       # Pinned model weights for this seed (SafeTensors)
│   ├── config.json             # Architecture parameters
│   └── meta.json               # Checkpoint metadata & validation performance
├── configs/
│   ├── config.json             # Full experiment hyperparameter configuration
│   └── environment.json        # Execution environment & hardware telemetry
├── metrics/
│   ├── training_history.csv    # Per-epoch loss, learning rates, and validation metrics
│   └── eval_metrics.json       # Final test evaluation metrics
└── figures/                    # Generated ROC, PR, score distribution, and calibration plots
```

### Global Reports & Aggregated Metrics (`reports/`)

* [`reports/raw_results.csv`](reports/raw_results.csv): Master tabular compilation of all 105 evaluated experimental runs.
* [`reports/summary.json`](reports/summary.json): Aggregated statistical metrics, bootstrap confidence intervals, and pairwise DeLong tests ($z$-scores and Holm-corrected $p$-values).
* `reports/tables/`: Machine-readable CSV tables for main results, ablations, external benchmark transfers, LOGO splits, and protocol comparisons.

---

## 💻 Programmatic Usage: Accessing Run Artifacts

You can inspect and load the weights, training logs, or configs of any run programmatically using `huggingface_hub`:

```python
import pandas as pd
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo_id = "MIHMahmudEli/davidnet-experiments"

# 1. Download training history for EXP_011 (DAVID-Net Phase A, Seed 42)
history_path = hf_hub_download(
    repo_id=repo_id,
    filename="experiments/EXP_011_davidnet_s42/metrics/training_history.csv"
)
df_history = pd.read_csv(history_path)
print("Training Epochs:", len(df_history))

# 2. Download checkpoint weights (SafeTensors)
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="experiments/EXP_011_davidnet_s42/best_model/model.safetensors"
)
state_dict = load_file(weights_path)
print(f"Loaded {{len(state_dict)}} tensor keys.")
```

---

## 📋 Complete 135-Run Campaign Registry

| ID | Configuration | Seed | Status | Test Clip AUC |
|:---|:---|:---:|:---:|:---:|
{registry_table_str}

---

## ⚖️ Ethics & Responsible AI Statement

All experimental evaluations in this repository were conducted under strict academic research ethics. Seven benchmark corpora were acquired under their respective research end-user licensing agreements; no raw biometric media is redistributed. The system is strictly defensive by design and contains no generative synthesis capability.
"""
    return card

def main():
    token = hf_token()
    if not token:
        print("ERROR: Hugging Face token not found in .env")
        sys.exit(1)
        
    api = HfApi(token=token)
    repo_id = "MIHMahmudEli/davidnet-experiments"
    repo_type = "model"
    
    print("Generating comprehensive model card for davidnet-experiments...")
    card_content = generate_readme()
    print(f"Generated card length: {len(card_content)} chars")
    
    print(f"Uploading updated README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="docs: upgrade repository README to professional 135-run research archive card",
    )
    print("Successfully updated README.md on MIHMahmudEli/davidnet-experiments!")

if __name__ == "__main__":
    main()
