# Reviewer Guide: DAVID-Net AV

Welcome to the reviewer-facing open-source repository for **DAVID-Net** (*Decoupled Audio-Visual Deepfake Network*).

This guide provides academic peer reviewers with a structured index mapping manuscript claims to software components, automated tests, empirical tables, and interactive demonstration interfaces.

---

## 1. Quick Navigation: Manuscript Claims to Code & Evidence Artifacts

| Paper Contribution / Claim | Software Component | Evidence & Test Artifacts |
| :--- | :--- | :--- |
| **Four-Quadrant Attribution (RVRA, RVFA, FVRA, FVFA)** | [`src/models/david_net.py`](src/models/david_net.py)<br>[`src/training/losses.py`](src/training/losses.py) | [`tables/main_results.csv`](tables/main_results.csv)<br>[`figures/results_confusion.png`](figures/results_confusion.png) |
| **Orthogonal Subspace Disentanglement** | [`src/models/david_net.py`](src/models/david_net.py)<br>[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | [`figures/fig_davidnet.png`](figures/fig_davidnet.png)<br>[`figures/fig_sync.png`](figures/fig_sync.png) |
| **Missing-Modality Tolerance via Null Tokens** | [`src/models/david_net.py`](src/models/david_net.py)<br>(`null_video_token`, `null_audio_token`) | [`tables/main_results.csv`](tables/main_results.csv)<br>Interactive UI silent mode test |
| **Temporal Action Localization (TAL)** | [`src/eval/localization.py`](src/eval/localization.py) | [`tests/test_localization_eval.py`](tests/test_localization_eval.py) |
| **Multi-Seed Verification ($N = 5$ Seeds)** | [`scripts/aggregate_results.py`](scripts/aggregate_results.py) | [`tables/main_results.csv`](tables/main_results.csv)<br>[`figures/results_roc.png`](figures/results_roc.png) |
| **Cross-Generator Resilience (LOGO)** | [`src/eval/evaluate.py`](src/eval/evaluate.py) | [`tables/logo_results.csv`](tables/logo_results.csv)<br>[`figures/results_logo.png`](figures/results_logo.png) |
| **Zero-Shot Transfer Across 6 External Benchmarks** | [`src/pipeline/evaluation.py`](src/pipeline/evaluation.py) | [`tables/cross_dataset_results.csv`](tables/cross_dataset_results.csv) |
| **Probability Calibration (ECE)** | [`src/eval/metrics.py`](src/eval/metrics.py) | [`figures/results_reliability.png`](figures/results_reliability.png) |
| **Negative Finding on Contrastive Pretraining (QACP)** | [`src/training/pretrain_qacp.py`](src/training/pretrain_qacp.py) | [`tables/ablation_results.csv`](tables/ablation_results.csv) (row: `no_qacp`) |

---

## 2. Reviewer Reproducibility & Automated Testing

Reviewers can verify code correctness and architectural forward passes locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run core unit tests (model forward passes, losses, preprocessing)
pytest tests/test_smoke.py tests/test_loss_floor.py tests/test_preprocess.py tests/test_clipcache.py
```

### Verification Without GPU Hardware
The core test suite includes lightweight CPU smoke tests and synthetic tensors (`tests/test_smoke.py`, `tests/smoke_test.py`) allowing reviewers to verify network instantiation, dimension compatibility, and loss computation in seconds without specialized GPU hardware.

---

## 3. Interactive Model Verification

Peer reviewers can test live model behavior directly using our production deployment:

- **Web Demonstration**: [https://david-net-av.vercel.app](https://david-net-av.vercel.app)
- **Features Available for Verification**:
  - Live video/audio upload with automatic face alignment and audio extraction.
  - Decoupled visual authenticity ($P_{\text{video}}$) and acoustic authenticity ($P_{\text{audio}}$) scores.
  - Four-quadrant classification badges ($\text{RVRA}$, $\text{RVFA}$, $\text{FVRA}$, $\text{FVFA}$).
  - Dynamic missing-modality handling (silent video detection routes inference through the visual head without acoustic hallucination).

---

## 4. Scope of Open-Source Release & IP Protection

This repository fulfills our open-source research commitment by providing:
1. **Complete Model Implementation**: Full model architectures, loss formulations, feature extractors, and evaluation scripts.
2. **Serving & Deployment**: Complete FastAPI backend and Next.js frontend code.
3. **Reproducibility Metadata**: Audited benchmark tables, dataset schema configurations, and configuration YAMLs.

**Preserved Privately / Withheld**:
- Multi-account cloud scraping/scheduling cluster infrastructure (`src/orchestrator/`, `src/scheduler/`) used during the private 135-experiment campaign.
- University internal grading rubrics, thesis LaTeX sources, and evaluation forms.
- Heavy binary checkpoint weights (weights are hosted publicly on Hugging Face Hub: [`MIHMahmudEli/david-net-av-v2`](https://huggingface.co/MIHMahmudEli/david-net-av-v2) rather than bloated Git history).
