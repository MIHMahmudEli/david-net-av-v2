# Reproducibility & Peer-Review Verification Guide

This document defines the verification tiers, statistical protocols, and disclosure boundaries for the empirical findings reported in the DAVID-Net research paper.

---

## 1. Disclosure Policy & Intellectual Property Boundaries

To protect author intellectual property and maintain compliance with responsible AI release principles, this research repository adheres to a **three-tier disclosure model**:

| Tier | Status | Included Artifacts |
| :--- | :---: | :--- |
| **Tier 1: Public Reviewer Assets** | **Fully Disclosed** | Audited benchmark metric tables (`tables/*.csv`), empirical visualization panels (`figures/*.png`), high-level architecture designs, and live interactive serving demonstration. |
| **Tier 2: Verification Assets** | **Available upon Request** | Pre-extracted representation hashes, evaluation split partition manifests, and checkpoint evaluation artifacts via institutional request. |
| **Tier 3: Proprietary Research Assets** | **Protected Locally** | Distributed multi-GPU training orchestration, private generator-proxy synthesis engines, internal hyperparameter sweep harnesses, and full proprietary training code. |

---

## 2. Statistical Protocols & Evaluation Integrity

All quantitative claims reported in the manuscript and included in [`tables/`](../tables/) adhere to rigorous empirical standards:

1. **Strict Identity-Disjoint Partitioning**: All in-domain splits on FakeAVCeleb enforce **0.0% train/test subject identity overlap** ($100\%$ leak-free subject separation).
2. **Multi-Seed Verification ($N = 5$)**: Results are reported as sample mean $\pm$ sample standard deviation across five locked random seeds:
   $$\mathcal{S} \in \{7, 42, 123, 456, 2024\}$$
3. **Statistical Significance Testing**: DeLong tests evaluate Area Under the ROC Curve (AUC) pairwise against unimodal and late-fusion baselines under family-wise Holm--Bonferroni error correction ($p < 0.05$ across all five seeds).
4. **Calibration Measurement**: Visual and acoustic branch probabilities are evaluated using Expected Calibration Error (ECE) over 15 uniform probability bins.

---

## 3. Audited In-Domain Benchmark Summary (FakeAVCeleb)

The table below summarizes performance on the FakeAVCeleb test split (source data: [`tables/main_results.csv`](../tables/main_results.csv)):

| Model | Clip AUC | Video AUC | Audio AUC | Quadrant Macro-$F_1$ (%) | Video ECE | Audio ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AASIST** (Audio baseline) | $0.748 \pm 0.004$ | — | $0.999 \pm 0.001$ | — | — | $0.012$ |
| **EfficientNet-B4** (Video baseline) | $0.926 \pm 0.002$ | $\mathbf{0.995 \pm 0.001}$ | — | — | $0.021$ | — |
| **Late Fusion** | $0.918 \pm 0.004$ | $0.932 \pm 0.005$ | $0.999 \pm 0.001$ | $84.2 \pm 0.8$ | $0.058$ | $0.028$ |
| **DAVID-Net (Proposed)** | $\mathbf{0.976 \pm 0.006}$ | $0.979 \pm 0.003$ | $\mathbf{0.999 \pm 0.002}$ | $\mathbf{91.8 \pm 1.7}$ | $\mathbf{0.032}$ | $\mathbf{0.024}$ |

### Pairwise DeLong Significance vs. Baselines
- DAVID-Net vs. Late Fusion: $p = 0.0009$ (Holm-corrected, $5/5$ seeds significant)
- DAVID-Net vs. EfficientNet-B4: $p = 0.0017$ (Holm-corrected, $5/5$ seeds significant)
- DAVID-Net vs. AASIST: $p < 0.0001$ (Holm-corrected, $5/5$ seeds significant)

---

## 4. Out-of-Domain Zero-Shot Evaluation

Models trained exclusively on FakeAVCeleb were evaluated zero-shot across six unseen external benchmarks without adaptation (source data: [`tables/cross_dataset_results.csv`](../tables/cross_dataset_results.csv)):

- **ASVspoof 2019 LA**: $0.839 \pm 0.019$ AUC ($+19.1$ point gain over AASIST baseline)
- **In-the-Wild Audio**: $0.691 \pm 0.021$ AUC ($+12.2$ point gain over unimodal audio)
- **Celeb-DF v2**: $0.672 \pm 0.014$ AUC
- **DFDC-10**: $0.647 \pm 0.018$ AUC
- **DeepFakeTIMIT**: $0.998 \pm 0.001$ Detection Rate (DR)
- **WaveFake**: $0.991 \pm 0.003$ Detection Rate (DR)

---

## 5. Leave-One-Generator-Out (LOGO) Generalization

To test resilience against unseen generation techniques, entire generator families were withheld during training (source data: [`tables/logo_results.csv`](../tables/logo_results.csv)):

| Withheld Generator Family | Manipulation Type | Retained Clip AUC |
| :--- | :--- | :---: |
| **FSGAN** | Face Swapping | $0.903 \pm 0.012$ |
| **Wav2Lip** | Lip-Sync Reenactment | $0.924 \pm 0.008$ |
| **RTVC** | Neural Voice Cloning | $0.951 \pm 0.005$ |

**Resilience Floor**: Even when facial manipulation or speech synthesis mechanisms are completely novel, cross-modal attribution retains an out-of-domain floor $\ge 90.0\%$ Clip AUC.

---

## 6. Live Interactive Verification

Reviewers can verify inference behavior, four-quadrant attribution, and missing-modality handling via our interactive web demonstration:
- **Deployment URL**: [https://david-net-av.vercel.app](https://david-net-av.vercel.app)
- **Evidentiary Notice**: Inference outputs represent calibrated risk signals designed to assist human analysts and are not intended as autonomous legal judgments.
