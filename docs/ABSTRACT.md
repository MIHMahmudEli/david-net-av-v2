# Abstract & Key Contributions

## Abstract

For media analysts, forensics investigators, fact-checking desks, and platform moderation teams, binary "real or fake" classifications offer little diagnostic value. When deepfake generation decouples facial video manipulation from acoustic cloning, binary labels cannot reveal which modality was altered, when tampering occurred, or catch internally consistent double-fake media ($\text{FVFA}$) where synthetic video is aligned to cloned speech. 

This research reformulates audio-visual deepfake detection from binary classification into a four-quadrant attribution task: classifying video-and-audio pairs into **Real Video with Real Audio (RVRA)**, **Real Video with Fake Audio (RVFA)**, **Fake Video with Real Audio (FVRA)**, and **Fake Video with Fake Audio (FVFA)**. 

To address this challenge, we developed **DAVID-Net** (*Decoupled Audio-Visual Forensic Network*), an asymmetric cross-modal transformer projecting representations onto orthogonal complements in synchronization space. This architecture isolates modality-specific forgery traces from cross-modal phoneme-viseme alignment dynamics. Predictive heads provide visual veracity, acoustic veracity, four-way categorical attribution, and temporal tampering boundaries, while learnable null tokens handle single-modality inputs (e.g., muted footage or isolated voice memos) without pipeline collapse.

Across a 135-experiment campaign over five locked random seeds $\mathcal{S} \in \{7, 42, 123, 456, 2024\}$ on FakeAVCeleb under strict subject-disjoint splits, DAVID-Net achieved $0.976 \pm 0.006$ Clip AUC, $0.979 \pm 0.003$ Video AUC, $0.999 \pm 0.002$ Audio AUC, and $91.8 \pm 1.7\%$ Quadrant Macro-$F_1$. Two-sided DeLong tests with Holm--Bonferroni correction confirm statistically defensible improvements over Late Fusion, EfficientNet-B4, and AASIST across all five seeds ($p < 0.05$). Under zero-shot transfer across six external benchmarks, the model demonstrated notable transferability ($+19.1$ percentage points on ASVspoof 2019 LA and $+12.2$ points on In-the-Wild Audio) while preserving an out-of-domain resilience floor $\ge 90.0\%$ Clip AUC under Leave-One-Generator-Out (LOGO) evaluation. Five-seed ablations demonstrate straightforwardly that QACP offers no statistically provable performance edge within this data scale (0/5 Holm-significant seeds), establishing an important negative finding.

---

## Key Research Contributions

1. **Four-Quadrant Attribution Formulation**: Reframing audio-visual deepfake detection from binary classification to four-quadrant attribution (RVRA, RVFA, FVRA, FVFA) alongside millisecond temporal boundary localization, overcoming the diagnostic blind spots of binary classifiers.
2. **Subspace Disentanglement Architecture**: Designing an asymmetric cross-modal transformer projecting features onto orthogonal complements of synchronization space, preventing shortcut memorization and identifying internally synchronized double-fake manipulations.
3. **Missing-Modality Tolerance via Null Tokens**: Integrating learnable null-token projection pathways, allowing a single unified model to process pure audio (voice memos) or pure video (muted footage) without requiring separate single-modality pipelines.
4. **Rigorous Five-Seed Statistical Verification**: Executing a 135-run benchmark matrix across five seeds $\mathcal{S} \in \{7, 42, 123, 456, 2024\}$ with DeLong test verification, demonstrating statistically significant advantages over unimodal and late-fusion baselines.
5. **Cross-Generator and Cross-Dataset Generalization**: Establishing out-of-domain resilience across six external transfer corpora (ASVspoof 2019 LA, In-the-Wild Audio, Celeb-DF v2, DFDC-10, DeepFakeTIMIT, WaveFake) and Leave-One-Generator-Out evaluation.
6. **Transparent Reporting of Negative Findings**: Honestly reporting that generator-free contrastive pretraining (QACP) imparts no statistically provable performance advantage at this data scale, providing a cautionary insight for future multimodal forensic research.
