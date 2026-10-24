# Study design v2 — pre-registered analysis plan
**Working title:** Discrimination metrics are blind to anatomical and lesion-level grounding: a controlled, multi-architecture ablation of lung masking in chest radiograph pneumonia classification
**Target venue:** Medical Image Analysis (after IEEE Access rejection)
**Status:** frozen before any v2 experiment is run. Any deviation is logged in §10 with the date and reason.

---

## 1. Why v2 exists
| Source | Weakness of v1 | v2 response |
|---|---|---|
| Professor (1) | Balanced Kaggle derivative: 2,690 of 4,265 NORMAL files are augmented copies; 86.6 % of negative test images share a source with training; exact duplicates span splits | Rebuild from the **original Kermany release**. Splits are patient-grouped. The test set holds originals only. No augmented file is ever written to disk. Class imbalance is handled in the sampler (§3.3) |
| Professor (2) | LFR shows attention stays in the lung, not that it hits the disease | Add **lesion-level grounding** against radiologist boxes (RSNA, VinDr-CXR): Lesion-Focus Ratio, lesion-within-lung ratio, pointing game, and causal lesion occlusion (§5) |
| Professor (3) | One architecture (ViT-B/16) | **4 architectures × 3 arms × 3 seeds** on byte-identical inputs (§4) |
| Professor (4) | Null results read as "p > 0.05" | **Equivalence testing (TOST)** with a pre-set margin, Holm control, mixed-effects models, effect sizes with CIs, and one plain-language interpretation per test (§7) |
| Own review | RSNA images are a subset of NIH ChestX-ray14 (Shih et al., 2019), so the two are not independent cohorts | Add **VinDr-CXR** (Vietnam, radiologist boxes) as a truly independent cohort; remove NIH images that duplicate RSNA images; report RSNA/NIH as one source family with two labelling schemes |
| Own review | Chance level for LFR given in the wrong frame for the cropped arms | Chance computed **per arm, in that arm's own coordinate frame** |
| Own review | Unreported C − B p = 0.011 in V4 | Every pre-registered contrast is reported, whatever its outcome |

## 2. Research questions and hypotheses
- **H1 — Discrimination equivalence.** For each architecture, lung masking (arm C) is equivalent to the unmasked arm A, and to the crop-only arm B, in ROC-AUC within ±0.02 on every cohort.
- **H2 — Lung-level grounding.** Arm C places substantially more explanation energy inside the lungs than A and B, relative to the per-arm chance level. This holds for every architecture.
- **H3 — Lesion-level grounding (professor's question).** Within the lungs, explanation energy concentrates on radiologist-marked lesions above chance. *Pre-registered as two-sided and exploratory in direction:* masking guarantees lung-level focus by construction, not lesion-level focus.
- **H4 — Causal dependence.** (a) Occluding the lungs harms arm C more than an area-matched random occlusion, but not arm A. (b) Occluding the lesion box harms positives more than an area-matched random box inside the lung.
- **H5 — Architecture × preprocessing interaction.** The effect of masking on AUC and on grounding does not depend on architecture family (CNN vs. transformer). This is tested, not assumed.

## 3. Data
### 3.1 Internal (training) source — Kermany et al. 2018, Mendeley v2 (ChestXRay2017)
- 5,856 paediatric AP radiographs (official train 5,232, official test 624). Kaggle mirror: `paultimothymooney/chest-xray-pneumonia`; nested duplicate folders and `__MACOSX` are ignored.
- **Patient key:** `personN_bacteria_*` / `personN_virus_*` give `person_N_bacteria` / `person_N_virus` (Kermany numbers persons separately per pathogen; verified in V11 §2b); `IM-XXXX-*` and `NORMAL2-IM-XXXX-*` give XXXX (with prefix). Keys are merged across folders. If two different patients happen to share a key, they are grouped together. This errs on the safe side (over-grouping only).
- **De-duplication:** exact duplicates by MD5; near-duplicates by 16×16 dHash with Hamming ≤ 4, confirmed by visual inspection of a random 50-pair sample. Each duplicate group is reduced to one image and inherits the union of its patient keys.
- **Split (seed 42):**
  - **Test** = official Kermany test set (labels checked by a third expert), minus any image whose patient key or duplicate group also occurs in the official train set. Pneumonia keys are pathogen-specific (`person_N_bacteria` / `person_N_virus`), see §10.
  - **Validation** = 15 % of the official train patients, chosen by StratifiedGroupKFold.
  - **Training** = the remaining official-train patients.
- **Assertions (the notebook fails if any is violated):** zero patient-key overlap, zero MD5 overlap and zero dHash ≤ 4 overlap across the three splits; no file in any split carries an augmentation suffix. Split fingerprints are written in v1 style.
- The v1 balanced derivative is kept only as a **leakage case study** (Table: v1 vs v2 leakage).

### 3.2 External cohorts
| Cohort | Source | Positive | Negative | Labels | Boxes | n per class |
|---|---|---|---|---|---|---|
| RSNA | NIH-CXR14 subset (Shih 2019) | Lung Opacity | Normal (the "No Lung Opacity / Not Normal" class is excluded) | radiologists | yes | 1,000 |
| NIH | ChestX-ray14 | Pneumonia | No Finding | NLP from reports | no (984-image box subset not used) | ≤ 1,431, after removing RSNA duplicates by dHash |
| **VinDr-CXR** | Vietnam, 2 hospitals (Nguyen et al. 2022) | ≥ 2 of 3 radiologists mark Consolidation, Lung Opacity or Infiltration | all 3 radiologists mark No finding | radiologists | yes (union of agreeing readers) | ≤ 1,000 |
All external samples are drawn with seed 42. RSNA and NIH are analysed as **one source family**; independence claims rest on VinDr-CXR.

### 3.3 Class imbalance and augmentation (professor's point 1)
- No offline augmentation and no balancing by copying. Training uses `WeightedRandomSampler` with inverse class frequency; validation and test keep their natural prevalence.
- On-the-fly training augmentation is identical to v1 (horizontal flip, 8° rotation, 4 % translation, 0.95–1.05 scale, 0.15 brightness/contrast) and is applied equally to both classes.
- The reported metrics are prevalence-free (AUC) or reported with prevalence stated (sensitivity, specificity, PPV).

## 4. Preprocessing arms and models
### 4.1 Arms (unchanged from v1, now applied identically for all architectures)
- **A raw** — the whole image, resized to 224².
- **B ROI** — square crop around the lungs, no pixels removed.
- **C mask** — same crop box as B, with non-lung pixels replaced by the dataset mean grey.
- B/C crop: padded (5 %) bounding box of the dilated lung mask, cropped as a rectangle and resized to 224² (§10).

Masks come from the CheXmask U-Net (`ianpan/chest-x-ray-basic`, right and left lung only). Dilation is 2.5 % of the short side, Gaussian feathering uses k = 9, and ROI padding is 5 %. Masks and boxes are computed **once** (V12) and cached, so all 36 models see byte-identical inputs.

### 4.2 Architectures (timm, ImageNet weights)
`vit_base_patch16_224` · `densenet121` · `convnext_tiny` · `swin_tiny_patch4_window7_224`

### 4.3 Training protocol (identical across architectures)
- Full fine-tuning; AdamW (weight decay 0.05); cosine schedule with a 1-epoch warm-up; at most 20 epochs; early stopping on validation AUC (patience 5); batch size 32; mixed precision.
- **Learning rate:** for each architecture, one value is picked from {1e-5, 3e-5, 1e-4} using arm A, seed 42, validation AUC only. It is then frozen for all arms and seeds (12 sweep runs; this prevents tuning on the test set).
- **Seeds:** 42, 1337, 2024. They vary weight initialisation of the head, batch order and augmentation draws. The data split and external samples stay fixed.
- **Deviation from v1, disclosed:** v1 fine-tuned only the last 2 ViT blocks. v2 uses full fine-tuning for every architecture so that the comparison across architectures is fair.

## 5. Grounding metrics (professor's point 2)
All metrics are computed in each arm's own 224² frame, with lung masks and lesion boxes transformed into that frame (V12).
| Metric | Definition | Chance level |
|---|---|---|
| **LFR** (lung) | Σ S·L / Σ S over pixels with S > τ | mean lung fraction *in that arm's frame* |
| **LeFR** (lesion) | Σ S·B / Σ S over pixels with S > τ, B = union of radiologist boxes | box-area fraction of the frame |
| **LeFR-in-lung** (conditional) | Σ S·B·L / Σ S·L | (B∩L area) / (L area) |
| **Pointing game** | argmax S ∈ B (with 8-px tolerance) | (B area) / frame area, dilated by the same tolerance |
| **Box AUROC** | pixel-level AUROC of S for B vs ¬B | 0.5 |
- **Why LeFR-in-lung is the key lesion metric.** For arm C, LFR is close to 1 by construction, so plain LeFR mixes lung-level and lesion-level focus. The conditional ratio asks whether, *given* the lungs, attention goes to the lesion.
- **Positives only.** Lesion metrics are computed on positive images that have at least one box.
- **τ:** primary 0.20 (same as v1); sensitivity analysis at τ ∈ {0, 0.1, 0.4}, where 0 gives a threshold-free energy ratio.
- **Explanation methods.** Primary for the comparison across architectures is **occlusion sensitivity**: 14×14 grid, 16-px patches, mean-grey fill, Δp(pneumonia). It is model-agnostic and therefore fair across CNNs and transformers. Secondary: Grad-CAM for all four (with `reshape_transform` for ViT/Swin), and attention rollout for ViT only, for continuity with v1.
- **Grounding sample per cohort:** 200 positives + 100 negatives, seed 42; the internal cohort has no boxes and uses 300 images for LFR only. Maps are computed for seed-42 models (12 models). For all 36 models, Grad-CAM metrics are computed on 100 images per cohort to estimate seed variability.

## 6. Causal occlusion (H4)
- Fill is the dataset mean grey, the same as the masking pipeline. **net effect = ΔAUC(region) − ΔAUC(area-matched random)**, with the random region redrawn per image.
- **Regions:** lungs; non-lungs; 5 geometric bands (as in v1); **lesion box**.
- **Lesion-box design.** In positives, the radiologist box (union) is occluded. In negatives, a box is occluded whose size is drawn from the positive-box distribution and whose placement is random inside the lungs. This keeps the perturbation symmetric across classes, so the change in AUC is not created by perturbing only one class.
  - Lesion control: a box of the same size placed at random **inside the lung** but outside B.
  - Lesion net effect = ΔAUC(lesion) − ΔAUC(in-lung random).
- **Cohorts:** internal test (lungs, non-lungs and bands only), RSNA and VinDr (all regions). All 12 seed-42 models are run.
- **Uncertainty:** net-effect CIs come from a paired bootstrap over images (2,000 resamples). In v1, no test was run on net effects (review item M13).

## 7. Statistical analysis plan (professor's point 4)
### 7.1 Primary endpoint and unit
ROC-AUC per (architecture, arm, cohort). The primary estimate is the **seed ensemble** (the mean of the three seeds' probabilities). The seed spread is reported as the noise floor.

### 7.2 Equivalence (H1)
- Paired ΔAUC (C − A and C − B) with DeLong SE on the same images.
- **TOST:** equivalence is concluded if the 90 % CI of ΔAUC lies inside [−0.02, +0.02].
- **Why ±0.02:**
  - It is about 2× the largest seed SD observed on RSNA in v1 (0.009), so the margin is wider than run-to-run noise.
  - It is smaller than the AUC differences usually treated as practically meaningful in CXR classification (≥ 0.03–0.05).
  - Pre-specified sensitivity margins: ±0.01 and ±0.03.
  - NIH is pre-declared as secondary, because noisy labels cap its AUC.
- Each contrast falls into one of four conclusions: **equivalent** / **different** (two-sided DeLong p < 0.05 after Holm) / **both** (significant but inside the margin) / **inconclusive**.

### 7.3 Multiplicity
| Family | Tests | Correction |
|---|---|---|
| H1 equivalence | 4 architectures × 4 cohorts × 2 contrasts (C−A, C−B) = 32 TOST | Holm on the larger of the two one-sided p values |
| H2 | 4 architectures × 4 cohorts, C−A LFR (Wilcoxon) | Holm |
| H3 | 4 architectures × 3 arms × 2 box cohorts, LeFR-in-lung vs chance (one-sample Wilcoxon on LeFR-in-lung − chance) and pointing game vs chance (binomial) | Holm within each metric |
| H4 | lungs net and lesion net, per architecture × arm × cohort | Holm |
B − A and the geometric-band occlusions are **exploratory**: they are reported with unadjusted p values and labelled as such.

### 7.4 Interaction and models (H5)
- **AUC:** linear mixed model `AUC ~ arch * arm + (1 | seed)` per cohort, on the 36 seed-level AUCs (statsmodels MixedLM). The arch × arm interaction is tested by likelihood ratio.
- **Grounding:** a beta or logit-transformed mixed model `logit(metric) ~ arch * arm + (1 | image)` on image-level values. A rank-based sensitivity analysis is also run.

### 7.5 Effect sizes and interpretation
- Every result is reported as estimate + 95 % CI + a practical-meaning sentence, for example: "Masking changed AUC by +0.003 (90 % CI −0.006 to +0.011), inside the ±0.02 margin: masking neither helps nor harms discrimination by any practically relevant amount."
- Grounding effects are given as the ratio to chance and Cliff's δ.
- McNemar results at a fixed threshold are reported only as operating-point descriptors, never as evidence about discrimination (the saturation caveat from v1).

### 7.6 Exploratory: acquisition-geometry cue
Per cohort: AUC of the original pixel count and the image aspect ratio alone (no model). Per architecture × arm: (a) Spearman correlation of the model logit with log pixel count and aspect, within each class; (b) likelihood-ratio test of adding log pixel count + aspect to a logistic model of the label on the model logit. A model whose decision is carried by geometry gains little from geometry in (b) but correlates strongly in (a). Exploratory, unadjusted.

## 8. Deliverables and notebook map
| Notebook | Runs on | Est. GPU-h | Consumes | Produces |
|---|---|---|---|---|
| **V11** clean-split-v2 | Kaggle CPU | 0 | Kermany original | `split_v2.csv`, `audit_v2_summary.json`, fingerprints, audit figure |
| **V12** preprocess-cache-v2 | Kaggle GPU | ~1 | V11 + RSNA + NIH + VinDr | cached arm arrays, masks, boxes per frame, chance levels, mask QC |
| **V13** factorial-train-v2 (×4 sessions, one per architecture) | Kaggle GPU | ~20–25 | V12 | `preds_{arch}.npz`, curves, LR-sweep log, seed-42 checkpoints |
| **V14** grounding-occlusion-v2 | Kaggle GPU | ~4–6 | V12 + seed-42 checkpoints (+ all seeds for Grad-CAM) | `grounding_per_image.csv`, `occlusion_v2.csv`, aggregate maps, panels |
| Stats | local CPU | 0 | V13 + V14 outputs | tables, figures, interpretation report |
**Total ≈ 25–32 Kaggle GPU-hours** (quota 30 h/week). Each V13 session stays under the 12 h Kaggle limit, with resume-safe checkpointing.

## 9. Pre-declared limitations
- The training data are paediatric and single-centre, while the external cohorts are adult.
- The lesion boxes are coarse rectangles, so LeFR has a ceiling below 1 even for perfect localisation (reported).
- Occlusion produces out-of-distribution inputs (mitigated by the matched controls).
- Patient keys are inferred from filenames.
- The learning rate is tuned on arm A only; this is fair across arms but could favour arm A slightly (disclosed).

## 10. Deviation log
| Date | Change | Reason |
|---|---|---|
| 2026-10-16 (pre-data) | Occlusion-sensitivity and Grad-CAM maps explain the **predicted class**, not Δp(pneumonia) | Clarification: for negatives, the drop in the predicted class's probability is the meaningful attribution |
| 2026-10-16 (pre-data) | Training augmentation fills borders with mean grey (122), not black | Prevents a black-border cue that would differ between arms; v1 used the torchvision default |
| 2026-10-16 (pre-data) | V13 saves checkpoints for all seeds | Needed for the Grad-CAM seed-variability analysis (§5) |
| 2026-10-16 (pre-data) | NIH ∩ RSNA duplicates detected with dHash ≤ 10 bits on min-max-normalised images | Smoke test showed that DICOM vs PNG intensity stretching flips up to 5 bits, so a threshold of 4 misses true copies |
| 2026-10-17 (V11 run 1, before any model) | Pneumonia patient key changed from `person_N` to `person_N_{bacteria,virus}` | The pre-registered >30 % purge rule fired (320 / 624 official test images; test left with 70 / 390 pneumonia). All 170 collisions were `personN_virus` (test) vs `personN_bacteria` (train). Image evidence: repeats of the same person + pathogen are more similar than random pairs (median dHash 60 vs 66, width difference 7.7 % vs 19.4 %, MWU p ≈ 1e-74), whereas same-number / other-pathogen pairs are indistinguishable from random (66, 21.7 %, p = 0.90). Kermany numbers persons separately per pathogen. With the corrected key no test image is purged; test = 231 N / 387 P after removing within-split duplicates. Normal keys unchanged (numbering is disjoint across folders). The check is built into V11 (§2b) |
| 2026-10-17 (V11 run 2) | Canonical split = Kaggle run: train 1,141 N / 3,272 P, val 207 / 586, test 231 / 387; fingerprints `c443e812516ecf30` / `db515415832232a1` / `e852070e975c6646` | Test identical to the local replay; train/val differ only because StratifiedGroupKFold draws differently across scikit-learn versions. All overlap assertions pass |
| 2026-10-17 (after V12 run 1, before any training) | (a) B/C crop = padded lung bounding box as a **rectangle**, resized to 224² (anisotropic, as arm A), instead of a square capped at the short side. (b) Arm-A lung mask stored **dilated**, like B/C (one lung definition for LFR, chance levels and lung occlusion). V12 now asserts lung coverage of the crop ≥ 0.99. V12 re-run required | V12 run 1 showed the capped square cut > 10 % of the lung off in 7.8 % of Kermany images, **449 pneumonia vs 7 normal**: a class-asymmetric artefact in arms B/C (also present in v1). v1/V12 also stored arm-A masks undilated but B/C masks dilated, biasing arm-A LFR downward. Pre-specified sensitivity analysis added: grounding tables excluding implausible mask areas (V12 QC: 0.5 % internal, 3.6–4.1 % external) |
| 2026-10-18 (V12 run 2, before any training) | **Exploratory analysis added (§7.6):** acquisition-geometry cue. Report AUC of original pixel count and aspect ratio alone per cohort, and test whether each model's predicted probability is explained by them (logistic regression of the label on model logit ± log pixel count ± aspect; partial association of logit with geometry within class) | V12 run 2 manifest: on the internal test set, original pixel count alone gives AUC 0.941 (smaller = pneumonia) and aspect ratio 0.703 (wider = pneumonia); both ≈ 0.50 on RSNA/NIH. A candidate explanation for the extra-pulmonary signal left open in v1. V12 run 2 passed all checks (crop lung coverage min 1.000) |
