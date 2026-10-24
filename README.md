# Lung masking in chest radiograph pneumonia classification

Code, Kaggle notebook outputs and result tables for the study

> **Karateke, H., Karaduman, G.** *Lung masking in chest radiograph pneumonia classification: a leakage-controlled
> factorial study of discrimination, lesion-level grounding and shortcut reliance.* Manuscript submitted to *Medical Image Analysis*.

Department of Computer Engineering, Faculty of Engineering, Fırat University, Elazığ, Türkiye.

---

## Study in one paragraph

Four architectures (DenseNet-121, ConvNeXt-T, Swin-T, ViT-B/16) are trained on three inputs: **A** the raw radiograph,
**B** a lung region-of-interest crop and **C** the lung mask with the same crop. Each runs with three seeds, giving
**36 models**, on a patient- and duplicate-disjoint split of the original Kermany et al. (2018) paediatric release.
The models are evaluated on the internal test set and on RSNA, NIH ChestX-ray14 and VinDr-CXR. Discrimination is compared
with paired DeLong and equivalence (TOST, ±0.02) tests under Holm control. Grounding is measured with logit-margin
occlusion sensitivity against radiologist lesion boxes, with arm-specific chance levels, and causal reliance with matched
region occlusion.

| Finding | Evidence (seed ensembles / seed-42 maps) |
|---|---|
| Masking beats raw input | C − A significant in 12 / 16 architecture × cohort cells, never significantly negative |
| …mostly through cropping | C − B equivalent within ±0.02 in 8 / 16 cells; masking adds a gain mainly for ViT-B/16 |
| Masking moves evidence onto lesions | lesion focus in lung for arm C at 1.38–2.27 × chance in 7 / 8 cells; box AUROC C > A in 8 / 8 |
| Lesion pixels are used causally | lesion-box occlusion beats a same-size in-lung control in 23 / 24 cells |
| A size shortcut survives masking | original pixel count alone gives AUC 0.941 on the internal test set; model scores track it in every arm |

## Repository layout

```
notebooks/                         Kaggle notebooks, run in this order; each has the real run artefacts in output/
  01_clean_split/                  V11  patient-grouped, duplicate-free split of the Kermany release (CPU)
  02_preprocess_cache/             V12  lung segmentation, arms A/B/C, boxes, chance levels, mask QC (GPU)
  03_factorial_train/              V13  training; one Kaggle version per architecture (GPU)
     output/<architecture>/        per-seed predictions, runs, LR sweep, learning curves, log
  04_grounding_occlusion/          V14  Grad-CAM, attention rollout, seed variability (part B) (GPU)
  05_grounding_occlusion_logit/    V14b logit-margin occlusion maps, logit predictions, region occlusion (GPU)
  06_figure2_examples/             Figure 2 at print resolution from the notebook-02 cache (CPU)
analysis/
  stats_v2.py                      statistics library (DeLong, TOST, Holm, bootstrap, grounding, occlusion)
  run_final_statistics.py          rebuilds every table in results/tables from notebooks/*/output
  paper_figures.py                 rebuilds Figs. 3-7 from results/tables
results/
  tables/                          all result tables used in the paper (CSV)
  figures/                         paper figures (vector PDF + PNG) and graphical abstract
docs/
  study_design_preregistration.md  analysis plan frozen before training, with the dated deviation log
```

**Which notebook provides what.** Discrimination statistics, lesion/lung grounding (primary) and causal occlusion all use
notebook **05**, because its explanations and predictions use the logit margin. Notebook **04** provides the secondary
explanation methods (Grad-CAM, rollout) and the Grad-CAM seed-variability analysis. Its probability-based occlusion
output is kept for transparency: probability saturation left up to 29 % of its occlusion maps empty (Supplementary S2 of
the paper).

## Data

No image data are redistributed. Download the datasets from their providers:

| Cohort | Source | Kaggle input used |
|---|---|---|
| Kermany et al. 2018 (internal) | Mendeley Data, doi:10.17632/rscbjbr9sj.2 | `paultimothymooney/chest-xray-pneumonia` |
| RSNA Pneumonia Detection Challenge | RSNA / Kaggle | `rsna-pneumonia-detection-challenge` |
| NIH ChestX-ray14 | NIH Clinical Center | `nih-chest-xrays/data` |
| VinDr-CXR | VinBigData (PhysioNet / Kaggle) | `vinbigdata-chest-xray-abnormalities-detection` |

`notebooks/01_clean_split/output/split_v2.csv` gives the split of every Kermany file, with fingerprints
`c443e812516ecf30` (train), `db515415832232a1` (val) and `e852070e975c6646` (test).
`notebooks/02_preprocess_cache/output/manifest_v2.csv` lists every image used, with its label, crop and mask QC.

## Reproducing

### 1. Kaggle notebooks (GPU: T4 or P100)

| # | Notebook | Accelerator | Inputs | Time |
|---|---|---|---|---|
| 01 | `clean-split-v2` | CPU | Kermany | ~10 min |
| 02 | `preprocess-cache-v2` | GPU, Internet on | 01 output + RSNA + NIH + VinDr | ~1 h |
| 03 | `factorial-train-v2` | GPU | 02 output | 3–7 h per architecture |
| 04 | `grounding-occlusion-v2` | GPU | 02 output + the four 03 outputs | ~4 h |
| 05 | `grounding-occlusion-v14b` | GPU | same inputs as 04 | ~5.5 h |
| 06 | `figure2-arm-examples` | CPU | 02 output | ~2 min |

* In **03**, set `ARCH = "..."` to one of `densenet121`, `convnext_tiny`, `swin_tiny_patch4_window7_224`,
  `vit_base_patch16_224`, and save one Kaggle version per architecture. Turn each version's output into its own Kaggle
  dataset (Output → New Dataset). Kaggle attaches only the newest version of a notebook as an input.
* **04** and **05** find checkpoints by searching `**/ckpt/*.pth` and must print `36 checkpoints found`. Checkpoints
  (≈5 GB) are not stored in this repository.
* Software used on Kaggle: Python 3.13, PyTorch 2.11.0 (CUDA 12.8), timm 1.0.29. The lung segmentation model
  `ianpan/chest-x-ray-basic` is loaded from the Hugging Face Hub with `trust_remote_code=True`.

### 2. Statistics and figures (local, CPU)

```bash
pip install -r requirements.txt
python analysis/run_final_statistics.py --n-boot 2000   # about 45 min; writes results/tables
python analysis/paper_figures.py                        # writes results/figures (Figs. 3-7)
```

## Citation

See `CITATION.cff`. Please also cite the datasets above.

## License

Code: MIT (see `LICENSE`). Datasets remain under their providers' licences; result tables derived from them are provided
for research use.
