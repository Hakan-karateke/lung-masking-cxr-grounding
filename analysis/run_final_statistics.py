"""Reproduce every table of the paper from the Kaggle notebook outputs.

Usage (from the repository root):
    python analysis/run_final_statistics.py --n-boot 2000

Inputs are read from notebooks/*/output/ (see README). Tables are written to results/tables/.
All discrimination statistics use the logit predictions of notebook 05 (V14b); the seed ensemble is the mean
probability, computed in float64 from the logits (pre-registered estimator, without 0/1 saturation ties).
"""
import argparse, glob, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import expit
sys.path.insert(0, os.path.dirname(__file__))
import stats_v2 as SV
warnings.filterwarnings("ignore")

AB = {"vit_base_patch16_224": "ViT", "swin_tiny_patch4_window7_224": "Swin", "densenet121": "DenseNet", "convnext_tiny": "ConvNeXt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="notebooks")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d12, d13 = f"{a.root}/02_preprocess_cache/output", sorted(glob.glob(f"{a.root}/03_factorial_train/output/*"))
    d14, d14b = f"{a.root}/04_grounding_occlusion/output", f"{a.root}/05_grounding_occlusion_logit/output"
    save = lambda df, n: df.to_csv(os.path.join(a.out, n), index=False)
    man = pd.read_csv(glob.glob(f"{d12}/**/manifest_v2.csv", recursive=True)[0]); man["image_id"] = man.image_id.astype(str)

    # ---- H1 discrimination, H5 interaction (logit predictions, notebook 05)
    P = SV.load_preds([d14b], "logit"); P["p"] = expit(P.p.astype(np.float64))
    seed_auc, noise, T = SV.discrimination_tables(P, a.n_boot); MM = SV.mixed_models(seed_auc)
    for df, n in ((seed_auc, "seed_auc.csv"), (noise, "noise_floor.csv"), (T, "h1_equivalence.csv"), (MM, "h5_mixed_models.csv")): save(df, n)

    # ---- probability (V13) vs logit (V14b) per-seed AUC: effect of saturation ties
    P13 = SV.load_preds(d13, "prob")
    s13 = P13.groupby(["arch", "arm", "seed", "cohort"]).apply(lambda d: SV.auc(d.y, d.p)).rename("auc_prob")
    s14 = P.groupby(["arch", "arm", "seed", "cohort"]).apply(lambda d: SV.auc(d.y, d.p)).rename("auc_logit")
    C = pd.concat([s13, s14], axis=1).reset_index(); C["diff"] = C.auc_logit - C.auc_prob; save(C, "v13prob_vs_v14blogit_auc.csv")

    # ---- H2/H3 grounding: primary = logit occlusion (05); secondary = Grad-CAM / rollout (04)
    Gb = pd.read_csv(glob.glob(f"{d14b}/**/grounding_per_image_v14b.csv", recursive=True)[0])
    Gv = pd.read_csv(glob.glob(f"{d14}/**/grounding_per_image.csv", recursive=True)[0])
    Gv = Gv.drop_duplicates(["arch", "arm", "seed", "cohort", "image_id", "method", "tau"])   # Part A/B overlap at seed 42
    L, Ls = SV.grounding_tables(Gb, man, method="occlusion_logit"); save(L, "h2_lung_grounding.csv"); save(Ls, "h3_lesion_grounding.csv")
    Lx, Lsx = SV.grounding_tables(Gb, man, method="occlusion_logit", exclude_suspect_masks=True)
    save(Lx, "h2_sens_exclude_suspect.csv"); save(Lsx, "h3_sens_exclude_suspect.csv")
    for meth in ("gradcam", "rollout"):
        x, y = SV.grounding_tables(Gv, man, method=meth)
        save(x.assign(method=meth), f"h2_secondary_{meth}.csv"); save(y.assign(method=meth), f"h3_secondary_{meth}.csv")
    L14, Ls14 = SV.grounding_tables(Gv, man, method="occlusion")
    cmp = Ls14[["arch", "cohort", "arm", "n", "ratio_to_chance"]].merge(Ls[["arch", "cohort", "arm", "n", "ratio_to_chance"]],
                                                                        on=["arch", "cohort", "arm"], suffixes=("_v14", "_v14b"))
    save(cmp, "v14_vs_v14b_lesion.csv")
    gb = Gv[(Gv.method == "gradcam") & (Gv.tau == 0.2)]
    sv = gb.groupby(["arch", "arm", "cohort", "seed"]).lfr.median().groupby(["arch", "arm", "cohort"]).agg(["mean", "std"]).reset_index()
    save(sv, "partB_gradcam_seed_variability.csv")

    # paired lesion-level tests, arm C vs A and vs B (Wilcoxon, Holm within metric x mode)
    g = Gb[(Gb.tau == 0.2) & (Gb.label == 1) & Gb.cohort.isin(["rsna", "vindr"])].copy(); g["image_id"] = g.image_id.astype(str)
    rows = []
    for (arch, c), W in g.groupby(["arch", "cohort"]):
        for metric in ("lefr_in_lung", "box_auroc"):
            piv = W.pivot_table(index="image_id", columns="arm", values=metric)
            for con in (("lung", "raw"), ("lung", "roi")):
                for mode in (("complete", "nan_as_0") if metric == "lefr_in_lung" else ("complete",)):
                    pv = piv.fillna(0) if mode == "nan_as_0" else piv
                    d = (pv[con[0]] - pv[con[1]]).dropna()
                    rows.append(dict(arch=AB[arch], cohort=c, metric=metric, contrast="-".join(con), mode=mode, n=len(d), median_diff=d.median(),
                                     frac_pos=(d > 0).mean(), cliffs=SV.cliffs_delta(pv[con[0]].dropna(), pv[con[1]].dropna()), p=stats.wilcoxon(d).pvalue))
    H3p = pd.DataFrame(rows)
    for _, q in H3p.groupby(["metric", "mode"]): H3p.loc[q.index, "p_holm"] = SV.holm(q.p.values)
    save(H3p, "h3_paired_lung_vs_raw_roi.csv")

    # threshold sensitivity of lesion focus
    tr = []
    for tau in (0.0, 0.1, 0.2, 0.4):
        q = Gb[(Gb.tau == tau) & (Gb.label == 1) & Gb.cohort.isin(["rsna", "vindr"])].groupby(["arch", "cohort", "arm"]).apply(
            lambda d: d.lefr_in_lung.median() / d.lesion_in_lung_chance.median()).unstack()
        tr.append(dict(tau=tau, C_above_chance=int((q.lung > 1).sum()), A_above_chance=int((q.raw > 1).sum()), C_gt_A=int((q.lung > q.raw).sum()),
                       C_min=q.lung.min(), C_max=q.lung.max()))
    save(pd.DataFrame(tr), "h3_tau_sensitivity.csv")

    # ---- H4 causal region occlusion (logit, notebook 05)
    O = SV.occlusion_tables(d14b, a.n_boot, prefix="occ14b", key="logit"); save(O, "h4_occlusion.csv")

    # ---- exploratory acquisition geometry (internal test)
    E = SV.seed_ensemble(P); man["log_px"] = np.log(man.W0 * man.H0); gi = man[man.cohort == "internal"][["image_id", "log_px"]]
    rows = []
    for (arch, arm), d in E[E.cohort == "test"].groupby(["arch", "arm"]):
        d = d.merge(gi, on="image_id"); r = dict(arch=AB[arch], arm=arm, auc=SV.auc(d.y, d.p))
        for cl in (0, 1): r[f"rho_size_y{cl}"] = stats.spearmanr(d[d.y == cl].p, d[d.y == cl].log_px).statistic
        for nb in (5, 10):
            d["bin"] = pd.qcut(d.log_px, nb, labels=False); num = den = 0
            for _, q in d.groupby("bin"):
                n1, n0 = (q.y == 1).sum(), (q.y == 0).sum()
                if n1 and n0: num += SV.auc(q.y, q.p) * n1 * n0; den += n1 * n0
            r[f"strat_auc_{nb}"] = num / den
        rows.append(r)
    GEO = pd.DataFrame(rows); GEO["auc_drop_5"] = GEO.auc - GEO.strat_auc_5; save(GEO, "h6_geometry.csv")

    I = pd.concat([SV.interpret_discrimination(T), SV.interpret_grounding(L, Ls)], ignore_index=True); save(I, "interpretation_sentences.csv")
    print("tables written to", a.out)


if __name__ == "__main__":
    main()
