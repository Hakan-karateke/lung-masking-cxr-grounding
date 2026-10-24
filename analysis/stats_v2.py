"""Offline statistics for study design v2 (§7). npz files are our own V13/V14 outputs (ids are object arrays -> allow_pickle=True). numpy / scipy / statsmodels only; runs on CPU.

Inputs (directories may contain several Kaggle output folders):
  v13_dirs : folders with preds/{arch}__{arm}__s{seed}.npz
  v14_dir  : folder with grounding_per_image.csv and occlusion/occ__*.npz
  v12_dir  : folder with manifest_v2.csv and chance_levels_v2.csv
Outputs: tables_v2/*.csv, figures_v2/*.(pdf|png)
"""
import os, glob, warnings
import numpy as np, pandas as pd
from scipy import stats

ARCHS = ["vit_base_patch16_224", "swin_tiny_patch4_window7_224", "densenet121", "convnext_tiny"]
ARCH_LABEL = {"vit_base_patch16_224": "ViT-B/16", "swin_tiny_patch4_window7_224": "Swin-T",
              "densenet121": "DenseNet-121", "convnext_tiny": "ConvNeXt-T"}
ARMS = ["raw", "roi", "lung"]
ARM_LABEL = {"raw": "A raw", "roi": "B ROI crop", "lung": "C lung mask"}
COHORTS = ["test", "rsna", "nih", "vindr"]
GCOHORTS = ["internal", "rsna", "nih", "vindr"]          # naming used by V12/V14
COHORT_LABEL = {"internal": "Internal (paediatric)", "test": "Internal (paediatric)", "rsna": "RSNA", "nih": "NIH", "vindr": "VinDr-CXR"}
MARGIN, SENS_MARGINS = 0.02, (0.01, 0.03)
CONTRASTS = [("lung", "raw"), ("lung", "roi")]              # primary; (roi, raw) exploratory
N_BOOT, SEED = 2000, 42

# ----------------------------------------------------------------------------- AUC and DeLong
def auc(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = stats.rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def _placements(y, p):
    """DeLong structural components via midranks (Sun & Xu 2014)."""
    y = np.asarray(y).astype(bool); X, Y = p[y], p[~y]; m, n = len(X), len(Y)
    rz = stats.rankdata(np.concatenate([X, Y])); rx, ry = stats.rankdata(X), stats.rankdata(Y)
    v10 = (rz[:m] - rx) / n                    # positives
    v01 = 1.0 - (rz[m:] - ry) / m              # negatives
    return v10, v01

def delong_paired(y, p1, p2):
    """Returns (auc1, auc2, delta = auc1 - auc2, se_delta)."""
    a1, b1 = _placements(y, np.asarray(p1, float)); a2, b2 = _placements(y, np.asarray(p2, float))
    s10 = np.cov(np.vstack([a1, a2])); s01 = np.cov(np.vstack([b1, b2]))
    m, n = len(a1), len(b1)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    return a1.mean(), a2.mean(), a1.mean() - a2.mean(), float(np.sqrt(max(var, 0.0)))

def tost(delta, se, margin):
    """Two one-sided tests (normal approximation). Returns (p_tost, p_lower, p_upper)."""
    if se <= 0: return (0.0 if abs(delta) < margin else 1.0), np.nan, np.nan
    p_lo = 1 - stats.norm.cdf((delta + margin) / se)       # H0: delta <= -margin
    p_hi = stats.norm.cdf((delta - margin) / se)           # H0: delta >= +margin
    return max(p_lo, p_hi), p_lo, p_hi

def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    q = p[ok]; order = np.argsort(q); m = len(q); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * q[i])); adj[i] = run
    out[ok] = adj; return out

def strat_boot_idx(y, n_boot, rng):
    y = np.asarray(y).astype(bool); pos, neg = np.where(y)[0], np.where(~y)[0]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])

def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if not len(x) or not len(y): return np.nan
    return float((x[:, None] > y[None, :]).mean() - (x[:, None] < y[None, :]).mean())

# ----------------------------------------------------------------------------- loading
def load_preds(v13_dirs, score="prob"):
    """score='prob': V13 probabilities (preds/, key __prob); score='logit': V14b logit margins (preds_logit/, key __logit)."""
    folder, suf = ("preds", "__prob") if score == "prob" else ("preds_logit", "__logit")
    rows = []
    for d in v13_dirs:
        for f in glob.glob(os.path.join(d, "**", folder, "*.npz"), recursive=True):
            arch, arm, s = os.path.basename(f)[:-4].split("__")
            z = np.load(f, allow_pickle=True)
            for k in z.files:
                if k.endswith(suf):
                    c = k[:-len(suf)]
                    rows.append(pd.DataFrame(dict(arch=arch, arm=arm, seed=int(s[1:]), cohort=c,
                                                  image_id=z[f"{c}__id"].astype(str), y=z[f"{c}__y"].astype(int), p=z[k].astype(float))))
    P = pd.concat(rows, ignore_index=True).drop_duplicates(["arch", "arm", "seed", "cohort", "image_id"])
    return P[P.cohort != "val"]

def seed_ensemble(P):
    return P.groupby(["arch", "arm", "cohort", "image_id"], as_index=False).agg(y=("y", "first"), p=("p", "mean"), n_seeds=("seed", "nunique"))

# ----------------------------------------------------------------------------- H1 discrimination, H5 interaction
def discrimination_tables(P, n_boot=N_BOOT):
    seed_auc = (P.groupby(["arch", "arm", "seed", "cohort"])[["y", "p"]].apply(lambda d: auc(d.y, d.p))
                .rename("auc").reset_index())
    noise = seed_auc.groupby(["arch", "arm", "cohort"]).auc.agg(["mean", "std", "count"]).reset_index()
    E = seed_ensemble(P); rng = np.random.default_rng(SEED); rows = []
    for arch in sorted(E.arch.unique()):
        for c in COHORTS:
            sub = E[(E.arch == arch) & (E.cohort == c)]
            if sub.empty: continue
            W = sub.pivot(index="image_id", columns="arm", values="p").dropna()
            y = sub.drop_duplicates("image_id").set_index("image_id").loc[W.index, "y"].values
            for a, b in CONTRASTS + [("roi", "raw")]:
                if a not in W or b not in W or len(np.unique(y)) < 2: continue
                A1, A2, dlt, se = delong_paired(y, W[a].values, W[b].values)
                p_diff = 2 * (1 - stats.norm.cdf(abs(dlt / se))) if se > 0 else float(dlt == 0)
                bs = np.array([auc(y[i], W[a].values[i]) - auc(y[i], W[b].values[i]) for i in strat_boot_idx(y, n_boot, rng)])
                r = dict(arch=arch, cohort=c, contrast=f"{a}-{b}", primary=(a, b) in CONTRASTS, n=len(y), n_pos=int(y.sum()),
                         auc_a=A1, auc_b=A2, delta=dlt, se=se, ci90_lo=dlt - 1.645 * se, ci90_hi=dlt + 1.645 * se,
                         ci95_lo=dlt - 1.96 * se, ci95_hi=dlt + 1.96 * se,
                         boot95_lo=np.percentile(bs, 2.5), boot95_hi=np.percentile(bs, 97.5), boot_sd=bs.std(ddof=1), p_diff=p_diff)
                for mg in (MARGIN,) + SENS_MARGINS:
                    r[f"p_tost_{mg:g}"] = tost(dlt, se, mg)[0]
                sa = seed_auc[(seed_auc.arch == arch) & (seed_auc.cohort == c)].pivot(index="seed", columns="arm", values="auc")
                if a in sa and b in sa:
                    dd = (sa[a] - sa[b]).dropna(); r["seed_delta_mean"] = dd.mean()
                    r["seed_delta_sd"] = dd.std(ddof=1) if len(dd) > 1 else np.nan
                rows.append(r)
    T = pd.DataFrame(rows)
    if T.empty: return seed_auc, noise, T
    fam = T.primary.values
    T.loc[fam, "p_tost_holm"] = holm(T.loc[fam, f"p_tost_{MARGIN:g}"].values)
    T.loc[fam, "p_diff_holm"] = holm(T.loc[fam, "p_diff"].values)
    def verdict(r):
        if not r.primary: return "exploratory"
        eq, df = r.p_tost_holm < 0.05, r.p_diff_holm < 0.05
        return "equivalent and different" if eq and df else "equivalent" if eq else "different" if df else "inconclusive"
    T["conclusion"] = T.apply(verdict, axis=1)
    return seed_auc, noise, T

def mixed_models(seed_auc):
    import statsmodels.formula.api as smf
    out = []
    for c in COHORTS:
        d = seed_auc[seed_auc.cohort == c].dropna()
        if d.arch.nunique() < 2 or d.arm.nunique() < 2 or d.seed.nunique() < 2: continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                full = smf.mixedlm("auc ~ C(arch) * C(arm)", d, groups=d["seed"]).fit(reml=False)
                add = smf.mixedlm("auc ~ C(arch) + C(arm)", d, groups=d["seed"]).fit(reml=False)
                kind, llf_f, llf_a, sv = "MixedLM (1|seed)", full.llf, add.llf, float(full.cov_re.iloc[0, 0])
            except Exception:
                full = smf.ols("auc ~ C(arch) * C(arm) + C(seed)", d).fit(); add = smf.ols("auc ~ C(arch) + C(arm) + C(seed)", d).fit()
                kind, llf_f, llf_a, sv = "OLS + seed fixed effect", full.llf, add.llf, np.nan
        df_int = (d.arch.nunique() - 1) * (d.arm.nunique() - 1); lr = 2 * (llf_f - llf_a)
        out.append(dict(cohort=c, model=kind, n=len(d), lr_stat=lr, df=df_int,
                        p_interaction=1 - stats.chi2.cdf(max(lr, 0), df_int), seed_variance=sv))
    return pd.DataFrame(out)

# ----------------------------------------------------------------------------- H2 / H3 grounding
def grounding_tables(G, manifest, method="occlusion", tau=0.2, exclude_suspect_masks=False):
    g = G[(G.method == method) & (G.tau == tau) & (G.seed == 42)].copy(); g["image_id"] = g.image_id.astype(str)
    if exclude_suspect_masks:                                  # sensitivity analysis: drop implausible mask areas (V12 QC)
        bad = set(manifest.loc[manifest.mask_suspect.astype(bool), "image_id"].astype(str)); g = g[~g.image_id.isin(bad)]
    m = manifest.copy(); m["image_id"] = m.image_id.astype(str)
    for col in ("lefr_in_lung", "point_hit", "point_chance", "box_auroc"):
        if col not in g: g[col] = np.nan
    lung_rows, les_rows = [], []
    for arch in sorted(g.arch.unique()):
        for c in g.cohort.unique():
            mc = m[m.cohort == c].drop_duplicates("image_id").set_index("image_id")
            W = g[(g.arch == arch) & (g.cohort == c)]
            piv = W.pivot_table(index="image_id", columns="arm", values="lfr")
            for arm in ARMS:
                if arm not in piv: continue
                ch = (W[W.arm == arm].set_index("image_id").reindex(piv.index)["lung_chance"].values if "lung_chance" in W
                      else mc.reindex(piv.index)[f"lung_frac_{arm}"].values)
                lung_rows.append(dict(arch=arch, cohort=c, arm=arm, n=int(piv[arm].notna().sum()), lfr_median=piv[arm].median(),
                                      chance=np.nanmean(ch), ratio_to_chance=piv[arm].median() / np.nanmean(ch)))
            if {"lung", "raw"} <= set(piv.columns):
                d = (piv["lung"] - piv["raw"]).dropna()
                p = stats.wilcoxon(d).pvalue if len(d) > 1 and (d != 0).any() else np.nan
                lung_rows.append(dict(arch=arch, cohort=c, arm="lung-raw", n=len(d), lfr_median=d.median(), p_wilcoxon=p,
                                      cliffs_delta=cliffs_delta(piv["lung"], piv["raw"]), frac_C_higher=float((d > 0).mean())))
            P = W[(W.label == 1) & W.lefr_in_lung.notna()]
            for arm in ARMS:
                q = P[P.arm == arm]
                if q.empty: continue
                chn = (q["lesion_in_lung_chance"].values if "lesion_in_lung_chance" in q and q["lesion_in_lung_chance"].notna().all()
                       else mc.reindex(q.image_id)[f"box_in_lung_frac_{arm}"].values)
                diff = q.lefr_in_lung.values - chn
                pw = stats.wilcoxon(diff).pvalue if len(diff) > 1 and (diff != 0).any() else np.nan
                hit, pch = q.point_hit.values, q.point_chance.values; dh = hit - pch
                ph = stats.wilcoxon(dh).pvalue if len(dh) > 1 and (dh != 0).any() else np.nan
                les_rows.append(dict(arch=arch, cohort=c, arm=arm, n=len(q), lefr_in_lung_median=float(np.median(q.lefr_in_lung)),
                                     chance_median=float(np.median(chn)), ratio_to_chance=float(np.median(q.lefr_in_lung) / np.median(chn)),
                                     p_vs_chance=pw, point_hit_rate=float(np.mean(hit)), point_chance_mean=float(np.mean(pch)),
                                     p_point_vs_chance=ph, box_auroc_median=float(np.nanmedian(q.box_auroc))))
    L, Ls = pd.DataFrame(lung_rows), pd.DataFrame(les_rows)
    if "p_wilcoxon" in L:
        s = L.p_wilcoxon.notna(); L.loc[s, "p_holm"] = holm(L.loc[s, "p_wilcoxon"].values)
    if len(Ls):
        Ls["p_vs_chance_holm"] = holm(Ls.p_vs_chance.values); Ls["p_point_holm"] = holm(Ls.p_point_vs_chance.values)
    return L, Ls

# ----------------------------------------------------------------------------- H4 occlusion
CTRL_OF = {"lungs": "rand_lungs", "corners": "rand_20", "center": "rand_20", "upper": "rand_20",
           "border": "rand_30", "subdiaphragm": "rand_30", "lesion": "lesion_ctrl"}
def occlusion_tables(v14_dir, n_boot=N_BOOT, prefix="occ", key="p"):
    """V14: prefix='occ', key='p' (probabilities). V14b: prefix='occ14b', key='logit' (logit margins)."""
    rng = np.random.default_rng(SEED); rows = []
    for f in sorted(glob.glob(os.path.join(v14_dir, "**", "occlusion", f"{prefix}__*.npz"), recursive=True)):
        _, arch, arm, _ = os.path.basename(f)[:-4].split("__"); z = np.load(f, allow_pickle=True)
        ci = {str(c): i for i, c in enumerate(z["conditions"])}
        for coh in [k[:-len(key) - 2] for k in z.files if k.endswith(f"__{key}")]:
            y, Pm = z[f"{coh}__y"].astype(int), z[f"{coh}__{key}"]
            if len(np.unique(y)) < 2: continue
            ok = z[f"{coh}__lesion_ok"] if f"{coh}__lesion_ok" in z.files else np.ones(len(y), bool)
            base = auc(y, Pm[:, 0])
            for reg, ctrl in CTRL_OF.items():
                sel = ok if reg == "lesion" else np.ones(len(y), bool)
                yy, pr, pc = y[sel], Pm[sel, ci[reg]], Pm[sel, ci[ctrl]]
                if len(np.unique(yy)) < 2: continue
                net = auc(yy, pc) - auc(yy, pr)                      # = ΔAUC(region) − ΔAUC(control)
                bs = np.array([auc(yy[i], pc[i]) - auc(yy[i], pr[i]) for i in strat_boot_idx(yy, n_boot, rng)])
                p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
                rows.append(dict(arch=arch, arm=arm, cohort=coh, region=reg, control=ctrl, n=int(sel.sum()), auc_intact=base,
                                 auc_region=auc(yy, pr), auc_control=auc(yy, pc), net_effect=net,
                                 ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), p_boot=max(p, 1 / n_boot),
                                 primary=reg in ("lungs", "lesion")))
            rows.append(dict(arch=arch, arm=arm, cohort=coh, region="non_lungs", control="", n=len(y), auc_intact=base,
                             auc_region=auc(y, Pm[:, ci["non_lungs"]]), net_effect=np.nan, primary=False))
    O = pd.DataFrame(rows)
    if len(O):
        for reg in ("lungs", "lesion"):
            s = (O.region == reg) & O.p_boot.notna(); O.loc[s, "p_holm"] = holm(O.loc[s, "p_boot"].values)
    return O

# ----------------------------------------------------------------------------- plain-language interpretation
def _f(x, d=3): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"
def interpret_discrimination(T):
    out = []
    for r in T[T.primary].itertuples():
        a, b = r.contrast.split("-")
        what = f"{ARCH_LABEL.get(r.arch, r.arch)}, {COHORT_LABEL.get(r.cohort, r.cohort)}: {ARM_LABEL[a]} vs {ARM_LABEL[b]}"
        ci = f"ΔAUC {_f(r.delta)} (90 % CI {_f(r.ci90_lo)} to {_f(r.ci90_hi)})"
        if r.conclusion == "equivalent":
            s = f"{ci} lies inside ±{MARGIN}: masking neither helps nor harms discrimination by a practically relevant amount (TOST, Holm p = {r.p_tost_holm:.3g})."
        elif r.conclusion == "different":
            s = f"{ci} differs from zero (Holm p = {r.p_diff_holm:.3g}) and is not contained in ±{MARGIN}: a practically relevant difference."
        elif r.conclusion == "equivalent and different":
            s = f"{ci} is statistically non-zero (Holm p = {r.p_diff_holm:.3g}) but inside ±{MARGIN}: real but too small to matter."
        else:
            s = f"{ci} is neither shown equivalent nor different: the data cannot rule out a difference of ±{MARGIN} (TOST, Holm p = {r.p_tost_holm:.3g})."
        out.append(dict(family="H1", test=what, conclusion=r.conclusion, sentence=s))
    return pd.DataFrame(out)

def interpret_grounding(L, Ls):
    out = []
    if "frac_C_higher" in L:
        for r in L[L.arm == "lung-raw"].itertuples():
            out.append(dict(family="H2", test=f"{ARCH_LABEL.get(r.arch, r.arch)}, {COHORT_LABEL.get(r.cohort, r.cohort)}: lung focus C vs A",
                            sentence=f"Median per-image difference {_f(r.lfr_median)}; C higher in {100 * r.frac_C_higher:.0f} % of images; "
                                     f"Cliff's δ {_f(r.cliffs_delta, 2)} (Holm p = {getattr(r, 'p_holm', np.nan):.3g})."))
    for r in Ls.itertuples():
        rel = "above" if r.ratio_to_chance > 1 else "below"
        out.append(dict(family="H3", test=f"{ARCH_LABEL.get(r.arch, r.arch)}, {COHORT_LABEL.get(r.cohort, r.cohort)}, {ARM_LABEL[r.arm]}: lesion focus within lung",
                        sentence=f"Median {r.lefr_in_lung_median:.3f} vs chance {r.chance_median:.3f} (×{r.ratio_to_chance:.1f}, {rel} chance; Holm p = {r.p_vs_chance_holm:.3g}); "
                                 f"pointing-game hit {100 * r.point_hit_rate:.0f} % vs {100 * r.point_chance_mean:.0f} % expected by chance."))
    return pd.DataFrame(out)

# ----------------------------------------------------------------------------- figures
ARM_COL = {"raw": "#B04A1F", "roi": "#C28E0E", "lung": "#0E8FA0"}
def fig_forest(T, path, margin=MARGIN):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = T[T.primary]
    if t.empty: return
    cohorts = [c for c in COHORTS if c in set(t.cohort)]; archs = [a for a in ARCHS if a in set(t.arch)]
    fig, axs = plt.subplots(1, len(cohorts), figsize=(7.16, 0.7 + 0.36 * len(archs) * 2), sharey=True, layout="constrained", squeeze=False)
    cols = {"lung-raw": "#0E8FA0", "lung-roi": "#6A4C93"}
    for ax, c in zip(axs[0], cohorts):
        ax.axvspan(-margin, margin, color="#E6E6E6", zorder=0, lw=0); ax.axvline(0, color="#5A5A5A", lw=0.6)
        for i, a in enumerate(archs):
            for j, con in enumerate(("lung-raw", "lung-roi")):
                r = t[(t.arch == a) & (t.cohort == c) & (t.contrast == con)]
                if r.empty: continue
                r = r.iloc[0]; yv = i * 2 + j * 0.8
                ax.plot([r.ci90_lo, r.ci90_hi], [yv, yv], color=cols[con], lw=1.1)
                ax.plot(r.delta, yv, "o", ms=3.4, color=cols[con], mfc=cols[con] if str(r.conclusion).startswith("equivalent") else "white", zorder=3)
        ax.set_title(COHORT_LABEL[c], loc="left"); ax.set_xlabel("ΔAUC (90 % CI)"); ax.margins(x=0.1)
    axs[0][0].set_yticks([i * 2 + 0.4 for i in range(len(archs))]); axs[0][0].set_yticklabels([ARCH_LABEL[a] for a in archs])
    axs[0][0].invert_yaxis()
    axs[0][-1].legend(handles=[Line2D([], [], color=cols["lung-raw"], marker="o", ms=3.4, label="C − A"),
                               Line2D([], [], color=cols["lung-roi"], marker="o", ms=3.4, label="C − B"),
                               Line2D([], [], color="#777777", marker="o", ms=3.4, mfc="white", lw=0, label="not shown equivalent")],
                      frameon=False, loc="lower right", fontsize=6)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=600); plt.close(fig)

def fig_grounding(L, Ls, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if L.empty: return
    cohorts = [c for c in GCOHORTS if c in set(L.cohort)]; archs = [a for a in ARCHS if a in set(L.arch)]
    fig, axs = plt.subplots(2, len(cohorts), figsize=(7.16, 3.6), sharey="row", layout="constrained", squeeze=False)
    for j, c in enumerate(cohorts):
        for k, (tab, ttl) in enumerate(((L, "Lung focus\n(× chance)"), (Ls, "Lesion focus in lung\n(× chance)"))):
            ax = axs[k, j]; ax.axhline(1, color="#5A5A5A", lw=0.6, ls=":")
            if tab.empty or c not in set(tab.cohort):
                ax.text(0.5, 0.5, "no radiologist boxes", ha="center", va="center", transform=ax.transAxes, fontsize=6, color="#777777")
                ax.set_xticks([]); continue
            for i, a in enumerate(archs):
                for m_, arm in enumerate(ARMS):
                    r = tab[(tab.arch == a) & (tab.cohort == c) & (tab.arm == arm)]
                    if len(r) and np.isfinite(r.ratio_to_chance.iloc[0]) and r.ratio_to_chance.iloc[0] > 0:
                        ax.plot(i + (m_ - 1) * 0.22, r.ratio_to_chance.iloc[0], "o", ms=3.2, color=ARM_COL[arm])
            ax.set_xticks(range(len(archs))); ax.set_xticklabels([ARCH_LABEL[a] for a in archs], fontsize=5.5, rotation=30)
            ax.set_yscale("log")
            if j == 0: ax.set_ylabel(ttl)
            if k == 0: ax.set_title(COHORT_LABEL[c], loc="left")
    axs[0][-1].legend(handles=[Line2D([], [], color=ARM_COL[a], marker="o", lw=0, ms=3.2, label=ARM_LABEL[a]) for a in ARMS], frameon=False, fontsize=6)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=600); plt.close(fig)

def fig_occlusion(O, path):
    import matplotlib.pyplot as plt
    o = O[O.region.isin(["lungs", "lesion"]) & O.net_effect.notna()] if len(O) else O
    if o is None or o.empty: return
    cohorts = [c for c in GCOHORTS if c in set(o.cohort)]; archs = [a for a in ARCHS if a in set(o.arch)]
    fig, axs = plt.subplots(1, len(cohorts), figsize=(7.16, 2.8), sharey=True, layout="constrained", squeeze=False)
    vmax = float(np.nanmax(np.abs(o.net_effect))) or 1.0
    rows = [(a, arm) for a in archs for arm in ARMS]
    for ax, c in zip(axs[0], cohorts):
        M = np.full((len(rows), 2), np.nan)
        for i, (a, arm) in enumerate(rows):
            for j, reg in enumerate(("lungs", "lesion")):
                r = o[(o.arch == a) & (o.arm == arm) & (o.cohort == c) & (o.region == reg)]
                if len(r): M[i, j] = r.net_effect.iloc[0]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            if not np.isnan(v): ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["lungs", "lesion box"]); ax.set_title(COHORT_LABEL[c], loc="left")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{ARCH_LABEL[a]} · {ARM_LABEL[arm][0]}" for a, arm in rows], fontsize=5.5)
    fig.colorbar(im, ax=axs[0][-1], label="net effect on AUC\n(region − matched control)", shrink=0.8)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=600); plt.close(fig)

# ----------------------------------------------------------------------------- driver
def run_all(v13_dirs, v14_dir, v12_dir, out, n_boot=N_BOOT):
    T_, F_ = os.path.join(out, "tables_v2"), os.path.join(out, "figures_v2")
    os.makedirs(T_, exist_ok=True); os.makedirs(F_, exist_ok=True); res = {}
    P = load_preds(v13_dirs); res["n_pred_rows"] = len(P)
    seed_auc, noise, T = discrimination_tables(P, n_boot); MM = mixed_models(seed_auc)
    for n, d in (("seed_auc", seed_auc), ("noise_floor", noise), ("h1_equivalence", T), ("h5_mixed_models", MM)):
        d.to_csv(os.path.join(T_, f"{n}.csv"), index=False); res[n] = d
    fig_forest(T, os.path.join(F_, "fig_h1_forest"))
    sentences = [interpret_discrimination(T)] if len(T) else []
    gp = glob.glob(os.path.join(v14_dir, "**", "grounding_per_image.csv"), recursive=True) if v14_dir else []
    mp = glob.glob(os.path.join(v12_dir, "**", "manifest_v2.csv"), recursive=True) if v12_dir else []
    if gp and mp:
        L, Ls = grounding_tables(pd.read_csv(gp[0]), pd.read_csv(mp[0]))
        L.to_csv(os.path.join(T_, "h2_lung_grounding.csv"), index=False); Ls.to_csv(os.path.join(T_, "h3_lesion_grounding.csv"), index=False)
        fig_grounding(L, Ls, os.path.join(F_, "fig_h2h3_grounding")); sentences.append(interpret_grounding(L, Ls))
        res["h2"], res["h3"] = L, Ls
    if v14_dir:
        O = occlusion_tables(v14_dir, n_boot); O.to_csv(os.path.join(T_, "h4_occlusion.csv"), index=False)
        fig_occlusion(O, os.path.join(F_, "fig_h4_occlusion")); res["h4"] = O
    I = pd.concat(sentences, ignore_index=True) if sentences else pd.DataFrame()
    I.to_csv(os.path.join(T_, "interpretation_sentences.csv"), index=False); res["interp"] = I
    return res
