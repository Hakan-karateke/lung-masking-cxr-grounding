
# Paper figures for the v2 MedIA manuscript. Inputs: final tables (v2/final/tables), V14b aggregate maps, V12 manifest, seed-ensemble E.
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FD = "results/figures"
mpl.rcParams["pdf.fonttype"] = 42; mpl.rcParams["ps.fonttype"] = 42      # embed TrueType fonts (Elsevier artwork rules)
ARCHO = ["ViT", "Swin", "DenseNet", "ConvNeXt"]
ANAME = {"ViT": "ViT-B/16", "Swin": "Swin-T", "DenseNet": "DenseNet-121", "ConvNeXt": "ConvNeXt-T"}
AB = {"vit_base_patch16_224": "ViT", "swin_tiny_patch4_window7_224": "Swin", "densenet121": "DenseNet", "convnext_tiny": "ConvNeXt"}
AK = {v: k for k, v in AB.items()}
ARMC = {"raw": "#b75c35", "roi": "#c89926", "lung": "#269aa9"}          # same as the design schematic (Fig. 1)
ARML = {"raw": "A raw", "roi": "B lung-ROI crop", "lung": "C lung mask + crop"}
CC = {"lung-raw": "#1f3f66", "lung-roi": "#8a5aa8"}
CL = {"lung-raw": "C − A (mask vs raw)", "lung-roi": "C − B (mask vs crop)"}
COH = [("test", "Kermany internal test"), ("rsna", "RSNA"), ("nih", "NIH ChestX-ray14"), ("vindr", "VinDr-CXR")]

def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def apply_style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

def save(fig, name):
    fig.savefig(f"{FD}/{name}.png", dpi=300); fig.savefig(f"{FD}/{name}.pdf"); plt.close(fig)

def fig_forest(T):
    t = T.copy(); t["arch"] = t.arch.map(AB)
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (c, ttl) in zip(axs, COH):
        ax.axvspan(-0.02, 0.02, color="#ececec", zorder=0); ax.axvline(0, color="#555", lw=0.6, zorder=1)
        for j, con in enumerate(("lung-raw", "lung-roi")):
            for i, a in enumerate(ARCHO):
                r = t[(t.arch == a) & (t.cohort == c) & (t.contrast == con)].iloc[0]; y = i + (-0.17 if j == 0 else 0.17)
                sig = r.p_diff_holm < 0.05
                ax.plot([r.ci90_lo, r.ci90_hi], [y, y], color=CC[con], lw=1.3, zorder=2)
                ax.plot(r.delta, y, "o", ms=4.5, mfc=CC[con] if sig else "white", mec=CC[con], mew=1.1, zorder=3)
        ax.set_title(ttl, loc="left"); ax.set_xlim(-0.065, 0.085); ax.set_xticks([-0.05, 0, 0.05]); ax.set_xlabel("ΔAUC (90 % CI)"); ax.set_ylim(3.6, -0.6)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels([ANAME[a] for a in ARCHO])
    h = [Line2D([], [], color=CC[k], marker="o", ms=4.5, lw=1.3, label=CL[k]) for k in CC] + \
        [Line2D([], [], color="#777", marker="o", mfc="white", ms=4.5, lw=0, label="open = not significant (Holm)"),
         mpl.patches.Patch(color="#ececec", label="±0.02 equivalence margin")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.25, top=0.90, left=0.13, right=0.99); save(fig, "fig3_discrimination_forest")

def fig_lesion(Ls):
    s = Ls.copy(); s["arch"] = s.arch.map(AB)
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.2), sharex=True, gridspec_kw=dict(hspace=0.28, wspace=0.18))
    for col, (c, ttl) in enumerate([("rsna", "RSNA (n = 200 boxed positives)"), ("vindr", "VinDr-CXR (n = 200 boxed positives)")]):
        for row, (met, yl, ch) in enumerate([("ratio_to_chance", "Lesion focus in lung\n(× chance)", 1.0), ("box_auroc_median", "Box AUROC\n(median)", 0.5)]):
            ax = axs[row, col]; ax.axhline(ch, color="#555", lw=0.7, ls=(0, (3, 2)), zorder=1)
            for k, arm in enumerate(("raw", "roi", "lung")):
                for i, a in enumerate(ARCHO):
                    v = s[(s.arch == a) & (s.cohort == c) & (s.arm == arm)][met].iloc[0]
                    ax.plot([i + (k - 1) * 0.22] * 2, [ch, v], color=ARMC[arm], lw=1.0, zorder=2); ax.plot(i + (k - 1) * 0.22, v, "o", ms=5, color=ARMC[arm], zorder=3)
            if row == 0: ax.set_title(ttl, loc="left")
            if col == 0: ax.set_ylabel(yl)
            ax.set_xticks(range(4)); ax.set_xticklabels([ANAME[a] for a in ARCHO]); ax.set_xlim(-0.5, 3.5)
        axs[0, col].set_ylim(0, 2.5); axs[1, col].set_ylim(0.45, 0.8)
    for ax, ch in ((axs[0, 0], 1.0), (axs[1, 0], 0.5)):
        ax.text(-0.45, ch + (0.025 if ch == 1 else 0.004), "chance", ha="left", va="bottom", fontsize=6, color="#555")
    h = [Line2D([], [], color=ARMC[a], marker="o", ms=5, lw=0, label=ARML[a]) for a in ARMC]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    for ax, lt in zip(axs.ravel(), "abcd"): panel_letter(ax, lt, dx=-0.13, dy=1.08)
    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.11, right=0.99); save(fig, "fig5_lesion_grounding")

def fig_maps(am):
    fig, axs = plt.subplots(4, 6, figsize=(7.16, 5.25), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for i, a in enumerate(ARCHO):
        for j, (coh, arm) in enumerate([(c, x) for c in ("rsna", "vindr") for x in ("raw", "roi", "lung")]):
            mp = am[f"{AK[a]}|{arm}|{coh}|occlusion_logit|1"]; ax = axs[i, j]
            im = ax.imshow(mp / mp.max(), cmap="magma", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(ARMC[arm]); sp.set_linewidth(1.6)
            if i == 0: ax.set_title(f"{'RSNA' if coh == 'rsna' else 'VinDr-CXR'} · {ARML[arm].split(' ')[0]}", fontsize=7, color="black")
            if j == 0: ax.set_ylabel(ANAME[a], fontsize=7)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label("mean evidence (normalised)", fontsize=6); cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.95, bottom=0.02); save(fig, "fig4_mean_occlusion_maps")

def fig_occ(O):
    o = O.copy(); o["arch"] = o.arch.map(AB); o["d_nonlung"] = o.auc_intact - o.auc_region
    rowsk = [(a, arm) for a in ARCHO for arm in ("raw", "roi", "lung")]
    CN = {"test": "Kermany", "rsna": "RSNA", "nih": "NIH", "vindr": "VinDr"}
    panels = [("lungs", "net_effect", ["test", "rsna", "nih", "vindr"], "Lungs occluded\n(net of area-matched random)"),
              ("non_lungs", "d_nonlung", ["test", "rsna", "nih", "vindr"], "Outside lungs occluded\n(ΔAUC)"),
              ("lesion", "net_effect", ["rsna", "vindr"], "Lesion box occluded\n(net of same-size box)")]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.9), gridspec_kw=dict(width_ratios=[4, 4, 2.3], wspace=0.10))
    for ax, (reg, col, cohs, ttl) in zip(axs, panels):
        sel = lambda a, arm, c: o[(o.arch == a) & (o.arm == arm) & (o.cohort == (c if c != "test" else "internal")) & (o.region == reg)].iloc[0]
        M = np.array([[sel(a, arm, c)[col] for c in cohs] for a, arm in rowsk]); Pm = np.array([[sel(a, arm, c)["p_holm"] for c in cohs] for a, arm in rowsk])
        vm = np.nanmax(np.abs(M)); ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            star = "" if reg == "non_lungs" or not (Pm[i, j] < 0.05) else "*"
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.55 * vm else "black")
        ax.set_xticks(range(len(cohs))); ax.set_xticklabels([CN[c] for c in cohs]); ax.set_title(ttl, loc="left")
        ax.set_yticks(range(len(rowsk)))
        if ax is axs[0]:
            ax.set_yticklabels([f"{ANAME[a]} · {ARML[arm].split(' ')[0]}" for a, arm in rowsk])
            for tl, (a, arm) in zip(ax.get_yticklabels(), rowsk): tl.set_color(ARMC[arm] if arm != "roi" else "#9a7414")
        else: ax.set_yticklabels([])
        for yb in (2.5, 5.5, 8.5): ax.axhline(yb, color="white", lw=1.5)
        ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.subplots_adjust(left=0.16, right=0.99, top=0.86, bottom=0.06); save(fig, "fig6_causal_occlusion")

def fig_geometry(GEO, prev):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw=dict(width_ratios=[1, 1.7], wspace=0.35))
    ax = axs[0]; ax.bar(range(5), prev * 100, color="#7a7a7a", width=0.7)
    for i, v in enumerate(prev): ax.text(i, v * 100 + 2, f"{v * 100:.0f} %", ha="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_xticklabels(["1\nsmallest", "2", "3", "4", "5\nlargest"]); ax.set_xlabel("Original image size (quintile)")
    ax.set_ylabel("Pneumonia prevalence (%)"); ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_title("Image size alone predicts the label", loc="left")
    ax = axs[1]
    for i, a in enumerate(ARCHO):
        for k, arm in enumerate(("raw", "roi", "lung")):
            r = GEO[(GEO.arch == a) & (GEO.arm == arm)].iloc[0]; x = i + (k - 1) * 0.24
            ax.plot([x, x], [r.strat_auc_5, r.auc], color=ARMC[arm], lw=1.2); ax.plot(x, r.auc, "o", ms=4.5, color=ARMC[arm])
            ax.plot(x, r.strat_auc_5, "o", ms=4.5, mfc="white", mec=ARMC[arm], mew=1.1)
    ax.set_xticks(range(4)); ax.set_xticklabels([ANAME[a] for a in ARCHO]); ax.set_ylabel("Internal-test AUC"); ax.set_ylim(0.89, 1.0); ax.set_xlim(-0.5, 3.5)
    ax.set_title("AUC falls when size cannot help (all arms)", loc="left")
    h = [Line2D([], [], color=ARMC[a], marker="o", ms=4.5, lw=0, label=ARML[a]) for a in ARMC] + \
        [Line2D([], [], color="#555", marker="o", ms=4.5, lw=0, label="all images"), Line2D([], [], color="#555", marker="o", mfc="white", ms=4.5, lw=0, label="within size quintiles")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.6, 0.0))
    for a_, lt in zip(axs, "ab"): panel_letter(a_, lt, dx=-0.16, dy=1.04)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.30); save(fig, "fig7_acquisition_geometry")


def main():
    """Regenerate Figs. 3-7 from results/tables and the notebook outputs (run from the repository root)."""
    import glob, os
    from scipy.special import expit
    apply_style(); os.makedirs(FD, exist_ok=True)
    T = pd.read_csv("results/tables/h1_equivalence.csv"); Ls = pd.read_csv("results/tables/h3_lesion_grounding.csv")
    O = pd.read_csv("results/tables/h4_occlusion.csv"); GEO = pd.read_csv("results/tables/h6_geometry.csv")
    am = np.load(glob.glob("notebooks/05_grounding_occlusion_logit/output/maps/aggregate_maps_occlusion_logit.npz")[0])
    man = pd.read_csv(glob.glob("notebooks/02_preprocess_cache/output/**/manifest_v2.csv", recursive=True)[0])
    t = man[(man.cohort == "internal") & (man.split == "test")].copy(); t["q"] = pd.qcut(np.log(t.W0 * t.H0), 5, labels=False)
    prev = t.groupby("q").label.mean().values
    fig_forest(T); fig_maps(am); fig_lesion(Ls); fig_occ(O); fig_geometry(GEO, prev)
    print("figures written to", FD)


if __name__ == "__main__":
    main()
