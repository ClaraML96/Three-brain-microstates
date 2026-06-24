"""07_permutation_violin.py — effect-size companion to 07_permutation.py.

Reduces each subject to ONE ERD value per condition and draws split-violin
distributions with a paired effect size. Two independent statements per panel:

  • PARAMETRIC (violin body): per-subject ERD averaged in a FIXED a-priori
    ROI×time window (erd_cluster_stats.APRIORI_ROI / APRIORI_WINDOW). The paired
    t and Cohen's dz on these values are legitimate — the window is specified in
    advance, not read off the significant cluster — so they are NOT the
    double-dipped quantities the earlier draft showed.

  • NON-PARAMETRIC (bracket stars): the cluster-permutation p from the sibling
    test over the full 0–4 s window. This answers "is there a cluster anywhere in
    the ROI", a different question from the windowed parametric test, so the two
    legs are independent and may both be reported.

The earlier draft summarised each subject by the mean ERD *inside the significant
cluster* and computed t / d / p on it — circular by construction. That path (and
its silent whole-ROI fallback) is gone; see erd_cluster_stats for the rationale.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

import erd_cluster_stats as ec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (presentation only — compute config lives in erd_cluster_stats)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\violin_erd"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

COND_LABELS = {
    "T1P":  "Solo + FB",
    "T1Pn": "Solo − FB",
    "T3P":  "Trio + FB",
    "T3Pn": "Trio − FB",
}
COLORS = {
    "a": "#b91c1c",   # deep red  (condition A)
    "b": "#1d4ed8",   # deep blue (condition B)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load per-subject TFRs (shared with 07_permutation.py)
# ─────────────────────────────────────────────────────────────────────────────
group_tfr, info_ref = ec.load_group_tfrs()
adjacency = ec.build_adjacency(info_ref)
ch_names = info_ref.ch_names
print(f"\nChannel adjacency built for: {ch_names}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Per contrast × band: cluster test (bracket) + a-priori window (violin)
# ─────────────────────────────────────────────────────────────────────────────
times_full = ec.reference_times(group_tfr)
results = {}

for contrast_label, cond_a, cond_b, contrast_title in ec.CONTRASTS:
    if cond_a not in group_tfr or cond_b not in group_tfr:
        print(f"Skipping {contrast_label}: missing condition data.")
        continue

    for band_name, (fmin, fmax) in ec.FREQ_BANDS.items():
        print(f"\n{'='*60}")
        print(f"Contrast: {contrast_title}  |  {band_name} ({fmin}–{fmax} Hz)")

        # ── Non-parametric: cluster test over the full window (for the bracket) ──
        X, st_a, st_b, sids = ec.build_X(
            group_tfr, cond_a, cond_b, fmin, fmax, times_full,
        )
        n_subjects = X.shape[0]
        T_obs, clusters, cluster_p, H0 = ec.run_cluster(X, adjacency)
        _, sig_clusters = ec.significant_clusters(clusters, cluster_p, T_obs)

        # ── Parametric: per-subject ERD in the FIXED a-priori window (not the ──
        #    cluster). Same matched `sids`, so A and B align and N matches the
        #    bracket test.
        subj_a = ec.apriori_window_values(group_tfr, cond_a, band_name,
                                          info_ref, times_full, sids)
        subj_b = ec.apriori_window_values(group_tfr, cond_b, band_name,
                                          info_ref, times_full, sids)
        t_stat, p_ttest, cohens_d = ec.paired_parametric(subj_a, subj_b)

        roi_lbl = "+".join(ec.APRIORI_ROI[band_name])
        window_label = (f"a-priori window: {roi_lbl}, "
                        f"{ec.APRIORI_WINDOW[0]:g}–{ec.APRIORI_WINDOW[1]:g} s")

        results[(contrast_label, band_name)] = dict(
            sig_clusters=sig_clusters,
            subj_a=subj_a,
            subj_b=subj_b,
            cohens_d=cohens_d,
            t_stat=t_stat,
            p_ttest=p_ttest,
            window_label=window_label,
            p_floor=ec.perm_p_floor(n_subjects),
            n_subjects=n_subjects,
            contrast_title=contrast_title,
            cond_a=cond_a,
            cond_b=cond_b,
            fmin=fmin,
            fmax=fmax,
        )

        best_p = min((p for _, p in sig_clusters), default=None)
        print(f"  Sig clusters: {len(sig_clusters)}  "
              f"(min cluster p = {best_p if best_p is None else round(best_p, 4)})  |  "
              f"a-priori window: paired t({n_subjects-1}) = {t_stat:.3f}, "
              f"p = {p_ttest:.4f}, dz = {cohens_d:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Split-violin figure
# ─────────────────────────────────────────────────────────────────────────────
def half_violin(ax, data, pos, side, color, width=0.35):
    """One-sided KDE violin mirrored about `pos`, with IQR box + subject dots."""
    kde = gaussian_kde(data, bw_method="scott")
    y_grid = np.linspace(data.min() - 2, data.max() + 2, 300)
    density = kde(y_grid)
    density = density / density.max() * width

    edge = pos - density if side == "left" else pos + density
    ax.fill_betweenx(y_grid, pos, edge, color=color, alpha=0.30)
    ax.plot(edge, y_grid, color=color, lw=1.2)

    # IQR box + median
    q25, med, q75 = np.percentile(data, [25, 50, 75])
    box_x = pos - width * 0.35 if side == "left" else pos
    ax.add_patch(mpatches.Rectangle(
        (box_x, q25), width * 0.35, q75 - q25,
        facecolor=color, alpha=0.65, zorder=3,
    ))
    ax.hlines(med, box_x, box_x + width * 0.35,
              colors="white", linewidths=1.8, zorder=4)

    # Individual subject dots (jittered)
    rng = np.random.default_rng(seed=0)
    jitter = rng.uniform(-width * 0.12, width * 0.12, size=len(data))
    dot_x = pos + jitter + (-width * 0.17 if side == "left" else width * 0.17)
    ax.scatter(dot_x, data, color=color, s=18, alpha=0.55, zorder=5, linewidths=0)


def plot_violin_summary():
    bands = list(ec.FREQ_BANDS.keys())
    TITLE_FONT, LABEL_FONT, TICK_FONT, SUB_FONT = 13, 11, 10, 9

    for contrast_label, cond_a, cond_b, contrast_title in ec.CONTRASTS:
        if not any((contrast_label, b) in results for b in bands):
            print(f"Skipping plot for {contrast_label}: No data collected.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=False)
        fig.suptitle(
            f"{contrast_title}\nPer-subject ERD — a-priori window "
            f"(parametric); bracket = cluster permutation",
            fontsize=TITLE_FONT + 1, fontweight="bold", y=1.02,
        )

        for col, band_name in enumerate(bands):
            ax = axes[col]
            key = (contrast_label, band_name)
            if key not in results:
                ax.axis("off")
                continue

            r = results[key]
            subj_a, subj_b = r["subj_a"], r["subj_b"]

            half_violin(ax, subj_a, pos=0, side="left",  color=COLORS["a"])
            half_violin(ax, subj_b, pos=0, side="right", color=COLORS["b"])
            ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=1)

            # ── Bracket stars from the NON-PARAMETRIC cluster-perm p ──────────
            if r["sig_clusters"]:
                best_p = min(p for _, p in r["sig_clusters"])
                ymax = max(subj_a.max(), subj_b.max())
                y_br = ymax + abs(ymax) * 0.08 + 1
                ax.set_ylim(top=y_br + abs(y_br) * 0.25)
                h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                ax.plot([-0.25, -0.25, 0.25, 0.25],
                        [y_br, y_br + h, y_br + h, y_br], lw=0.8, color="0.35")
                stars = ("***" if best_p < 0.001 else "**" if best_p < 0.01
                         else "*" if best_p < 0.05 else "n.s.")
                ax.text(0, y_br + h * 1.2, stars, ha="center", va="bottom",
                        fontsize=LABEL_FONT, color="0.35")
                ptext = f"cluster perm. p = {best_p:.3f} (floor ≈ {r['p_floor']:.2g})"
            else:
                ptext = "cluster perm.: n.s."

            # ── Parametric annotation (legitimate — a-priori window) ──────────
            ax.text(0.97, 0.04,
                    f"dz = {r['cohens_d']:.2f}\n"
                    f"t({r['n_subjects']-1}) = {r['t_stat']:.2f}\n"
                    f"p = {r['p_ttest']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=SUB_FONT, color="0.4", style="italic")

            ax.set_xlim(-0.55, 0.55)
            ax.set_xticks([-0.17, 0.17])
            ax.set_xticklabels(
                [COND_LABELS.get(cond_a, cond_a), COND_LABELS.get(cond_b, cond_b)],
                fontsize=TICK_FONT, fontweight="bold",
            )
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel("Mean ERD (%)", fontsize=LABEL_FONT)
            ax.yaxis.set_tick_params(labelsize=TICK_FONT)
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.set_title(
                f"{band_name.capitalize()} ({r['fmin']}–{r['fmax']} Hz)\n"
                f"N = {r['n_subjects']}",
                fontsize=LABEL_FONT, fontweight="bold", pad=10,
            )
            ax.set_xlabel(f"{r['window_label']}\n{ptext}",
                          fontsize=SUB_FONT, color="0.5", labelpad=8)

        legend_handles = [
            mpatches.Patch(facecolor=COLORS["a"], alpha=0.65,
                           label=COND_LABELS.get(cond_a, "Cond A")),
            mpatches.Patch(facecolor=COLORS["b"], alpha=0.65,
                           label=COND_LABELS.get(cond_b, "Cond B")),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=TICK_FONT, framealpha=0.8, bbox_to_anchor=(0.5, -0.06))

        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"violin_erd_{contrast_label}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved separate plot: {fname}")
        plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Results table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VIOLIN SUMMARY — A-PRIORI-WINDOW EFFECT SIZES (parametric, non-circular)")
print("=" * 70)
print("\n" + ec.MULTIPLE_COMPARISONS)
for (contrast_label, band_name), r in results.items():
    print(f"\n{r['contrast_title']}  |  {band_name.capitalize()} "
          f"({r['fmin']}–{r['fmax']} Hz)  |  N = {r['n_subjects']}")
    print(f"  {r['window_label']}")
    print(f"  {COND_LABELS.get(r['cond_a'], r['cond_a'])}: "
          f"M = {r['subj_a'].mean():.2f}%,  SD = {r['subj_a'].std(ddof=1):.2f}%")
    print(f"  {COND_LABELS.get(r['cond_b'], r['cond_b'])}: "
          f"M = {r['subj_b'].mean():.2f}%,  SD = {r['subj_b'].std(ddof=1):.2f}%")
    print(f"  Parametric (a-priori window): paired t({r['n_subjects']-1}) = "
          f"{r['t_stat']:.3f}, p = {r['p_ttest']:.4f}, Cohen's dz = {r['cohens_d']:.3f}")
    if r["sig_clusters"]:
        best_p = min(p for _, p in r["sig_clusters"])
        print(f"  Non-parametric (cluster perm., full window): "
              f"{len(r['sig_clusters'])} sig cluster(s), min p = {best_p:.4f} "
              f"(floor ≈ {r['p_floor']:.2g})")
    else:
        print(f"  Non-parametric (cluster perm., full window): no significant cluster")

plot_violin_summary()
print("\nDone.")
