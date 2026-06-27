"""
2×2 PLV overview — four-contrast boxplot summary (descriptive).

Builds ONE figure (two panels: alpha, beta) with four boxplots — one per
simple-effect contrast of the 2×2 design — so the contrasts can be eyeballed
side by side. Pure post-processing: reads the per-triad ΔPLV arrays and cluster
CSVs already written by `plv_condition_contrast_v4_2x2.py`. No EEG, no recompute.

WHAT EACH BOX IS (decisions taken with the user, 2026-06-27):
  • Unit = ONE point per triad = that triad's ΔPLV averaged over all channel
    pairs. So the box's spread is BETWEEN-TRIAD variability — the same unit the
    permutation_cluster_1samp_test operates on. (~30 points/box.)
  • These are DIFFERENCES, not levels. Each box is centred near zero; the y-axis
    is ΔPLV, NOT PLV. The plot shows effect sizes, not coupling magnitude.
  • Colour = the band's MIN cluster p (blue = sig at p<.05, red = n.s.), one
    colour per box. This is coarser than a per-element surrogate test.

HONESTY FLAGS BAKED INTO THE FIGURE (do not strip):
  • Boxes sit almost on the zero line: the effects are tiny, spatially broad,
    triad-consistent differences (Finding 1), not large shifts. The figure should
    LOOK near-flat — that is the honest picture.

Source: raw/output/plv_2x2_v4/<contrast>/{diff_alpha.npy,diff_beta.npy,cluster_results.csv}
Output: raw/output/plv_2x2_v4/overview_boxplots.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── paths ────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
VAULT    = os.path.abspath(os.path.join(HERE, "..", ".."))
# RUN_DIR  = os.path.join(VAULT, "raw", "output", "plv_2x2_v4")
# OUT_PNG  = os.path.join(RUN_DIR, "overview_boxplots.png")
RUN_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\plv\plv_2x2_v4"
OUT_PNG  = os.path.join(RUN_DIR, "overview_boxplots.png")

BANDS = ["alpha", "beta"]
SIG_ALPHA = 0.05

# Order + labels follow the user's request:
#   T1P−T1Pn, T3P−T3Pn, T3P−T1P, T3Pn−T1Pn
CONTRASTS = [
    {"dir": "feedback_in_solo",  "math": "T1P − T1Pn",  "name": "Feedback,\nSolo",      "feedback": True},
    {"dir": "feedback_in_triad", "math": "T3P − T3Pn",  "name": "Feedback,\nTrio",      "feedback": True},
    {"dir": "grpsize_in_fb",     "math": "T3P − T1P",   "name": "Trio vs Solo,\nfeedback","feedback": False},
    {"dir": "grpsize_in_nofb",   "math": "T3Pn − T1Pn", "name": "Trio vs Solo,\nno-fb",   "feedback": False},
]

C_SIG   = "#4C72B0"   # blue  — significant cluster (p<.05)
C_NS    = "#C44E52"   # red   — not significant
C_EDGE  = "#2b2b2b"


def min_cluster_p(contrast_dir: str, band: str) -> float:
    """Smallest cluster p for this contrast within this band."""
    csv = os.path.join(RUN_DIR, contrast_dir, "cluster_results.csv")
    df = pd.read_csv(csv)
    sub = df[df["bands"] == band]          # block-diagonal → each cluster is one band
    return float(sub["p_value"].min()) if len(sub) else 1.0


def per_triad_mean(contrast_dir: str, band: str) -> np.ndarray:
    """One ΔPLV value per triad = mean over all channel pairs.
    diff_<band>.npy is (n_triads, n_ch, n_ch)."""
    arr = np.load(os.path.join(RUN_DIR, contrast_dir, f"diff_{band}.npy"))
    return arr.reshape(arr.shape[0], -1).mean(axis=1)


# ── gather ───────────────────────────────────────────────────────────────────
data = {b: [per_triad_mean(c["dir"], b) for c in CONTRASTS] for b in BANDS}
pmin = {b: [min_cluster_p(c["dir"], b) for c in CONTRASTS] for b in BANDS}
n_triads = {b: [len(x) for x in data[b]] for b in BANDS}

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
positions = np.arange(len(CONTRASTS)) + 1

for ax, band in zip(axes, BANDS):
    vals = data[band]
    ps   = pmin[band]
    bp = ax.boxplot(vals, positions=positions, widths=0.62, patch_artist=True,
                    showmeans=True, meanline=False,
                    medianprops=dict(color="black", linewidth=1.4),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for i, (box, c, p) in enumerate(zip(bp["boxes"], CONTRASTS, ps)):
        sig = p < SIG_ALPHA
        box.set_facecolor(C_SIG if sig else C_NS)
        box.set_alpha(0.85)
        box.set_edgecolor(C_EDGE)

    # jittered raw triad points
    for i, v in enumerate(vals):
        jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.28
        ax.scatter(positions[i] + jit, v, s=11, color="0.15", alpha=0.45, zorder=3)

    ax.axhline(0, color="0.4", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c['name']}\n({c['math']})" for c in CONTRASTS], fontsize=10)
    ax.set_title(f"{band} band", fontweight="bold")
    ax.set_xlim(0.4, len(CONTRASTS) + 0.6)

# p-value annotations placed after shared ylim is settled
for ax, band in zip(axes, BANDS):
    ymin = ax.get_ylim()[0]
    for i, p in enumerate(pmin[band]):
        tag = f"p={p:.3f}" + (" " if p < SIG_ALPHA else "")
        ax.text(positions[i], ymin + 0.02 * (ax.get_ylim()[1] - ymin), tag,
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if p < SIG_ALPHA else "normal",
                color=C_SIG if p < SIG_ALPHA else "0.35")

axes[0].set_ylabel("Difference in PLV", fontsize=11)  #(per triad, mean over all channel pairs)

legend_handles = [
    Patch(facecolor=C_SIG, edgecolor=C_EDGE, alpha=0.85, label="Cluster p < .05"),
    Patch(facecolor=C_NS,  edgecolor=C_EDGE, alpha=0.85, label="Not significant"),

]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.045))

fig.suptitle(
    "Inter-brain PLV differences across conditions ·  "
    f"N={n_triads['alpha'][0]} triads",
    fontsize=13.5, fontweight="bold")

plt.tight_layout(rect=[0, 0.07, 1, 0.93])
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved", OUT_PNG)
for band in BANDS:
    print(f"\n{band}:")
    for c, v, p in zip(CONTRASTS, data[band], pmin[band]):
        print(f"  {c['math']:14s} mean Δ={v.mean():+.4f}  sd={v.std():.4f}  "
              f"min cluster p={p:.4f}  ({'SIG' if p < SIG_ALPHA else 'ns'})")
