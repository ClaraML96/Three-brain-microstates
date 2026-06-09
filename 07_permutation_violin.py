import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
import mne
from mne.stats import spatio_temporal_cluster_1samp_test

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (keep in sync with 07_permutation.py)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\violin_erd"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNELS = ["C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "Oz", "O2"]

FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

FOI      = np.linspace(1, 30, 30, dtype=int)
N_CYCLES = 3 + 0.5 * FOI
BASELINE = (-0.25, 0)
PLOT_TMIN, PLOT_TMAX = 0.0, 4.0

N_PERMUTATIONS = 1024
ALPHA_CLUSTER  = 0.05
P_ACCEPT       = 0.05

CONTRASTS = [
    ("solo_feedback",    "T1P",  "T1Pn", "Solo: With vs. No Feedback"),
    ("trio_feedback",    "T3P",  "T3Pn", "Trio: With vs. No Feedback"),
    ("solo_vs_trio_fb",  "T1P",  "T3P",  "Solo vs. Trio (With Feedback)"),
    ("solo_vs_trio_nfb", "T1Pn", "T3Pn", "Solo vs. Trio (No Feedback)"),
]

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
# STEP 1 — Load epochs, compute TFR (identical to 07_permutation.py)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Found {len(EPOCH_FILES)} epoch files\n")

group_tfr: dict[str, list] = {}
info_ref = None

for epoch_file in EPOCH_FILES:
    if not os.path.isfile(epoch_file):
        raise FileNotFoundError(epoch_file)

    print(f"Processing: {os.path.basename(epoch_file)}")
    epochs = mne.read_epochs(epoch_file, preload=True)

    available = [ch for ch in CHANNELS if ch in epochs.ch_names]
    epochs.pick(available)

    if info_ref is None:
        info_ref = epochs.info.copy()

    solo_trio_keys = [k for k in epochs.event_id if k in ("T1P", "T1Pn", "T3P", "T3Pn")]

    for condition in solo_trio_keys:
        epochs_cond = epochs[condition]
        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=FOI,
            n_cycles=N_CYCLES,
            return_itc=False,
            average=False,
        )
        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(BASELINE, mode="percent")
        tfr_avg.data *= 100
        group_tfr.setdefault(condition, []).append(tfr_avg)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build adjacency (identical to 07_permutation.py)
# ─────────────────────────────────────────────────────────────────────────────
adjacency, _ = mne.channels.find_ch_adjacency(info_ref, ch_type="eeg")
ch_names = info_ref.ch_names
print(f"\nChannel adjacency built for: {ch_names}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (identical to 07_permutation.py)
# ─────────────────────────────────────────────────────────────────────────────
def band_spatiotemporal(tfr, fmin, fmax, t_mask):
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    band_data = tfr.data[:, f_mask, :].mean(axis=1)
    return band_data[:, t_mask].T


def build_X(tfr_list_a, tfr_list_b, fmin, fmax, times_full, tmin, tmax):
    t_mask = (times_full >= tmin) & (times_full <= tmax)
    n = min(len(tfr_list_a), len(tfr_list_b))
    st_a = np.stack([band_spatiotemporal(t, fmin, fmax, t_mask) for t in tfr_list_a[:n]])
    st_b = np.stack([band_spatiotemporal(t, fmin, fmax, t_mask) for t in tfr_list_b[:n]])
    return st_a - st_b, st_a, st_b


def run_spatio_temporal_cluster(X, n_obs, adjacency):
    df        = n_obs - 1
    threshold = scipy_stats.t.ppf(1.0 - ALPHA_CLUSTER / 2.0, df=df)
    T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=N_PERMUTATIONS,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        out_type="mask",
        verbose=True,
        seed=42,
        n_jobs=1,
    )
    return T_obs, clusters, cluster_p, H0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run cluster tests and collect per-subject cluster-masked ERD
# ─────────────────────────────────────────────────────────────────────────────
first_key  = next(iter(group_tfr))
times_full = group_tfr[first_key][0].times
t_mask     = (times_full >= PLOT_TMIN) & (times_full <= PLOT_TMAX)
times_plot = times_full[t_mask]

results = {}

for contrast_label, cond_a, cond_b, contrast_title in CONTRASTS:
    if cond_a not in group_tfr or cond_b not in group_tfr:
        print(f"Skipping {contrast_label}: missing condition data.")
        continue

    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        print(f"\n{'='*60}")
        print(f"Contrast: {contrast_title}  |  {band_name} ({fmin}–{fmax} Hz)")

        X, st_a, st_b = build_X(
            group_tfr[cond_a], group_tfr[cond_b],
            fmin, fmax, times_full, PLOT_TMIN, PLOT_TMAX,
        )
        n_subjects = X.shape[0]

        T_obs, clusters, cluster_p, H0 = run_spatio_temporal_cluster(
            X, n_subjects, adjacency
        )

        sig_mask = np.zeros_like(T_obs, dtype=bool)
        sig_clusters = []
        for c, p in zip(clusters, cluster_p):
            if p < P_ACCEPT:
                sig_mask |= c
                sig_clusters.append((c, p))

        # ── Per-subject mean ERD within the significant cluster mask ──────────
        # Shape of st_a / st_b: (n_subjects, n_times, n_channels)
        # sig_mask shape:        (n_times, n_channels)
        # We average each subject's ERD values at the cluster's (time, channel)
        # points.  When no significant cluster exists we fall back to averaging
        # across all retained time points and channels (whole-ROI mean).

        if sig_mask.any():
            # Flatten time×channel into a 1-D mask for indexing
            flat_mask = sig_mask.reshape(-1)                           # (T*C,)
            subj_a = st_a.reshape(n_subjects, -1)[:, flat_mask].mean(axis=1)
            subj_b = st_b.reshape(n_subjects, -1)[:, flat_mask].mean(axis=1)
            cluster_label = "cluster-masked mean ERD (%)"
        else:
            subj_a = st_a.mean(axis=(1, 2))   # full ROI / time average
            subj_b = st_b.mean(axis=(1, 2))
            cluster_label = "whole-ROI mean ERD (%)"

        # Cohen's d (paired)
        diff   = subj_a - subj_b
        cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)

        # Paired t-test on the cluster-averaged values (descriptive complement)
        t_stat, p_ttest = scipy_stats.ttest_rel(subj_a, subj_b)

        results[(contrast_label, band_name)] = dict(
            T_obs=T_obs,
            sig_mask=sig_mask,
            sig_clusters=sig_clusters,
            subj_a=subj_a,          # (n_subjects,) — per-subject ERD, cond A
            subj_b=subj_b,          # (n_subjects,) — per-subject ERD, cond B
            cohens_d=cohens_d,
            t_stat=t_stat,
            p_ttest=p_ttest,
            cluster_label=cluster_label,
            n_subjects=n_subjects,
            contrast_title=contrast_title,
            cond_a=cond_a,
            cond_b=cond_b,
            fmin=fmin,
            fmax=fmax,
        )

        print(f"  Sig clusters: {len(sig_clusters)}  |  "
              f"Cohen's d = {cohens_d:.3f}  |  "
              f"paired t({n_subjects-1}) = {t_stat:.3f}, p = {p_ttest:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Violin plot: 4 contrasts × 2 bands in one figure (2 × 4 grid)
# ─────────────────────────────────────────────────────────────────────────────
def half_violin(ax, data, pos, side, color, width=0.35):
    """Draw a half-violin on one side of `pos`.

    Parameters
    ----------
    ax    : matplotlib Axes
    data  : 1-D array of per-subject values
    pos   : x-position of the violin spine
    side  : 'left' or 'right'  (which half to draw)
    color : fill color
    width : half-width of the violin in data coordinates
    """
    kde = gaussian_kde(data, bw_method="scott")
    y_grid = np.linspace(data.min() - 2, data.max() + 2, 300)
    density = kde(y_grid)
    density = density / density.max() * width   # normalise to `width`

    if side == "left":
        xs_fill = np.concatenate([[pos], pos - density, [pos]])
    else:
        xs_fill = np.concatenate([[pos], pos + density, [pos]])

    ys_fill = np.concatenate([[y_grid[0]], y_grid, [y_grid[-1]]])
    ax.fill_betweenx(y_grid,
                     pos,
                     pos - density if side == "left" else pos + density,
                     color=color, alpha=0.30)
    ax.plot(pos - density if side == "left" else pos + density,
            y_grid, color=color, lw=1.2)

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
    ax.scatter(dot_x, data, color=color, s=18, alpha=0.55, zorder=5,
               linewidths=0)


def significance_bracket(ax, x1, x2, y, p_val):
    """Draw a horizontal bracket with p-value annotation."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.8, color="0.35")
    if p_val < 0.001:
        label = "***"
    elif p_val < 0.01:
        label = "**"
    elif p_val < 0.05:
        label = "*"
    else:
        label = "n.s."
    ax.text((x1 + x2) / 2, y + h * 1.2, label,
            ha="center", va="bottom", fontsize=9, color="0.35")


def plot_violin_summary():
    bands   = list(FREQ_BANDS.keys())
    n_bands = len(bands)
    n_cont  = len(CONTRASTS)

    fig, axes = plt.subplots(
        n_bands, n_cont,
        figsize=(4.2 * n_cont, 4.5 * n_bands),
        sharey=False,
    )
    # Ensure 2-D indexing even with 1 row/col
    if n_bands == 1:
        axes = axes[np.newaxis, :]
    if n_cont == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "ERD per Subject — Cluster-masked Mean\n"
        "(split violin: left = cond A, right = cond B)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, band_name in enumerate(bands):
        for col, (contrast_label, cond_a, cond_b, contrast_title) in enumerate(CONTRASTS):
            ax  = axes[row, col]
            key = (contrast_label, band_name)

            if key not in results:
                ax.axis("off")
                continue

            r       = results[key]
            subj_a  = r["subj_a"]
            subj_b  = r["subj_b"]
            fmin    = r["fmin"]
            fmax    = r["fmax"]

            # ── Draw split violins centred at x = 0 ──────────────────────────
            half_violin(ax, subj_a, pos=0, side="left",  color=COLORS["a"])
            half_violin(ax, subj_b, pos=0, side="right", color=COLORS["b"])

            # ── Baseline at zero ──────────────────────────────────────────────
            ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=1)

            # ── Significance bracket (cluster permutation p) ──────────────────
            # Use the smallest cluster p if multiple sig clusters exist
            if r["sig_clusters"]:
                best_p = min(p for _, p in r["sig_clusters"])
                ymax   = max(subj_a.max(), subj_b.max())
                y_br   = ymax + abs(ymax) * 0.08 + 1
                ax.set_ylim(top=y_br + abs(y_br) * 0.25)
                significance_bracket(ax, -0.25, 0.25, y_br, best_p)
                ptext = f"p = {best_p:.3f} (cluster perm.)"
            else:
                ptext = "n.s. (cluster perm.)"

            # ── Cohen's d annotation ──────────────────────────────────────────
            ax.text(0.97, 0.04,
                    f"d = {r['cohens_d']:.2f}",
                    transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="0.4",
                    style="italic")

            # ── Labels / formatting ───────────────────────────────────────────
            ax.set_xlim(-0.55, 0.55)
            ax.set_xticks([-0.17, 0.17])
            ax.set_xticklabels(
                [COND_LABELS.get(cond_a, cond_a),
                 COND_LABELS.get(cond_b, cond_b)],
                fontsize=8,
            )
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel("Mean ERD (%)", fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.spines[["top", "right", "bottom"]].set_visible(False)

            if row == 0:
                ax.set_title(
                    f"{contrast_title}\n"
                    f"{band_name.capitalize()} ({fmin}–{fmax} Hz)\n"
                    f"N = {r['n_subjects']}",
                    fontsize=9, fontweight="bold",
                )
            else:
                ax.set_title(
                    f"{band_name.capitalize()} ({fmin}–{fmax} Hz)",
                    fontsize=9,
                )

            # Subtitle with cluster label + p-text
            ax.set_xlabel(
                f"{r['cluster_label']}\n{ptext}",
                fontsize=7, color="0.5",
            )

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLORS["a"], alpha=0.65, label="Condition A"),
        mpatches.Patch(facecolor=COLORS["b"], alpha=0.65, label="Condition B"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2,
        fontsize=9, framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "violin_erd_summary.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Results table (cluster-masked effect sizes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VIOLIN SUMMARY — CLUSTER-MASKED EFFECT SIZES")
print("=" * 70)
for (contrast_label, band_name), r in results.items():
    print(f"\n{r['contrast_title']}  |  {band_name.capitalize()} "
          f"({r['fmin']}–{r['fmax']} Hz)  |  N = {r['n_subjects']}")
    print(f"  {r['cluster_label']}")
    print(f"  {COND_LABELS.get(r['cond_a'], r['cond_a'])}: "
          f"M = {r['subj_a'].mean():.2f}%,  SD = {r['subj_a'].std(ddof=1):.2f}%")
    print(f"  {COND_LABELS.get(r['cond_b'], r['cond_b'])}: "
          f"M = {r['subj_b'].mean():.2f}%,  SD = {r['subj_b'].std(ddof=1):.2f}%")
    print(f"  Cohen's d = {r['cohens_d']:.3f}  |  "
          f"paired t({r['n_subjects']-1}) = {r['t_stat']:.3f}, "
          f"p = {r['p_ttest']:.4f} (descriptive — do not interpret as primary test)")
    if r["sig_clusters"]:
        for i, (c, p) in enumerate(r["sig_clusters"]):
            t_idx, ch_idx = np.where(c)
            t_span = times_plot[t_idx]
            chans  = [ch_names[j] for j in np.unique(ch_idx)]
            print(f"  ✓ Cluster {i+1}: {t_span.min():.3f}–{t_span.max():.3f} s  "
                  f"| channels: {chans}  | p = {p:.4f}")
    else:
        print(f"  ✗ No significant clusters (p < {P_ACCEPT})")

plot_violin_summary()
print("\nDone.")