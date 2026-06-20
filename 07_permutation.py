import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import mne
from mne.stats import spatio_temporal_cluster_1samp_test

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\cluster_perm"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ROI
CHANNELS = ["C3", "O1", "Oz", "O2"]


# Frequency bands
FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

# TFR / Morlet parameters (must match your ERD pipeline)
FOI      = np.linspace(1, 30, 30, dtype=int)
N_CYCLES = 3 + 0.5 * FOI
BASELINE = (-0.25, 0)        # seconds
PLOT_TMIN, PLOT_TMAX = 0.0, 4.0

# Cluster permutation settings
N_PERMUTATIONS = 1024        # increase to 5000 for final analysis
ALPHA_CLUSTER  = 0.05        # cluster-forming threshold (p-value, two-tailed)
P_ACCEPT       = 0.05        # cluster significance level

# Solo/Trio contrasts only
CONTRASTS = [
    ("solo_feedback",    "T1P",  "T1Pn", "Solo: With vs. No Feedback"),
    ("trio_feedback",    "T3P",  "T3Pn", "Trio: With vs. No Feedback"),
    ("solo_vs_trio_fb",  "T1P",  "T3P",  "Solo vs. Trio (With Feedback)"),
    ("solo_vs_trio_nfb", "T1Pn", "T3Pn", "Solo vs. Trio (No Feedback)"),
]

COND_LABELS = {
    "T1P":  "Solo — With Feedback",
    "T1Pn": "Solo — No Feedback",
    "T3P":  "Trio — With Feedback",
    "T3Pn": "Trio — No Feedback",
}

COLORS = {"a": "firebrick", "b": "steelblue"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load epochs, compute TFR, store per-subject AverageTFR objects
# ─────────────────────────────────────────────────────────────────────────────
print(f"Found {len(EPOCH_FILES)} epoch files\n")

# group_tfr[condition_key] = [AverageTFR_subject1, AverageTFR_subject2, ...]
group_tfr: dict[str, list] = {}

# We also need one Info object to build the adjacency matrix later
info_ref = None

for epoch_file in EPOCH_FILES:
    if not os.path.isfile(epoch_file):
        raise FileNotFoundError(epoch_file)

    print(f"Processing: {os.path.basename(epoch_file)}")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Keep only the channels we want, in the order defined above
    available = [ch for ch in CHANNELS if ch in epochs.ch_names]
    epochs.pick(available)

    if info_ref is None:
        info_ref = epochs.info.copy()

    # Only Solo and Trio conditions
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
        tfr_avg.data *= 100  # → % signal change

        group_tfr.setdefault(condition, []).append(tfr_avg)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build channel adjacency matrix from the electrode montage
# ─────────────────────────────────────────────────────────────────────────────
adjacency, ch_names_adj = mne.channels.find_ch_adjacency(info_ref, ch_type="eeg")

# Reorder to match our CHANNELS list (may differ from info order)
ch_order = [info_ref.ch_names.index(ch) for ch in info_ref.ch_names]
# adjacency already matches info_ref order — just note the channel order
print(f"\nChannel adjacency built for: {info_ref.ch_names}")
print(f"Adjacency matrix shape: {adjacency.shape}\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# Frequency-averaged ERD for every channel at every (masked) time point
def band_spatiotemporal(tfr, fmin: float, fmax: float,
                        t_mask: np.ndarray) -> np.ndarray:
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    # mean over frequency axis -> (n_channels, n_times)
    band_data = tfr.data[:, f_mask, :].mean(axis=1)
    # apply time mask and transpose -> (n_times, n_channels)

    # Returns
    # -------
    # arr : (n_times_masked, n_channels)

    return band_data[:, t_mask].T

# Build X = condA − condB for the paired cluster test
def build_X(tfr_list_a, tfr_list_b, fmin, fmax, times_full, tmin, tmax):
    t_mask = (times_full >= tmin) & (times_full <= tmax)
    n = min(len(tfr_list_a), len(tfr_list_b))

    st_a = np.stack([band_spatiotemporal(t, fmin, fmax, t_mask)
                     for t in tfr_list_a[:n]])   # (n, n_times, n_ch)
    st_b = np.stack([band_spatiotemporal(t, fmin, fmax, t_mask)
                     for t in tfr_list_b[:n]])

    # Returns
    # -------
    # X      : (n_subjects, n_times, n_channels)  — difference array
    # st_a   : (n_subjects, n_times, n_channels)  — raw condA values
    # st_b   : (n_subjects, n_times, n_channels)  — raw condB values

    return st_a - st_b, st_a, st_b


# Paired spatio-temporal cluster permutation test
def run_spatio_temporal_cluster(X, n_obs, adjacency):
    df        = n_obs - 1
    threshold = scipy_stats.t.ppf(1.0 - ALPHA_CLUSTER / 2.0, df=df)

    # spatio_temporal_cluster_1samp_test expects (observations, times, space)
    T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=N_PERMUTATIONS,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        out_type="mask",   # boolean mask same shape as T_obs
        verbose=True,
        seed=42,
        n_jobs=1,
    )

    # Returns
    # -------
    # T_obs (n_times × n_channels), clusters, cluster_p, H0
    return T_obs, clusters, cluster_p, H0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run all contrasts
# ─────────────────────────────────────────────────────────────────────────────
first_key  = next(iter(group_tfr))
times_full = group_tfr[first_key][0].times
t_mask     = (times_full >= PLOT_TMIN) & (times_full <= PLOT_TMAX)
times_plot = times_full[t_mask]

ch_names = info_ref.ch_names  # channel order used throughout

results = {}  # (contrast_label, band_name) → dict

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
        print(f"N subjects = {n_subjects},  "
              f"data shape = {X.shape}  (subj × times × channels)")

        T_obs, clusters, cluster_p, H0 = run_spatio_temporal_cluster(
            X, n_subjects, adjacency
        )

        # Significant cluster mask: union over all sig. clusters
        sig_mask = np.zeros_like(T_obs, dtype=bool)  # (n_times, n_channels)
        sig_clusters = []
        for c, p in zip(clusters, cluster_p):
            if p < P_ACCEPT:
                sig_mask |= c
                sig_clusters.append((c, p))

        results[(contrast_label, band_name)] = dict(
            T_obs=T_obs,              # (n_times, n_channels)
            clusters=clusters,
            cluster_p=cluster_p,
            sig_mask=sig_mask,        # (n_times, n_channels)
            sig_clusters=sig_clusters,
            mean_a=st_a.mean(axis=0), # (n_times, n_channels)
            sem_a=st_a.std(axis=0) / np.sqrt(n_subjects),
            mean_b=st_b.mean(axis=0),
            sem_b=st_b.std(axis=0) / np.sqrt(n_subjects),
            n_subjects=n_subjects,
            contrast_title=contrast_title,
            cond_a=cond_a,
            cond_b=cond_b,
            fmin=fmin,
            fmax=fmax,
        )

        print(f"Significant spatio-temporal clusters (p < {P_ACCEPT}): "
              f"{len(sig_clusters)}")
        for i, (c, p) in enumerate(sig_clusters):
            t_idx, ch_idx = np.where(c)
            t_span = times_plot[t_idx]
            chans  = [ch_names[j] for j in np.unique(ch_idx)]
            print(f"  Cluster {i+1}: {t_span.min():.3f}–{t_span.max():.3f} s  "
                  f"| channels: {chans}  | p = {p:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Plot: per-channel ERD timecourses with cluster shading
#          One figure per contrast, sub-panels = channels, columns = bands
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_channel(contrast_label, cond_a, cond_b, contrast_title):
    bands   = list(FREQ_BANDS.keys())
    n_ch    = len(ch_names)
    n_bands = len(bands)

    fig, axes = plt.subplots(
        n_ch, n_bands,
        figsize=(5 * n_bands, 2.2 * n_ch),
        sharey=False, sharex=True,
    )
    if n_ch == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Cluster Permutation Test: {contrast_title}\n"
        f"Per-channel ERD/ERS  |  gold = significant cluster (p < {P_ACCEPT})",
        fontsize=12, fontweight="bold",
    )

    for col, band_name in enumerate(bands):
        key = (contrast_label, band_name)
        if key not in results:
            continue
        r = results[key]

        for row, ch in enumerate(ch_names):
            ax  = axes[row, col]
            ci  = ch_names.index(ch)   # channel index in data

            ma  = r["mean_a"][:, ci]
            sa  = r["sem_a"][:, ci]
            mb  = r["mean_b"][:, ci]
            sb  = r["sem_b"][:, ci]
            sig = r["sig_mask"][:, ci]  # boolean (n_times,)

            ax.plot(times_plot, ma, lw=1.5, color=COLORS["a"],
                    label=COND_LABELS.get(cond_a, cond_a))
            ax.fill_between(times_plot, ma - sa, ma + sa,
                            alpha=0.15, color=COLORS["a"])

            ax.plot(times_plot, mb, lw=1.5, color=COLORS["b"],
                    label=COND_LABELS.get(cond_b, cond_b))
            ax.fill_between(times_plot, mb - sb, mb + sb,
                            alpha=0.15, color=COLORS["b"])

            # Shade significant time points for this channel
            if sig.any():
                # Convert boolean mask to contiguous spans
                changes  = np.diff(sig.astype(int), prepend=0, append=0)
                starts   = np.where(changes == 1)[0]
                ends     = np.where(changes == -1)[0]
                for s, e in zip(starts, ends):
                    ax.axvspan(times_plot[s], times_plot[e - 1],
                               color="gold", alpha=0.40)

            ax.axhline(0,   color="k",    lw=0.6, ls="--")
            ax.axvline(0.0, color="gray", lw=0.6, ls=":")
            ax.set_xlim(PLOT_TMIN, PLOT_TMAX)
            ax.set_ylabel(f"{ch}\n(%)", fontsize=8)

            if row == 0:
                fmin, fmax = r["fmin"], r["fmax"]
                ax.set_title(
                    f"{band_name.capitalize()} ({fmin}–{fmax} Hz)\n"
                    f"N = {r['n_subjects']}",
                    fontsize=10,
                )
            if row == n_ch - 1:
                ax.set_xlabel("Time (s)", fontsize=9)

            # Legend only on the top-left panel
            if row == 0 and col == 0:
                ax.legend(fontsize=7, framealpha=0.8, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"cluster_perm_perchannel_{contrast_label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Topographic cluster-mass map
#          Shows how strongly each channel participates in sig. clusters
# ─────────────────────────────────────────────────────────────────────────────
def plot_topomap(contrast_label, cond_a, cond_b, contrast_title):
    bands   = list(FREQ_BANDS.keys())
    n_bands = len(bands)

    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        f"Cluster T-mass Topomap: {contrast_title}\n"
        f"(sum of |T| within significant clusters per channel)",
        fontsize=11, fontweight="bold",
    )

    for ax, band_name in zip(axes, bands):
        key = (contrast_label, band_name)
        if key not in results:
            ax.axis("off")
            continue
        r = results[key]

        # Channel t-mass: sum of |T_obs| at time points belonging to any
        # significant cluster for each channel
        t_mass = np.zeros(len(ch_names))
        if r["sig_mask"].any():
            for ci in range(len(ch_names)):
                ch_sig = r["sig_mask"][:, ci]
                t_mass[ci] = np.abs(r["T_obs"][ch_sig, ci]).sum()

        mne.viz.plot_topomap(
            t_mass,
            info_ref,
            axes=ax,
            show=False,
            cmap="Reds",
            vlim=(0, t_mass.max() if t_mass.max() > 0 else 1),
        )
        fmin, fmax = r["fmin"], r["fmax"]
        ax.set_title(
            f"{band_name.capitalize()} ({fmin}–{fmax} Hz)\n"
            f"N = {r['n_subjects']}",
            fontsize=10,
        )

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"cluster_perm_topomap_{contrast_label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Run plotting for every contrast
# ─────────────────────────────────────────────────────────────────────────────
for contrast_label, cond_a, cond_b, contrast_title in CONTRASTS:
    if (contrast_label, "alpha") not in results:
        continue
    plot_per_channel(contrast_label, cond_a, cond_b, contrast_title)
    plot_topomap(contrast_label, cond_a, cond_b, contrast_title)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SPATIO-TEMPORAL CLUSTER PERMUTATION TEST — RESULTS SUMMARY")
print("=" * 70)
for (contrast_label, band_name), r in results.items():
    print(f"\n{r['contrast_title']}  |  {band_name.capitalize()} "
          f"({r['fmin']}–{r['fmax']} Hz)  |  N = {r['n_subjects']}")
    print(f"  Cond A: {COND_LABELS.get(r['cond_a'], r['cond_a'])}")
    print(f"  Cond B: {COND_LABELS.get(r['cond_b'], r['cond_b'])}")
    if r["sig_clusters"]:
        for i, (c, p) in enumerate(r["sig_clusters"]):
            t_idx, ch_idx = np.where(c)
            t_span = times_plot[t_idx]
            chans  = [ch_names[j] for j in np.unique(ch_idx)]
            print(f"  ✓ Cluster {i+1}: {t_span.min():.3f}–{t_span.max():.3f} s  "
                  f"| channels: {chans}  | p = {p:.4f}")
    else:
        print(f"  ✗ No significant clusters (p < {P_ACCEPT})")

print("\nDone.")