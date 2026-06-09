"""
08_plv_inter_brain.py
─────────────────────────────────────────────────────────────────────────────
Inter-Brain PLV Analysis — Friend vs. Non-Friend (Scalar approach)

For each brain pair × frequency band we compute ONE scalar PLV value:
  PLV = mean over [channels × channel-pairs × time × trials]
       of |exp(i·ΔΦ)|

This is fast (~seconds per pair) and fully sufficient for the learning
objective: "are friend pairs more synchronised than non-friend pairs?"

Statistics: two-sample Mann-Whitney U (non-parametric, no normality assumed,
appropriate for the small N typical of triad EEG studies).

Pipeline
────────
1. Load FG_overview_df.pkl → friend/non-friend label per within-triad pair
2. Load epochs per subject, pool T3P + T3Pn
3. Bandpass + Hilbert → instantaneous phase  (n_trials, n_ch, n_times)
4. Scalar PLV per pair per band
5. Mann-Whitney U test + effect size (rank-biserial r)
6. Boxplot + strip plot of PLV distributions
7. Topomap: mean PLV per channel (averaged over partner channels & trials)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.signal import butter, filtfilt, hilbert
import mne

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths here
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

OVERVIEW_PKL = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\FG_overview_df_v2.pkl"
)

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Conditions
PLV_CONDITIONS = ["T3P", "T3Pn"]

# Frequency bands
FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

# Time window of interest (seconds post-stimulus)
PLV_TMIN, PLV_TMAX = 0.0, 4.0

# Butterworth filter order
FILTER_ORDER = 4

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Pair labels from metadata
# ─────────────────────────────────────────────────────────────────────────────

print("Loading overview dataframe …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} subjects, {fg_df['Triad_id'].nunique()} triads\n")


def build_pair_labels(fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per within-triad pair with columns:
        Triad_id | participant_A | participant_B |
        subj_A   | subj_B        | pair_label
    pair_label = 'friend' iff BOTH members have Friend_status == 'Yes'.
    """
    rows = []
    for triad_id, grp in fg_df.groupby("Triad_id"):
        members = grp.set_index("Participant")[["Subject_id", "Friend_status"]]
        for p_a, p_b in itertools.combinations(sorted(members.index), 2):
            both_friends = (
                members.loc[p_a, "Friend_status"] == "Yes" and
                members.loc[p_b, "Friend_status"] == "Yes"
            )
            rows.append({
                "Triad_id":      triad_id,
                "participant_A": p_a,
                "participant_B": p_b,
                "subj_A":        members.loc[p_a, "Subject_id"],
                "subj_B":        members.loc[p_b, "Subject_id"],
                "pair_label":    "friend" if both_friends else "non-friend",
            })
    return pd.DataFrame(rows)


pair_df = build_pair_labels(fg_df)
print("Pair label distribution:")
print(pair_df["pair_label"].value_counts().to_string(), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load epochs
# ─────────────────────────────────────────────────────────────────────────────

print(f"Found {len(EPOCH_FILES)} epoch files")

subject_epochs: dict[int, mne.Epochs] = {}
info_ref = None

for fpath in EPOCH_FILES:
    exp_id = os.path.basename(fpath).split("_")[0]          # e.g. "301A"
    match  = fg_df[fg_df["Exp_id"] == exp_id]
    if match.empty:
        print(f"  WARNING: no metadata for {exp_id}, skipping.")
        continue

    subj_id = int(match["Subject_id"].iloc[0])
    epochs  = mne.read_epochs(fpath, preload=True, verbose=False)

    available = [c for c in PLV_CONDITIONS if c in epochs.event_id]
    if not available:
        print(f"  {exp_id}: no trio conditions, skipping.")
        continue

    epochs_trio = mne.concatenate_epochs([epochs[c] for c in available])
    subject_epochs[subj_id] = epochs_trio

    if info_ref is None:
        info_ref = epochs_trio.info.copy()

    print(f"  {exp_id} (id={subj_id}): {len(epochs_trio)} trials")

print(f"\n{len(subject_epochs)} subjects loaded\n")

# Shared time mask
sfreq      = info_ref["sfreq"]
times_full = subject_epochs[next(iter(subject_epochs))].times
t_mask     = (times_full >= PLV_TMIN) & (times_full <= PLV_TMAX)
ch_names   = info_ref["ch_names"]
n_channels = len(ch_names)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Phase extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_phase(epochs: mne.Epochs, fmin: float, fmax: float) -> np.ndarray:
    """
    Returns instantaneous phase (n_trials, n_channels, n_times_in_window).
    Zero-phase Butterworth bandpass + Hilbert.
    """
    data     = epochs.get_data()[:, :, t_mask]          # (n_trials, n_ch, n_t)
    b, a     = butter(FILTER_ORDER, [fmin, fmax],
                      btype="bandpass", fs=sfreq, output="ba")
    filtered = filtfilt(b, a, data, axis=-1)
    return np.angle(hilbert(filtered, axis=-1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Scalar PLV per pair per band
#
# For each (channel_A, channel_B) pair:
#   PLV = | mean_trials( exp(i·(φ_A − φ_B)) ) |   averaged over time
# Then average that scalar over all (ch_A, ch_B) combinations.
#
# Memory note: we process one channel of brain A at a time to avoid
# allocating the full (n_trials, n_ch, n_ch, n_times) tensor.
# ─────────────────────────────────────────────────────────────────────────────

def scalar_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """
    Parameters
    ----------
    phase_a, phase_b : (n_trials, n_channels, n_times)

    Returns
    -------
    scalar PLV averaged over all channel pairs and time points.
    """
    n_trials = min(phase_a.shape[0], phase_b.shape[0])
    pa = phase_a[:n_trials]   # (n_trials, n_ch, n_t)
    pb = phase_b[:n_trials]

    # Accumulate PLV channel-by-channel to stay memory-efficient
    total, count = 0.0, 0
    for ci in range(pa.shape[1]):
        # dphi: (n_trials, n_ch_B, n_times)
        dphi = pa[:, ci, np.newaxis, :] - pb          # broadcast over ch_B
        # PLV for this ch_A vs all ch_B: mean over trials → (n_ch_B, n_times)
        plv_ct = np.abs(np.mean(np.exp(1j * dphi), axis=0))
        # Average over ch_B and time → scalar
        total += plv_ct.mean()
        count += 1

    return total / count


print("Computing scalar PLV per pair …\n")

# Results: list of dicts
records = []

for _, row in pair_df.iterrows():
    sid_a, sid_b = row["subj_A"], row["subj_B"]

    if sid_a not in subject_epochs or sid_b not in subject_epochs:
        print(f"  Pair ({sid_a},{sid_b}): missing data, skipping.")
        continue

    rec = {
        "Triad_id":      row["Triad_id"],
        "participant_A": row["participant_A"],
        "participant_B": row["participant_B"],
        "subj_A":        sid_a,
        "subj_B":        sid_b,
        "pair_label":    row["pair_label"],
    }

    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        phase_a = extract_phase(subject_epochs[sid_a], fmin, fmax)
        phase_b = extract_phase(subject_epochs[sid_b], fmin, fmax)
        plv_val = scalar_plv(phase_a, phase_b)
        rec[f"plv_{band_name}"] = plv_val
        print(f"  Triad {row['Triad_id']} "
              f"{row['participant_A']}–{row['participant_B']} "
              f"({row['pair_label']})  {band_name}: PLV={plv_val:.4f}")

    records.append(rec)

results_df = pd.DataFrame(records)
csv_out = os.path.join(OUTPUT_DIR, "plv_scalar_results.csv")
results_df.to_csv(csv_out, index=False)
print(f"\nResults saved → {csv_out}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Statistics: Mann-Whitney U + rank-biserial effect size
#
# rank-biserial r = 1 − (2·U) / (n_friend · n_nonfriend)
# Interpretation: |r| < 0.3 small, 0.3–0.5 medium, > 0.5 large
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STATISTICAL TEST: Mann-Whitney U  (friend vs. non-friend)")
print("=" * 60)

for band_name in FREQ_BANDS:
    col     = f"plv_{band_name}"
    friends = results_df.loc[results_df["pair_label"] == "friend",     col].values
    nonfr   = results_df.loc[results_df["pair_label"] == "non-friend", col].values

    U, p = scipy_stats.mannwhitneyu(friends, nonfr, alternative="two-sided")
    r_rb = 1 - (2 * U) / (len(friends) * len(nonfr))   # rank-biserial r

    print(f"\n{band_name.upper()} ({FREQ_BANDS[band_name][0]}–"
          f"{FREQ_BANDS[band_name][1]} Hz)")
    print(f"  Friend     N={len(friends)}  mean={friends.mean():.4f}  "
          f"sd={friends.std():.4f}")
    print(f"  Non-friend N={len(nonfr)}   mean={nonfr.mean():.4f}  "
          f"sd={nonfr.std():.4f}")
    print(f"  U={U:.1f}  p={p:.4f}  rank-biserial r={r_rb:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Boxplot + strip plot
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"friend": "firebrick", "non-friend": "steelblue"}

def plot_boxstrip():
    n_bands = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]

    fig.suptitle("Inter-Brain PLV: Friend vs. Non-Friend\n(T3P + T3Pn)",
                 fontsize=13, fontweight="bold")

    for ax, (band_name, (fmin, fmax)) in zip(axes, FREQ_BANDS.items()):
        col     = f"plv_{band_name}"
        groups  = ["friend", "non-friend"]
        data    = [results_df.loc[results_df["pair_label"] == g, col].values
                   for g in groups]
        colors  = [COLORS[g] for g in groups]

        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Strip (jitter)
        rng = np.random.default_rng(42)
        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.7,
                       s=30, zorder=3, edgecolors="white", linewidths=0.4)

        # Annotate p-value
        U, p = scipy_stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
        y_top = max(np.concatenate(data)) * 1.05
        ax.plot([1, 1, 2, 2], [y_top, y_top * 1.02, y_top * 1.02, y_top],
                color="black", lw=1)
        sig_str = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else f"p={p:.3f}")
        ax.text(1.5, y_top * 1.03, sig_str, ha="center", va="bottom", fontsize=11)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Friend\n(N={len(data[0])})", f"Non-Friend\n(N={len(data[1])})"],
            fontsize=10,
        )
        ax.set_ylabel("PLV (mean over channels & time)", fontsize=9)
        ax.set_title(f"{band_name.capitalize()} ({fmin}–{fmax} Hz)", fontsize=11)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "plv_boxplot_friend_vs_nonfriend.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.close(fig)


plot_boxstrip()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Topomap: mean PLV per channel
#
# For each channel c_A on brain A, average PLV over:
#   - all partner channels c_B
#   - all trials
#   - the task time window
# Then average across all pairs in each group → one value per channel.
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_plv(band_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (friend_ch, nonfriend_ch) each of shape (n_channels,)."""
    fmin, fmax = FREQ_BANDS[band_name]
    friend_ch, nf_ch = [], []

    for _, row in pair_df.iterrows():
        sid_a, sid_b = row["subj_A"], row["subj_B"]
        if sid_a not in subject_epochs or sid_b not in subject_epochs:
            continue

        phase_a = extract_phase(subject_epochs[sid_a], fmin, fmax)
        phase_b = extract_phase(subject_epochs[sid_b], fmin, fmax)
        n_trials = min(phase_a.shape[0], phase_b.shape[0])
        pa, pb   = phase_a[:n_trials], phase_b[:n_trials]

        # PLV per ch_A: mean over ch_B, trials, time → (n_ch_A,)
        ch_plv = np.zeros(n_channels)
        for ci in range(n_channels):
            dphi = pa[:, ci, np.newaxis, :] - pb
            ch_plv[ci] = np.abs(np.mean(np.exp(1j * dphi))).mean()

        if row["pair_label"] == "friend":
            friend_ch.append(ch_plv)
        else:
            nf_ch.append(ch_plv)

    f_arr  = np.array(friend_ch).mean(axis=0) if friend_ch else np.zeros(n_channels)
    nf_arr = np.array(nf_ch).mean(axis=0)     if nf_ch     else np.zeros(n_channels)
    return f_arr, nf_arr


def plot_topomaps():
    n_bands = len(FREQ_BANDS)
    fig, axes = plt.subplots(3, n_bands, figsize=(4.5 * n_bands, 11))
    if n_bands == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Inter-Brain PLV — Channel Topography\n"
        "Rows: Friend | Non-Friend | Difference (F − NF)",
        fontsize=11, fontweight="bold",
    )

    for col, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        print(f"  Building channel PLV for topomap: {band_name} …")
        f_ch, nf_ch = build_channel_plv(band_name)
        diff_ch     = f_ch - nf_ch

        vmax_raw  = max(f_ch.max(), nf_ch.max(), 1e-9)
        vmax_diff = max(abs(diff_ch).max(), 1e-9)

        row_specs = [
            (f_ch,    "Friend",     "Reds",   (0, vmax_raw)),
            (nf_ch,   "Non-Friend", "Reds",   (0, vmax_raw)),
            (diff_ch, "Diff F−NF",  "RdBu_r", (-vmax_diff, vmax_diff)),
        ]

        for row_idx, (data, label, cmap, vlim) in enumerate(row_specs):
            ax = axes[row_idx, col]
            mne.viz.plot_topomap(data, info_ref, axes=ax, show=False,
                                 cmap=cmap, vlim=vlim)
            ax.set_title(
                f"{band_name.capitalize()} ({fmin}–{fmax} Hz)\n{label}",
                fontsize=9,
            )

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "plv_topomap_friend_vs_nonfriend.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


plot_topomaps()

print("\nDone.")