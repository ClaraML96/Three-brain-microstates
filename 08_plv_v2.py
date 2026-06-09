"""
08_plv.py — v2 (revision of `raw/scripts/mine/08_plv.py`)
─────────────────────────────────────────────────────────────────────────────
Inter-Brain PLV Analysis — Friend vs. Non-Friend (scalar approach)

Addresses the three blocking findings in `08_plv-review.md`:

  (1) Selection-based trial alignment per triad.
      `epochs.selection` is intersected across A, B, C so that across-trial
      PLV averages correspond to the SAME original task events. The v1
      script used positional `min(n_a, n_b)` — that paired
      non-corresponding trials.

  (2) Triad-level aggregation.
      For each triad, the two non-friend pairs (A–C, B–C) share
      participant C and are not independent. v1 kept all three pairs
      (N=31 vs N=60). v2 averages the two non-friend pairs per triad →
      one friend PLV and one averaged-non-friend PLV per triad. The test
      becomes a within-triad paired comparison (Wilcoxon signed-rank).

  (4) Topomap PLV reduction fixed.
      v1 used `np.abs(np.mean(np.exp(1j·dphi)))` with no `axis`, which
      collapses trials, ch_B, and time *before* taking the magnitude —
      a different (smaller) quantity than the scalar table. v2 takes the
      magnitude after reducing trials only, matching `scalar_plv`.

Other improvements:
  - filter-then-crop: bandpass on the full epoch, then mask the window.
  - per-(subject, band) phase cache: no double Hilbert/filter when
    computing scalar PLV and per-channel PLV.
  - optional trial-shuffle null distribution (finding 5; off by default
    via N_PERM=0 — set N_PERM≥200 to enable).

Not addressed here (deferred):
  - Triad 330 / Triad 319 missing-subject substitution (finding 3).
  - Channel-pair averaging dilution (finding 6) — kept as all-pairs.
  - T3P / T3Pn pooling (finding 7) — pooled, single condition set.
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
# CONFIGURATION
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

PLV_CONDITIONS = ["T3P", "T3Pn"]

FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

PLV_TMIN, PLV_TMAX = 0.0, 4.0
FILTER_ORDER = 4

# Trial-shuffle null. Set to e.g. 200 to enable; expensive (~N_PERM × scalar
# PLV per pair × band). Off by default.
N_PERM = 0
RNG_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Pair labels from metadata
# ─────────────────────────────────────────────────────────────────────────────

print("Loading overview dataframe …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} subjects, {fg_df['Triad_id'].nunique()} triads\n")


def build_pair_labels(fg_df: pd.DataFrame) -> pd.DataFrame:
    """One row per within-triad pair, with friend/non-friend label."""
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
print("Pair label distribution (before triad-level aggregation):")
print(pair_df["pair_label"].value_counts().to_string(), "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load epochs
#
# Concatenate T3P + T3Pn per subject. `mne.concatenate_epochs` preserves
# per-source `selection`, but the concatenated object's `selection` is
# reindexed. We keep the concatenated epochs because the v1 design pooled
# conditions, AND because per-subject selection still uniquely identifies
# trials WITHIN that subject's pooled set — what matters for across-trial
# PLV is that A's trial k and B's trial k correspond to the SAME original
# task event. We achieve that by intersecting `selection` (the original
# trial indices that survived autoreject) across A and B.
# ─────────────────────────────────────────────────────────────────────────────

print(f"Found {len(EPOCH_FILES)} epoch files")

subject_epochs: dict[int, mne.Epochs] = {}
info_ref = None

for fpath in EPOCH_FILES:
    exp_id = os.path.basename(fpath).split("_")[0]
    match = fg_df[fg_df["Exp_id"] == exp_id]
    if match.empty:
        print(f"  WARNING: no metadata for {exp_id}, skipping.")
        continue

    subj_id = int(match["Subject_id"].iloc[0])
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)

    available = [c for c in PLV_CONDITIONS if c in epochs.event_id]
    if not available:
        print(f"  {exp_id}: no trio conditions, skipping.")
        continue

    epochs_trio = mne.concatenate_epochs([epochs[c] for c in available])

    if subj_id in subject_epochs:
        print(f"  WARNING: Subject_id {subj_id} already loaded "
              f"(exp_id={exp_id}); overwriting. Investigate.")
    subject_epochs[subj_id] = epochs_trio

    if info_ref is None:
        info_ref = epochs_trio.info.copy()

    print(f"  {exp_id} (id={subj_id}): {len(epochs_trio)} trials, "
          f"selection range [{epochs_trio.selection.min()}, "
          f"{epochs_trio.selection.max()}]")

print(f"\n{len(subject_epochs)} subjects loaded\n")

sfreq      = info_ref["sfreq"]
times_full = subject_epochs[next(iter(subject_epochs))].times
t_mask     = (times_full >= PLV_TMIN) & (times_full <= PLV_TMAX)
ch_names   = info_ref["ch_names"]
n_channels = len(ch_names)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Phase extraction with caching
#
# Filter on the FULL epoch (avoids cropping into Butterworth transients),
# then mask the analysis window. Cache per (subject, band) so each
# subject's phase is computed exactly once per band.
# ─────────────────────────────────────────────────────────────────────────────

_phase_cache: dict[tuple[int, str], np.ndarray] = {}


def get_phase(subj_id: int, band: str) -> np.ndarray:
    """
    (n_trials, n_channels, n_times_in_window) for the given subject and band.
    Uses cache; filters the full epoch then crops.
    """
    key = (subj_id, band)
    if key in _phase_cache:
        return _phase_cache[key]

    fmin, fmax = FREQ_BANDS[band]
    epochs = subject_epochs[subj_id]
    data_full = epochs.get_data()                       # (n_tr, n_ch, n_t_full)

    b, a = butter(FILTER_ORDER, [fmin, fmax],
                  btype="bandpass", fs=sfreq, output="ba")
    filtered_full = filtfilt(b, a, data_full, axis=-1)
    analytic_full = hilbert(filtered_full, axis=-1)
    phase = np.angle(analytic_full[:, :, t_mask])

    _phase_cache[key] = phase
    return phase


def align_by_selection(subj_a: int, subj_b: int):
    """
    Return (idx_a, idx_b, common_orig_indices) such that
    subject_epochs[subj_a][idx_a[i]] and subject_epochs[subj_b][idx_b[i]]
    correspond to the SAME original task event.

    Empty arrays if no common trials.
    """
    sel_a = subject_epochs[subj_a].selection
    sel_b = subject_epochs[subj_b].selection
    common = np.intersect1d(sel_a, sel_b)
    if len(common) == 0:
        return np.array([], int), np.array([], int), common
    idx_a = np.searchsorted(sel_a, common)
    idx_b = np.searchsorted(sel_b, common)
    return idx_a, idx_b, common

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Scalar PLV per aligned pair
#
# Same formula as v1's scalar_plv, applied to selection-aligned phase
# arrays. PLV = mean over channel-pairs and time of
# |mean_trials( exp(i·(φ_A − φ_B)) )|.
# ─────────────────────────────────────────────────────────────────────────────

def scalar_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """
    phase_a, phase_b : (n_trials, n_channels, n_times)  — already aligned.

    Returns scalar PLV averaged over all (ch_A, ch_B) and time.
    """
    assert phase_a.shape == phase_b.shape, "Phase arrays must be aligned."
    total, count = 0.0, 0
    for ci in range(phase_a.shape[1]):
        dphi = phase_a[:, ci, np.newaxis, :] - phase_b      # (n_tr, n_ch_B, n_t)
        plv_ct = np.abs(np.mean(np.exp(1j * dphi), axis=0)) # mean over trials, then abs
        total += plv_ct.mean()
        count += 1
    return total / count


def channel_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Per-channel-A PLV: |mean_trials(exp(i·dphi))| averaged over ch_B and time.

    Returns shape (n_channels,) — same formula as scalar_plv, but kept
    per ch_A instead of collapsing. v1's bug was averaging exp(i·dphi)
    over ALL axes before taking abs.
    """
    n_ch = phase_a.shape[1]
    out = np.zeros(n_ch)
    for ci in range(n_ch):
        dphi = phase_a[:, ci, np.newaxis, :] - phase_b
        plv_ct = np.abs(np.mean(np.exp(1j * dphi), axis=0))   # (n_ch_B, n_t)
        out[ci] = plv_ct.mean()
    return out


def trial_shuffle_null(phase_a: np.ndarray, phase_b: np.ndarray,
                       n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """
    Null PLV distribution under destroyed trial-by-trial correspondence.
    Shuffles phase_a's trial order; phase_b is held fixed.
    """
    n_tr = phase_a.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n_tr)
        null[k] = scalar_plv(phase_a[perm], phase_b)
    return null

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Compute PLV per pair (selection-aligned)
# ─────────────────────────────────────────────────────────────────────────────

print("Computing scalar PLV per pair (selection-aligned) …\n")

rng = np.random.default_rng(RNG_SEED)
records = []

for _, row in pair_df.iterrows():
    sid_a, sid_b = row["subj_A"], row["subj_B"]

    if sid_a not in subject_epochs or sid_b not in subject_epochs:
        print(f"  Pair ({sid_a},{sid_b}): missing data, skipping.")
        continue

    idx_a, idx_b, common = align_by_selection(sid_a, sid_b)
    if len(common) == 0:
        print(f"  Pair ({sid_a},{sid_b}): no shared trials after selection "
              f"intersection, skipping.")
        continue

    rec = {
        "Triad_id":      row["Triad_id"],
        "participant_A": row["participant_A"],
        "participant_B": row["participant_B"],
        "subj_A":        sid_a,
        "subj_B":        sid_b,
        "pair_label":    row["pair_label"],
        "n_aligned":     len(common),
    }

    for band_name in FREQ_BANDS:
        phase_a_full = get_phase(sid_a, band_name)
        phase_b_full = get_phase(sid_b, band_name)
        phase_a = phase_a_full[idx_a]
        phase_b = phase_b_full[idx_b]

        plv_val = scalar_plv(phase_a, phase_b)
        rec[f"plv_{band_name}"] = plv_val

        if N_PERM > 0:
            null = trial_shuffle_null(phase_a, phase_b, N_PERM, rng)
            rec[f"null_mean_{band_name}"] = null.mean()
            rec[f"null_p_{band_name}"] = float((null >= plv_val).mean())

        print(f"  Triad {row['Triad_id']} "
              f"{row['participant_A']}–{row['participant_B']} "
              f"({row['pair_label']})  {band_name}: "
              f"PLV={plv_val:.4f}  n_aligned={len(common)}")

    records.append(rec)

pair_results_df = pd.DataFrame(records)
pair_csv = os.path.join(OUTPUT_DIR, "plv_scalar_pair_results.csv")
pair_results_df.to_csv(pair_csv, index=False)
print(f"\nPer-pair results saved → {pair_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Triad-level aggregation
#
# Per triad: ONE friend PLV (the single friend-friend pair) and ONE
# averaged-non-friend PLV (mean of the two non-friend pairs A–C, B–C).
# Triads missing any of the three pair rows are dropped, so we don't
# silently compare a single non-friend pair against an aggregated one.
# ─────────────────────────────────────────────────────────────────────────────

triad_rows = []
for triad_id, grp in pair_results_df.groupby("Triad_id"):
    friend_rows = grp[grp["pair_label"] == "friend"]
    nf_rows     = grp[grp["pair_label"] == "non-friend"]

    if len(friend_rows) != 1 or len(nf_rows) != 2:
        print(f"  Triad {triad_id}: incomplete pair set "
              f"(friend={len(friend_rows)}, non-friend={len(nf_rows)}), "
              f"dropping from triad-level test.")
        continue

    row = {"Triad_id": triad_id}
    for band_name in FREQ_BANDS:
        col = f"plv_{band_name}"
        row[f"friend_{band_name}"] = friend_rows[col].iloc[0]
        row[f"nf_{band_name}"]     = nf_rows[col].mean()
    triad_rows.append(row)

triad_df = pd.DataFrame(triad_rows)
triad_csv = os.path.join(OUTPUT_DIR, "plv_scalar_triad_results.csv")
triad_df.to_csv(triad_csv, index=False)
print(f"Triad-level results saved → {triad_csv}")
print(f"  {len(triad_df)} triads with complete pair sets\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Statistics: Wilcoxon signed-rank (paired)
#
# Each triad contributes a (friend, mean-non-friend) pair — these are
# matched observations from the same triad, so the paired Wilcoxon test
# is appropriate. v1 used Mann-Whitney U on an inflated unpaired sample.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STATISTICAL TEST: Wilcoxon signed-rank  (friend vs. mean non-friend, paired)")
print("=" * 60)

for band_name in FREQ_BANDS:
    fcol = f"friend_{band_name}"
    ncol = f"nf_{band_name}"
    f_vals = triad_df[fcol].values
    n_vals = triad_df[ncol].values
    diff   = f_vals - n_vals

    W, p = scipy_stats.wilcoxon(f_vals, n_vals, alternative="two-sided")
    # Matched-pair rank-biserial r ≈ (W+ − W−) / (W+ + W−); use diffs
    n_nonzero = (diff != 0).sum()
    r_rb = (np.sign(diff).sum()) / n_nonzero if n_nonzero else 0.0

    print(f"\n{band_name.upper()} ({FREQ_BANDS[band_name][0]}–"
          f"{FREQ_BANDS[band_name][1]} Hz)  N_triads={len(triad_df)}")
    print(f"  Friend     mean={f_vals.mean():.4f}  sd={f_vals.std():.4f}")
    print(f"  Non-friend mean={n_vals.mean():.4f}  sd={n_vals.std():.4f}")
    print(f"  Diff (F−NF) mean={diff.mean():+.4f}  sd={diff.std():.4f}")
    print(f"  W={W:.1f}  p={p:.4f}  rank-biserial r={r_rb:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Paired boxplot + per-triad lines
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"friend": "firebrick", "non-friend": "steelblue"}


def plot_paired_box():
    n_bands = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        "Inter-Brain PLV: Friend vs. Non-Friend (triad-paired)\n"
        "T3P + T3Pn pooled — non-friend = mean of two non-friend pairs",
        fontsize=12, fontweight="bold",
    )

    for ax, (band_name, (fmin, fmax)) in zip(axes, FREQ_BANDS.items()):
        f_vals = triad_df[f"friend_{band_name}"].values
        n_vals = triad_df[f"nf_{band_name}"].values

        bp = ax.boxplot([f_vals, n_vals], patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"],
                                [COLORS["friend"], COLORS["non-friend"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Per-triad connecting lines
        rng_jit = np.random.default_rng(RNG_SEED)
        jx_f = 1 + rng_jit.uniform(-0.05, 0.05, size=len(f_vals))
        jx_n = 2 + rng_jit.uniform(-0.05, 0.05, size=len(n_vals))
        for i in range(len(f_vals)):
            ax.plot([jx_f[i], jx_n[i]], [f_vals[i], n_vals[i]],
                    color="grey", alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(jx_f, f_vals, color=COLORS["friend"], alpha=0.8,
                   s=30, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter(jx_n, n_vals, color=COLORS["non-friend"], alpha=0.8,
                   s=30, zorder=3, edgecolors="white", linewidths=0.4)

        W, p = scipy_stats.wilcoxon(f_vals, n_vals, alternative="two-sided")
        y_top = max(f_vals.max(), n_vals.max()) * 1.05
        ax.plot([1, 1, 2, 2], [y_top, y_top * 1.02, y_top * 1.02, y_top],
                color="black", lw=1)
        sig_str = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else f"p={p:.3f}")
        ax.text(1.5, y_top * 1.03, sig_str, ha="center", va="bottom", fontsize=11)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Friend\n(N={len(f_vals)})",
             f"Non-Friend\n(mean of 2 pairs,\nN={len(n_vals)})"],
            fontsize=9,
        )
        ax.set_ylabel("PLV (mean over channels & time)", fontsize=9)
        ax.set_title(f"{band_name.capitalize()} ({fmin}–{fmax} Hz)", fontsize=11)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "plv_boxplot_friend_vs_nonfriend.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.close(fig)


plot_paired_box()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Topomap: mean PLV per channel
#
# Uses `channel_plv` (correct reduction). Aggregation matches the
# scalar test: per triad, friend = the friend pair's per-channel PLV;
# non-friend = mean of the two non-friend pairs' per-channel PLVs.
# Then mean across triads for each group.
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_plv_by_triad(band_name: str):
    """
    Returns (friend_mean, nonfriend_mean): each shape (n_channels,).
    Triad-level aggregation, then mean across triads.
    """
    by_pair: dict[tuple[int, int], np.ndarray] = {}

    for _, row in pair_df.iterrows():
        sid_a, sid_b = row["subj_A"], row["subj_B"]
        if sid_a not in subject_epochs or sid_b not in subject_epochs:
            continue
        idx_a, idx_b, common = align_by_selection(sid_a, sid_b)
        if len(common) == 0:
            continue
        phase_a = get_phase(sid_a, band_name)[idx_a]
        phase_b = get_phase(sid_b, band_name)[idx_b]
        by_pair[(row["Triad_id"], row["participant_A"], row["participant_B"])] = (
            channel_plv(phase_a, phase_b), row["pair_label"]
        )

    friend_triads, nf_triads = [], []
    for triad_id, grp in pair_df.groupby("Triad_id"):
        friend_vals, nf_vals = [], []
        for _, r in grp.iterrows():
            key = (triad_id, r["participant_A"], r["participant_B"])
            if key not in by_pair:
                continue
            ch_vec, label = by_pair[key]
            (friend_vals if label == "friend" else nf_vals).append(ch_vec)
        if len(friend_vals) == 1 and len(nf_vals) == 2:
            friend_triads.append(friend_vals[0])
            nf_triads.append(np.mean(nf_vals, axis=0))

    f_arr  = np.mean(friend_triads, axis=0) if friend_triads else np.zeros(n_channels)
    nf_arr = np.mean(nf_triads,     axis=0) if nf_triads     else np.zeros(n_channels)
    return f_arr, nf_arr


def plot_topomaps():
    n_bands = len(FREQ_BANDS)
    fig, axes = plt.subplots(3, n_bands, figsize=(4.5 * n_bands, 11))
    if n_bands == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Inter-Brain PLV — Channel Topography (triad-paired aggregation)\n"
        "Rows: Friend | Non-Friend (mean of 2 pairs) | Difference (F − NF)",
        fontsize=11, fontweight="bold",
    )

    for col, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        print(f"  Building channel PLV for topomap: {band_name} …")
        f_ch, nf_ch = build_channel_plv_by_triad(band_name)
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
