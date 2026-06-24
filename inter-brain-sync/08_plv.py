"""
Inter-brain PLV — 2×2 condition design (v3).

Changes vs plv_condition_contrast_v2_n29.py, by review comment:

  • C1 — Filtering now uses MNE's default FIR (firwin, zero-phase) via
         mne.filter.filter_data, replacing the SciPy Butterworth+filtfilt.
         Field-standard, well-controlled passband; FILTER_ORDER is gone
         (FIR length is auto). Everything downstream (Hilbert → phase) is
         unchanged.

  • C5 — The epoch store is keyed by Exp_id (one recording session) instead
         of Subject_id. A subject who took part in two triads no longer
         collides / silently overwrites; each session stays distinct and is
         looked up per (Triad_id, Participant). A diagnostic at load time
         lists any Subject_id that spans >1 triad.

  • C6 — Full 2×2 design (group size × feedback) instead of two ad-hoc
         contrasts. Cells: T1P (solo+fb), T1Pn (solo−fb), T3P (triad+fb),
         T3Pn (triad−fb). Analyses run:
             grpsize_in_fb    T3P  − T1P     (group size | feedback on)
             grpsize_in_nofb  T3Pn − T1Pn    (group size | feedback off)
             feedback_in_solo T1P  − T1Pn    (feedback | solo)
             feedback_in_triad T3P − T3Pn    (feedback | triad)
             interaction      (T3P−T1P) − (T3Pn−T1Pn)
         The four simple-effect contrasts are matched within their own two
         cells (max N, floor √(π/4n) cancels in the difference). The
         interaction matches across all four cells (one common n) so the
         floor cancels in both inner differences AND the difference-of-
         differences; the per-cell absolute-PLV maps reuse that 4-matched set.

Design stance (carried over / made explicit):
  • Each analysis gets its own permutation null (no correction ACROSS the
    five analyses) — same stance as v2's separate RQ runs.
  • WITHIN an analysis, alpha and beta share one max-statistic null
    (block-diagonal adjacency → no cross-band clustering; joint FWER over
    the band family). This is the conservative direction (see C4 note in
    plv-v2-comments-review.md).
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy import sparse
import mne
from mne.stats import permutation_cluster_1samp_test

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
OVERVIEW_PKL = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\FG_overview_df_v2.pkl"
)
OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv_2x2"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

# ── The four 2×2 cells ───────────────────────────────────────────────────────
#                feedback        no feedback
#   solo         T1P             T1Pn
#   triad        T3P             T3Pn
CELLS = ["T1P", "T1Pn", "T3P", "T3Pn"]

# Each analysis: which cells it needs, the contrast over per-cell matrices,
# and direction labels for the sign of the summed t.
ANALYSES = [
    {"name": "grpsize_in_fb",     "cells": ["T3P", "T1P"],
     "fn": lambda M: M["T3P"] - M["T1P"],
     "pos": "T3P > T1P",   "neg": "T1P > T3P",
     "title": "Group size within feedback (T3P − T1P)"},
    {"name": "grpsize_in_nofb",   "cells": ["T3Pn", "T1Pn"],
     "fn": lambda M: M["T3Pn"] - M["T1Pn"],
     "pos": "T3Pn > T1Pn", "neg": "T1Pn > T3Pn",
     "title": "Group size within no-feedback (T3Pn − T1Pn)"},
    {"name": "feedback_in_solo",  "cells": ["T1P", "T1Pn"],
     "fn": lambda M: M["T1P"] - M["T1Pn"],
     "pos": "T1P > T1Pn",  "neg": "T1Pn > T1P",
     "title": "Feedback within solo (T1P − T1Pn)"},
    {"name": "feedback_in_triad", "cells": ["T3P", "T3Pn"],
     "fn": lambda M: M["T3P"] - M["T3Pn"],
     "pos": "T3P > T3Pn",  "neg": "T3Pn > T3P",
     "title": "Feedback within triad (T3P − T3Pn)"},
    {"name": "interaction",       "cells": ["T3P", "T1P", "T3Pn", "T1Pn"],
     "fn": lambda M: (M["T3P"] - M["T1P"]) - (M["T3Pn"] - M["T1Pn"]),
     "pos": "size effect larger WITH feedback",
     "neg": "size effect larger WITHOUT feedback",
     "title": "Group-size × feedback interaction"},
]

# ── Pre-registered analysis choices ──────────────────────────────────────────
FREQ_BANDS     = {"alpha": (8, 12), "beta": (13, 30)}   # cluster family
band_order     = list(FREQ_BANDS)
PLV_TMIN, PLV_TMAX = 0.0, 4.0            # the push phase

CLUSTER_T_THRESHOLD = 2.0                # Dumas |t|>2
CLUSTER_TAIL        = 0                  # two-sided
N_PERMUTATIONS      = 5000
RNG_SEED            = 42

MATCH_N        = True                    # equalise trial count per pair (per analysis)
N_MIN          = 25                      # min matched trials per pair per cell
EXCLUDE_TRIADS = []                      # e.g. [330]; 302 no longer needs manual drop (see C5)

TIME_CHUNK     = 500                     # matrix_plv memory knob (perf only)

rng = np.random.default_rng(RNG_SEED)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — pairs (within-triad), carrying the per-member Exp_id  (C5)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading overview dataframe (triad/subject/session structure only) …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} rows, {fg_df['Triad_id'].nunique()} triads\n")

# C5 diagnostic — does any subject span >1 triad? (the repeat-subject hazard).
_multi = fg_df.groupby("Subject_id")["Triad_id"].nunique()
_repeat = _multi[_multi > 1]
if len(_repeat):
    print("NOTE (C5): Subject_id(s) appearing in >1 triad — handled, because the")
    print("           epoch store is keyed by Exp_id so the two sessions stay split:")
    for sid in _repeat.index:
        tris = sorted(fg_df.loc[fg_df["Subject_id"] == sid, "Triad_id"].unique())
        print(f"             Subject_id {sid}: triads {tris}")
    print()
else:
    print("  No Subject_id appears in >1 triad (repeat-subject hazard absent).\n")


def build_pairs(fg_df: pd.DataFrame) -> pd.DataFrame:
    """One row per within-triad pair (A-B, A-C, B-C). Carries each member's
    Exp_id (the session key used for epoch lookup, C5) and Subject_id (logging
    only). Structure only — no friendship column is read."""
    rows = []
    for triad_id, grp in fg_df.groupby("Triad_id"):
        g = grp.set_index("Participant")
        for p_a, p_b in itertools.combinations(sorted(g.index), 2):
            rows.append({
                "Triad_id": triad_id,
                "participant_A": p_a, "participant_B": p_b,
                "exp_A": g.loc[p_a, "Exp_id"], "exp_B": g.loc[p_b, "Exp_id"],
                "subj_A": int(g.loc[p_a, "Subject_id"]),
                "subj_B": int(g.loc[p_b, "Subject_id"]),
            })
    return pd.DataFrame(rows)


pair_df = build_pairs(fg_df)
print(f"{len(pair_df)} within-triad pairs across "
      f"{pair_df['Triad_id'].nunique()} triads\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — load epochs, keyed by Exp_id, keeping every available cell  (C5)
# ═════════════════════════════════════════════════════════════════════════════
print(f"Found {len(EPOCH_FILES)} epoch files; keeping cells {CELLS} where present.")

session_epochs: dict[str, dict[str, mne.Epochs]] = {}   # exp_id -> {cell: Epochs}
session_subj:   dict[str, int] = {}                      # exp_id -> Subject_id (logging)
info_ref = None

for fpath in EPOCH_FILES:
    exp_id = os.path.basename(fpath).split("_")[0]
    match = fg_df[fg_df["Exp_id"] == exp_id]
    if match.empty:
        print(f"  WARNING: no metadata for {exp_id}, skipping.")
        continue
    subj_id = int(match["Subject_id"].iloc[0])
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    present = {c: epochs[c] for c in CELLS if c in epochs.event_id}
    if not present:
        print(f"  {exp_id} (subj={subj_id}): none of {CELLS} present, skipping.")
        continue
    if exp_id in session_epochs:                          # should never happen
        print(f"  WARNING: Exp_id {exp_id} already loaded; overwriting. Investigate.")
    session_epochs[exp_id] = present
    session_subj[exp_id]   = subj_id
    if info_ref is None:
        info_ref = next(iter(present.values())).info.copy()
    print(f"  {exp_id} (subj={subj_id}): "
          + ", ".join(f"{c}={len(present[c])}" for c in present))

print(f"\n{len(session_epochs)} sessions loaded\n")

sfreq      = info_ref["sfreq"]
times_full = next(iter(next(iter(session_epochs.values())).values())).times
t_mask     = (times_full >= PLV_TMIN) & (times_full <= PLV_TMAX)
ch_names   = info_ref["ch_names"]
n_channels = len(ch_names)
n_pairs    = n_channels * n_channels

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — phase extraction (cached) — MNE default FIR (firwin)  (C1)
# ═════════════════════════════════════════════════════════════════════════════
_phase_cache: dict[tuple, np.ndarray] = {}


def get_phase(exp_id: str, cond: str, band: str) -> np.ndarray:
    """(n_trials, n_channels, n_times_in_window) for one session × cell × band.
    Band-pass the full epoch with MNE's default zero-phase FIR (firwin),
    Hilbert, then crop to the push window."""
    key = (exp_id, cond, band)
    if key in _phase_cache:
        return _phase_cache[key]
    fmin, fmax = FREQ_BANDS[band]
    data_full = session_epochs[exp_id][cond].get_data()          # (n_tr, n_ch, n_t)
    data_filt = mne.filter.filter_data(
        data_full, sfreq, l_freq=fmin, h_freq=fmax,
        method="fir", phase="zero", fir_design="firwin",
        l_trans_bandwidth="auto", h_trans_bandwidth="auto",
        filter_length="auto", verbose=False,
    )
    analytic = hilbert(data_filt, axis=-1)
    phase = np.angle(analytic[:, :, t_mask])
    _phase_cache[key] = phase
    return phase


def aligned_idx(exp_a: str, exp_b: str, cond: str):
    """idx_a, idx_b, n — trials of the two sessions that share an original event
    index WITHIN this cell (intersect epochs.selection). Selection-based, so
    band-independent."""
    sel_a = session_epochs[exp_a][cond].selection
    sel_b = session_epochs[exp_b][cond].selection
    common = np.intersect1d(sel_a, sel_b)
    return (np.searchsorted(sel_a, common),
            np.searchsorted(sel_b, common), len(common))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — full cross-brain PLV matrix
# ═════════════════════════════════════════════════════════════════════════════
def matrix_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    assert phase_a.shape == phase_b.shape, "Phase arrays must be aligned."
    n_tr, n_ch, n_t = phase_a.shape
    za_t = np.exp(1j * phase_a).transpose(2, 1, 0)           # (n_t, n_ch, n_tr)
    zb_t = np.conj(np.exp(1j * phase_b).transpose(2, 0, 1))  # (n_t, n_tr, n_ch)
    acc = np.zeros((n_ch, n_ch))
    for s in range(0, n_t, TIME_CHUNK):
        e = min(s + TIME_CHUNK, n_t)
        cross = np.matmul(za_t[s:e], zb_t[s:e]) / n_tr       # (chunk, n_ch, n_ch)
        acc += np.abs(cross).sum(axis=0)
    return acc / n_t


def subsample(idx_a, idx_b, n):
    """Randomly keep n of the aligned trials (counts-matching for the floor)."""
    if len(idx_a) <= n:
        return idx_a, idx_b
    keep = np.sort(rng.permutation(len(idx_a))[:n])
    return idx_a[keep], idx_b[keep]

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — per-cell PLV per triad, matched over the cells an analysis needs
# ═════════════════════════════════════════════════════════════════════════════
def compute_cells(cells):
    """Per band → per triad → per cell, the mean-over-3-pairs cross-brain PLV
    matrix, computed on a MATCHED trial count (min aligned n across `cells`,
    per pair). One subsample per pair×cell, reused across bands so alpha and
    beta see the same trials. Returns (triads, cell_arr, matched_counts) where
    cell_arr[band][cell] is (n_triads, n_ch, n_ch)."""
    store = {b: {} for b in FREQ_BANDS}          # band -> tid -> cell -> [per-pair M]
    counts = []
    band0 = band_order[0]
    for _, row in pair_df.iterrows():
        tid = row["Triad_id"]
        if tid in EXCLUDE_TRIADS:
            continue
        ea, eb = row["exp_A"], row["exp_B"]
        if ea not in session_epochs or eb not in session_epochs:
            continue
        if not all(c in session_epochs[ea] and c in session_epochs[eb] for c in cells):
            continue
        # aligned indices per cell (band-independent), then a common matched n
        idxs, ns = {}, {}
        for c in cells:
            ia, ib, n = aligned_idx(ea, eb, c)
            idxs[c], ns[c] = (ia, ib), n
        n = min(ns.values())
        if n < N_MIN:
            continue
        sub = ({c: subsample(*idxs[c], n) for c in cells} if MATCH_N
               else dict(idxs))
        for band in FREQ_BANDS:
            for c in cells:
                ia, ib = sub[c]
                M = matrix_plv(get_phase(ea, c, band)[ia],
                               get_phase(eb, c, band)[ib])
                store[band].setdefault(tid, {}).setdefault(c, []).append(M)
        counts.append(n)

    triads = sorted(
        tid for tid in store[band0]
        if all(len(store[b][tid].get(c, [])) == 3
               for b in FREQ_BANDS for c in cells))
    cell_arr = {
        b: {c: np.stack([np.mean(store[b][tid][c], axis=0) for tid in triads])
            for c in cells}
        for b in FREQ_BANDS}
    return triads, cell_arr, counts

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — inter-brain adjacency  =  (A_chan ⊗ A_chan) per band, block-diagonal
# ═════════════════════════════════════════════════════════════════════════════
def build_single_head_adjacency(info) -> sparse.csr_matrix:
    A, names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    assert names == info["ch_names"], "adjacency channel order mismatch"
    A = (A + sparse.eye(A.shape[0])) > 0
    return A.tocsr()


def build_pair_adjacency(A_chan: sparse.csr_matrix) -> sparse.csr_matrix:
    return (sparse.kron(A_chan, A_chan, format="csr") > 0).tocsr()


A_chan    = build_single_head_adjacency(info_ref)
pair_adj  = build_pair_adjacency(A_chan)
adjacency = sparse.block_diag([pair_adj] * len(FREQ_BANDS), format="csr")
print(f"Adjacency: {n_channels} ch → {pair_adj.shape[0]} pairs/band, "
      f"{adjacency.shape[0]} band×pair nodes "
      f"({len(FREQ_BANDS)} disjoint band blocks).\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — figures (parametrised per analysis)
# ═════════════════════════════════════════════════════════════════════════════
def significant_mask(clusters, sig, band_idx) -> np.ndarray:
    m = np.zeros(n_pairs, bool)
    for ci in sig:
        m |= clusters[ci][band_idx]
    return m.reshape(n_channels, n_channels)


def plot_tmaps(T_map, clusters, sig, triads, spec, subdir):
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, nb, figsize=(6 * nb, 5.5))
    axes = np.atleast_1d(axes)
    vmax = max(abs(T_map).max(), 1e-9)
    for col, band in enumerate(band_order):
        ax = axes[col]
        im = ax.imshow(T_map[col], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="upper")
        msk = significant_mask(clusters, sig, col)
        if msk.any():
            yy, xx = np.where(msk)
            ax.scatter(xx, yy, s=4, marker="s", facecolors="none",
                       edgecolors="k", linewidths=0.3)
        ax.set_title(f"{band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)\n"
                     f"t; black = sig cluster", fontsize=10)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    fig.suptitle(f"{spec['title']} · t-map · N={len(triads)} triads",
                 fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(subdir, "tmap.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {fn}")


def plot_diffmap(diff_by_band, triads, spec, subdir):
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, nb, figsize=(6 * nb, 5.5))
    axes = np.atleast_1d(axes)
    mean_diff = {b: diff_by_band[b].mean(axis=0) for b in band_order}
    vmax = max(max(abs(m).max() for m in mean_diff.values()), 1e-9)
    for col, band in enumerate(band_order):
        ax = axes[col]
        im = ax.imshow(mean_diff[band], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="upper")
        ax.set_title(f"{band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)\n"
                     f"mean Δ; max|Δ|={abs(mean_diff[band]).max():.3f}", fontsize=10)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, label="ΔPLV")
    fig.suptitle(f"{spec['title']} · effect size in PLV units · "
                 f"N={len(triads)} triads", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(subdir, "diffmap.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {fn}")


def plot_participation(clusters, sig, spec, subdir):
    if not sig:
        print("    participation topomap skipped (no significant clusters).")
        return
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(2, nb, figsize=(4.5 * nb, 8))
    axes = axes.reshape(2, nb)
    for col, band in enumerate(band_order):
        msk = significant_mask(clusters, sig, col)
        for r, (deg, side) in enumerate([(msk.sum(axis=1).astype(float), "head A"),
                                         (msk.sum(axis=0).astype(float), "head B")]):
            ax = axes[r, col]
            mne.viz.plot_topomap(deg, info_ref, axes=ax, show=False,
                                 cmap="Reds", vlim=(0, max(deg.max(), 1e-9)))
            ax.set_title(f"{band} · {side}\n# sig pairs / channel", fontsize=9)
    fig.suptitle(f"{spec['title']} · where significant pairs concentrate",
                 fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(subdir, "participation.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {fn}")


def plot_null_histogram(H0, T_obs, clusters, cluster_pv, triads, spec, subdir):
    if H0 is None or len(H0) == 0:
        print("    null histogram skipped (no H0).")
        return
    null = np.abs(H0)
    obs_stats = np.array([abs(float(T_obs[clusters[ci]].sum()))
                          for ci in range(len(clusters))]) if clusters else np.array([0.0])
    obs_max = float(obs_stats.max())
    p_obs   = float(min(cluster_pv)) if len(cluster_pv) else 1.0
    p95     = float(np.percentile(null, 95))
    pct     = float((null < obs_max).mean() * 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null, bins=60, color="0.82", edgecolor="0.55", linewidth=0.3)
    ax.axvline(p95, color="0.30", linestyle="--", linewidth=1.2,
               label=f"95th percentile = {p95:.1f}")
    ax.axvline(obs_max, color="crimson", linewidth=2.2,
               label=f"observed max = {obs_max:.1f}  (p = {p_obs:.3f}, ~{pct:.0f}th pct)")
    ax.set_xlabel("max cluster mass  |Σ t|  per permutation")
    ax.set_ylabel("permutations")
    ax.set_title(f"{spec['title']} · permutation null vs observed\n"
                 f"N={len(triads)} triads · {len(H0)} perms · "
                 f"two-sided max-stat over alpha+beta", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    plt.tight_layout()
    fn = os.path.join(subdir, "null_histogram.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {fn}")


def plot_cell_abs(cell_arr, triads, counts, subdir):
    """Descriptive absolute PLV for all four cells, on the 4-matched set, vs the
    across-trial chance floor √(π/4N). No inference — coupling level only."""
    n_med = int(np.median(counts)) if counts else N_MIN
    floor = float(np.sqrt(np.pi / (4 * n_med)))
    gm = {b: {c: cell_arr[b][c].mean(axis=0) for c in CELLS} for b in band_order}
    vmax = max(m.max() for b in band_order for m in gm[b].values())
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(len(CELLS), nb, figsize=(5 * nb, 4.2 * len(CELLS)))
    axes = np.atleast_2d(axes).reshape(len(CELLS), nb)
    for r, cell in enumerate(CELLS):
        for col, band in enumerate(band_order):
            ax = axes[r, col]
            m = gm[band][cell]
            im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax,
                           aspect="equal", origin="upper")
            ax.set_title(f"{cell} · {band} "
                         f"({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)\n"
                         f"mean={m.mean():.3f} (floor≈{floor:.3f})", fontsize=9)
            ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, label="PLV")
    fig.suptitle(f"Absolute inter-brain PLV per cell · floor √(π/4N) ≈ {floor:.3f} "
                 f"at N≈{n_med} · descriptive, not inferential",
                 fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(subdir, "cell_abs_plv.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    saved {fn}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — run each analysis
# ═════════════════════════════════════════════════════════════════════════════
def run_analysis(spec):
    print("=" * 70)
    print(f"ANALYSIS: {spec['name']}  —  {spec['title']}")
    print("=" * 70)
    triads, cell_arr, counts = compute_cells(spec["cells"])
    if not triads:
        print("  No triads with all 3 pairs in all cells/bands — skipped.\n")
        return None
    subdir = os.path.join(OUTPUT_DIR, spec["name"])
    os.makedirs(subdir, exist_ok=True)
    print(f"  {len(triads)} triads; matched n per pair: "
          f"min={min(counts)}, median={int(np.median(counts))}, max={max(counts)}")

    diff_by_band = {b: spec["fn"](cell_arr[b]) for b in FREQ_BANDS}
    for b in FREQ_BANDS:
        np.save(os.path.join(subdir, f"diff_{b}.npy"), diff_by_band[b])

    X = np.stack([diff_by_band[b].reshape(len(triads), -1) for b in band_order], axis=1)
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X, threshold=CLUSTER_T_THRESHOLD, n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL, adjacency=adjacency, out_type="mask",
        seed=RNG_SEED, n_jobs=1, verbose=False,
    )
    T_map = T_obs.reshape(len(FREQ_BANDS), n_channels, n_channels)
    sig = [ci for ci, p in enumerate(cluster_pv) if p < 0.05]
    print(f"  {len(clusters)} candidate clusters; {len(sig)} significant at p<0.05.")

    cluster_rows = []
    for ci in np.argsort(cluster_pv):
        mask = clusters[ci]
        bands_in, pidx = np.where(mask)
        ii, kk = pidx // n_channels, pidx % n_channels
        sum_t = float(T_obs[mask].sum())
        band_tag = "/".join(sorted({band_order[b] for b in np.unique(bands_in)}))
        direction = spec["pos"] if sum_t > 0 else spec["neg"]
        cluster_rows.append({
            "cluster": ci, "p_value": round(float(cluster_pv[ci]), 4),
            "n_pairs": int(mask.sum()), "sum_t": round(sum_t, 2),
            "bands": band_tag, "direction": direction,
            "example_pairs": "; ".join(
                f"{ch_names[i]}~{ch_names[k]}" for i, k in list(zip(ii, kk))[:8]),
        })
        if cluster_pv[ci] < 0.05:
            print(f"    cluster {ci}: p={cluster_pv[ci]:.4f} bands={band_tag} "
                  f"n_pairs={int(mask.sum())} sumT={sum_t:+.1f} ({direction})")
    if not sig:
        print("    No significant clusters at p<0.05.")
    pd.DataFrame(cluster_rows).to_csv(
        os.path.join(subdir, "cluster_results.csv"), index=False)
    np.save(os.path.join(subdir, "H0.npy"), H0)

    plot_tmaps(T_map, clusters, sig, triads, spec, subdir)
    plot_diffmap(diff_by_band, triads, spec, subdir)
    plot_participation(clusters, sig, spec, subdir)
    plot_null_histogram(H0, T_obs, clusters, cluster_pv, triads, spec, subdir)
    if spec["name"] == "interaction":                      # all 4 cells matched here
        plot_cell_abs(cell_arr, triads, counts, subdir)
    print()
    return {"spec": spec, "triads": triads, "counts": counts,
            "cell_arr": cell_arr, "diff_by_band": diff_by_band,
            "cluster_pv": cluster_pv, "sig": sig}


results = {a["name"]: run_analysis(a) for a in ANALYSES}

print("=" * 70)
print("2×2 SUMMARY")
print("=" * 70)
for name, r in results.items():
    if r is None:
        print(f"  {name:18s}: skipped (no triads)")
        continue
    pmin = min(r["cluster_pv"]) if len(r["cluster_pv"]) else float("nan")
    print(f"  {name:18s}: N={len(r['triads']):2d}  "
          f"{len(r['sig'])} sig cluster(s)  min p={pmin:.3f}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — verification
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70 + "\nVERIFICATION\n" + "=" * 70)

_exp  = next(iter(session_epochs))
_cell = next(iter(session_epochs[_exp]))
_p = get_phase(_exp, _cell, band_order[0])
_self = matrix_plv(_p, _p)
print(f"  (V1) PLV(x,x) diag mean = {np.diag(_self).mean():.6f} (expect 1.0) … "
      f"{'PASS' if np.allclose(np.diag(_self), 1.0, atol=1e-6) else 'FAIL'}")

toy = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
toy_pair = (sparse.kron(toy, toy, format="csr") > 0).tocsr()
deg00 = int(toy_pair[0].toarray().sum())
print(f"  (V2) toy a/b/c: pair(0,0) degree = {deg00} (expect 4) … "
      f"{'PASS' if deg00 == 4 else 'FAIL'}")

sym = (pair_adj != pair_adj.T).nnz == 0
diag_ok = bool((pair_adj.diagonal() > 0).all())
print(f"  (V3) pair adjacency symmetric={sym}, self-inclusive={diag_ok} … "
      f"{'PASS' if sym and diag_ok else 'FAIL'}")


class _SelOnly:
    def __init__(self, sel): self.selection = np.asarray(sel)


session_epochs["__TEST_A__"] = {"T3P": _SelOnly([0, 1, 2, 5, 9]),
                                "T1P": _SelOnly([0, 1, 2, 5, 9])}
session_epochs["__TEST_B__"] = {"T3P": _SelOnly([1, 2, 3, 5, 8, 9]),
                                "T1P": _SelOnly([100, 101, 102])}
_, _, n_share = aligned_idx("__TEST_A__", "__TEST_B__", "T3P")
_, _, n_disjoint = aligned_idx("__TEST_A__", "__TEST_B__", "T1P")
del session_epochs["__TEST_A__"], session_epochs["__TEST_B__"]
print(f"  (V4) trial-alignment: shared={n_share} (expect 4), "
      f"disjoint={n_disjoint} (expect 0) … "
      f"{'PASS' if n_share == 4 and n_disjoint == 0 else 'FAIL'}")

# (V5) matched-N across all analyses
_allc = [c for r in results.values() if r for c in r["counts"]]
_ok = MATCH_N and len(_allc) > 0 and all(c >= N_MIN for c in _allc)
print(f"  (V5) matched-N: {len(_allc)} pair-instances, all ≥ {N_MIN} "
      f"(min={min(_allc) if _allc else 'NA'}) … "
      f"{'PASS' if _ok else 'FAIL' if MATCH_N else 'SKIPPED (MATCH_N=False)'}")

# (V6) C5 sanity: no Subject_id is silently dropped by a key collision —
# every loaded session has a distinct Exp_id key.
print(f"  (V6) session keys: {len(session_epochs)} sessions, "
      f"{len(set(session_epochs))} unique Exp_id keys … "
      f"{'PASS' if len(session_epochs) == len(set(session_epochs)) else 'FAIL'}")

# (V7) contrast linearity on a 2-cell analysis: mean(fn) == fn(means)
_r = next((results[a["name"]] for a in ANALYSES if len(a["cells"]) == 2
           and results[a["name"]]), None)
if _r is not None:
    b0 = band_order[0]
    spec = _r["spec"]
    lhs = _r["diff_by_band"][b0].mean(axis=0)
    rhs = spec["fn"]({c: _r["cell_arr"][b0][c].mean(axis=0) for c in spec["cells"]})
    dev = float(np.abs(lhs - rhs).max())
    print(f"  (V7) contrast/abs consistency ({spec['name']}): max|Δ|={dev:.2e} "
          f"(expect ~0) … {'PASS' if dev < 1e-9 else 'FAIL'}")
else:
    print("  (V7) SKIPPED (no 2-cell analysis produced triads).")

print("\nDone. Outputs under:", OUTPUT_DIR)
print("  one subfolder per analysis, each with:")
print("    cluster_results.csv · tmap.png · diffmap.png · participation.png")
print("    null_histogram.png · H0.npy · diff_<band>.npy")
print("  interaction/ also has cell_abs_plv.png (all four cells vs floor)")
