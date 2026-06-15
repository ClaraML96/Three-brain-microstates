"""
hypyp_overtime_plv.py — complementary LO4 synchrony analysis: HyPyP over-time PLV
═══════════════════════════════════════════════════════════════════════════════
This is a SECOND inter-brain-synchronisation result for LO4, computed with
HyPyP's PLV (the library the supervisor named). It is NOT a cross-validation of
`plv_matrix_pipeline.py` — HyPyP computes a DIFFERENT estimator — it is a
*complementary* analysis with a different but legitimate measure. If it ALSO
returns a friend-vs-non-friend null, that is robustness across estimators
(stronger than either alone). See `../hypyp-decision.md` and the adjacency
cross-check `hypyp_adjacency_crossval.py` for why HyPyP can't reproduce the
primary numbers.

─────────────────────────────────────────────────────────────────────────────
WHAT DIFFERS FROM plv_matrix_pipeline.py — AND WHAT DELIBERATELY DOES NOT
─────────────────────────────────────────────────────────────────────────────
DIFFERENT (the one variable under test):
  • MEASURE = HyPyP over-time PLV. `compute_freq_bands` (band-pass+Hilbert) →
    `compute_sync(mode='plv')`. HyPyP computes PLV by averaging the phase
    difference over TIME SAMPLES within each trial (one PLV per trial), then
    averaging over trials (`hypyp/sync/plv.py`: con = |Σ_t z|/n_samp). The
    primary pipeline instead averages over TRIALS at each time sample, then over
    time (ACROSS-TRIALS PLV). Verified from source 2026-06-15.

IDENTICAL (so the comparison isolates the estimator):
  • Data layer — file discovery, friendship join, `selection`-based trial
    alignment, N_MIN / EXCLUDE_TRIADS. Copied verbatim from the primary pipeline.
  • Contrast — per-triad friend − mean(two non-friend) difference matrices.
  • Adjacency — self-inclusive single-head A, A⊗A inter-brain neighbourhood,
    BLOCK-DIAGONAL across the two bands (no cross-band fusion).
  • INFERENCE — our paired sign-flip `permutation_cluster_1samp_test`, two-sided,
    |t|>2, 5000 perms. NOT HyPyP's `statscluster` (which uses the wrong,
    independent-groups null for a paired design — `../hypyp-decision.md`).
  • Pre-registered choices — T3P primary, alpha-mu + beta family.

─────────────────────────────────────────────────────────────────────────────
HONEST CAVEATS
─────────────────────────────────────────────────────────────────────────────
  • ABSOLUTE LEVELS ARE NOT COMPARABLE across the two scripts. Over-time PLV on
    narrowband-filtered data sits on a different (and generally inflated) floor
    than across-trials PLV — its random-phase floor is √(π/4·n_samples), and the
    local smoothness of band-pass-filtered phase inflates single-trial PLV. Do
    NOT compare the raw PLV magnitudes here to v4's √(π/4N) baseline.
  • The CONTRAST cancels that floor. Friend and non-friend pairs share the same
    n_samples and filtering, so the per-triad difference removes the common floor
    — inference on the difference is valid even though the absolute level isn't
    interpretable the v4 way.
  • Filtering differs from the primary (HyPyP's compute_freq_bands vs our
    butter order-4). Intentional: this is HyPyP's measure end-to-end.

Runs in the HyPyP venv (requirements-hypyp.txt). Reads the same *-epo.fif.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import mne
from mne.stats import permutation_cluster_1samp_test

from hypyp.analyses import compute_freq_bands, compute_sync

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  EDIT PATHS FOR YOUR MACHINE   (same paths as the primary)
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
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv_matrix"
    r"\hypyp_overtime"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

# ── Pre-registered choices (identical to the primary pipeline) ───────────────
PLV_CONDITIONS = ["T3P"]
FREQ_BANDS     = {"alpha": (8, 12), "beta": (13, 30)}
HYPYP_BANDS    = {b: list(v) for b, v in FREQ_BANDS.items()}   # HyPyP wants lists
PLV_TMIN, PLV_TMAX = 0.0, 4.0

# ── Cluster-test parameters (identical) ──────────────────────────────────────
CLUSTER_T_THRESHOLD = 2.0
CLUSTER_TAIL        = 0
N_PERMUTATIONS      = 5000
RNG_SEED            = 42

# ── Data-quality filters (identical) ─────────────────────────────────────────
N_MIN          = 30
EXCLUDE_TRIADS = [330]

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Pair labels from metadata   (verbatim from plv_matrix_pipeline.py)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading overview dataframe …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} subjects, {fg_df['Triad_id'].nunique()} triads\n")


def build_pair_labels(fg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for triad_id, grp in fg_df.groupby("Triad_id"):
        members = grp.set_index("Participant")[["Subject_id", "Friend_status"]]
        for p_a, p_b in itertools.combinations(sorted(members.index), 2):
            st_a = members.loc[p_a, "Friend_status"]
            st_b = members.loc[p_b, "Friend_status"]
            rows.append({
                "Triad_id":      triad_id,
                "participant_A": p_a, "participant_B": p_b,
                "subj_A": members.loc[p_a, "Subject_id"],
                "subj_B": members.loc[p_b, "Subject_id"],
                "status_A": st_a, "status_B": st_b,
                "pair_label": "friend" if (st_a == "Yes" and st_b == "Yes")
                              else "non-friend",
            })
    return pd.DataFrame(rows)


pair_df = build_pair_labels(fg_df)
print("Pair label distribution:")
print(pair_df["pair_label"].value_counts().to_string(), "\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load epochs   (verbatim)
# ═════════════════════════════════════════════════════════════════════════════
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
        continue
    epochs_sel = mne.concatenate_epochs([epochs[c] for c in available]) \
        if len(available) > 1 else epochs[available[0]]
    if subj_id in subject_epochs:
        print(f"  WARNING: Subject_id {subj_id} already loaded "
              f"(exp_id={exp_id}); overwriting. Investigate.")
    subject_epochs[subj_id] = epochs_sel
    if info_ref is None:
        info_ref = epochs_sel.info.copy()
    print(f"  {exp_id} (id={subj_id}): {len(epochs_sel)} trials, "
          f"selection [{epochs_sel.selection.min()}, {epochs_sel.selection.max()}]")

print(f"\n{len(subject_epochs)} subjects loaded for conditions {PLV_CONDITIONS}\n")

sfreq      = info_ref["sfreq"]
times_full = subject_epochs[next(iter(subject_epochs))].times
t_mask     = (times_full >= PLV_TMIN) & (times_full <= PLV_TMAX)
ch_names   = info_ref["ch_names"]
n_channels = len(ch_names)
band_order = list(HYPYP_BANDS)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Trial alignment   (verbatim)
# ═════════════════════════════════════════════════════════════════════════════
def align_by_selection(subj_a: int, subj_b: int):
    sel_a = subject_epochs[subj_a].selection
    sel_b = subject_epochs[subj_b].selection
    common = np.intersect1d(sel_a, sel_b)
    if len(common) == 0:
        return np.array([], int), np.array([], int), common
    return np.searchsorted(sel_a, common), np.searchsorted(sel_b, common), common

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — HyPyP over-time PLV matrices   (THE measure swap)
#
# Filter the FULL epoch then crop the analytic signal to the push window (matches
# the primary's "filter full, then crop" — avoids window-edge artefacts), then
# compute_sync averages over trials. The inter-brain block of the 2n×2n output
# (rows = head A, cols = head B) is our pair matrix, per band.
# ═════════════════════════════════════════════════════════════════════════════
def hypyp_pair_matrices(data_a: np.ndarray, data_b: np.ndarray) -> dict:
    """data_* : (n_trials_aligned, n_ch, n_times_full). Returns band -> (n_ch, n_ch)."""
    data = np.stack([data_a, data_b])                       # (2, n_tr, n_ch, n_t_full)
    complex_signal = compute_freq_bands(data, sfreq, HYPYP_BANDS)
    complex_win    = complex_signal[..., t_mask]            # crop time to the window
    result = compute_sync(complex_win, mode="plv", epochs_average=True)  # (n_freq,2n,2n)
    return {band: result[fi, :n_channels, n_channels:]      # head-A rows, head-B cols
            for fi, band in enumerate(band_order)}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Per-triad friend / averaged-non-friend matrices   (same as primary)
# ═════════════════════════════════════════════════════════════════════════════
print("Computing HyPyP over-time PLV matrices per pair …\n")
mats: dict[str, dict[int, dict[str, list]]] = {b: {} for b in FREQ_BANDS}

for _, row in pair_df.iterrows():
    tid = row["Triad_id"]
    sid_a, sid_b = row["subj_A"], row["subj_B"]
    if tid in EXCLUDE_TRIADS:
        continue
    if sid_a not in subject_epochs or sid_b not in subject_epochs:
        continue
    idx_a, idx_b, common = align_by_selection(sid_a, sid_b)
    if len(common) < N_MIN:
        print(f"  Triad {tid} {row['participant_A']}–{row['participant_B']}: "
              f"n_aligned={len(common)} < {N_MIN}, skipping.")
        continue

    transpose = (row["pair_label"] == "non-friend" and row["status_A"] == "No")
    data_a = subject_epochs[sid_a].get_data()[idx_a]
    data_b = subject_epochs[sid_b].get_data()[idx_b]
    Mbands = hypyp_pair_matrices(data_a, data_b)
    for band in FREQ_BANDS:
        M = Mbands[band].T if transpose else Mbands[band]
        slot = mats[band].setdefault(tid, {"friend": [], "nonfriend": []})
        slot["friend" if row["pair_label"] == "friend" else "nonfriend"].append(M)
    print(f"  Triad {tid} {row['participant_A']}–{row['participant_B']} "
          f"({row['pair_label']}): n_aligned={len(common)}")

triad_ids = sorted(
    tid for tid in mats[next(iter(FREQ_BANDS))]
    if all(len(mats[b][tid]["friend"]) == 1 and len(mats[b][tid]["nonfriend"]) == 2
           for b in FREQ_BANDS)
)
print(f"\n{len(triad_ids)} triads with complete pair sets (both bands)\n")

diff_by_band, friend_mean, nf_mean = {}, {}, {}
for band in FREQ_BANDS:
    D = np.stack([
        mats[band][tid]["friend"][0] - np.mean(mats[band][tid]["nonfriend"], axis=0)
        for tid in triad_ids
    ])
    diff_by_band[band] = D
    friend_mean[band] = np.mean([mats[band][tid]["friend"][0] for tid in triad_ids], axis=0)
    nf_mean[band]     = np.mean([np.mean(mats[band][tid]["nonfriend"], axis=0)
                                 for tid in triad_ids], axis=0)
    np.save(os.path.join(OUTPUT_DIR, f"plv_overtime_diff_{band}.npy"), D)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Adjacency = A⊗A, block-diagonal across bands   (verbatim)
# ═════════════════════════════════════════════════════════════════════════════
def build_single_head_adjacency(info) -> sparse.csr_matrix:
    A, names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    assert names == info["ch_names"], "adjacency channel order mismatch"
    A = (A + sparse.eye(A.shape[0])) > 0
    return A.tocsr()


def build_pair_adjacency(A_chan: sparse.csr_matrix) -> sparse.csr_matrix:
    return (sparse.kron(A_chan, A_chan, format="csr") > 0).tocsr()


A_chan   = build_single_head_adjacency(info_ref)
pair_adj = build_pair_adjacency(A_chan)
adjacency = sparse.block_diag([pair_adj] * len(FREQ_BANDS), format="csr")
print(f"Adjacency: {n_channels} ch → {pair_adj.shape[0]} pairs/band, "
      f"{adjacency.shape[0]} band×pair nodes ({len(FREQ_BANDS)} disjoint blocks).\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Cluster test: OUR paired sign-flip (NOT HyPyP statscluster)   (verbatim)
# ═════════════════════════════════════════════════════════════════════════════
X = np.stack([diff_by_band[b].reshape(len(triad_ids), -1) for b in band_order], axis=1)
print(f"Cluster test input X: {X.shape}  (n_triads, n_bands, n_pairs)")
print(f"  threshold |t|>{CLUSTER_T_THRESHOLD}, tail={CLUSTER_TAIL}, "
      f"{N_PERMUTATIONS} permutations (sign-flip) …\n")

T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
    X, threshold=CLUSTER_T_THRESHOLD, n_permutations=N_PERMUTATIONS,
    tail=CLUSTER_TAIL, adjacency=adjacency, out_type="mask",
    seed=RNG_SEED, n_jobs=1, verbose=False,
)
n_pairs = n_channels * n_channels
T_map = T_obs.reshape(len(FREQ_BANDS), n_channels, n_channels)
sig = [ci for ci, p in enumerate(cluster_pv) if p < 0.05]
print("=" * 70)
print(f"HyPyP OVER-TIME PLV — CLUSTER RESULTS — {PLV_CONDITIONS}, {len(triad_ids)} triads")
print("=" * 70)
print(f"{len(clusters)} candidate clusters; {len(sig)} significant at p<0.05.\n")

cluster_rows = []
for ci in np.argsort(cluster_pv):
    mask = clusters[ci]
    bands_in, pidx = np.where(mask)
    ii, kk   = pidx // n_channels, pidx % n_channels
    sum_t    = float(T_obs[mask].sum())
    band_tag = "/".join(sorted({band_order[b] for b in np.unique(bands_in)}))
    cluster_rows.append({
        "cluster": ci, "p_value": round(float(cluster_pv[ci]), 4),
        "n_pairs": int(mask.sum()), "sum_t": round(sum_t, 2),
        "bands": band_tag, "direction": "F>NF" if sum_t > 0 else "NF>F",
        "example_pairs": "; ".join(
            f"{ch_names[i]}~{ch_names[k]}" for i, k in list(zip(ii, kk))[:8]),
    })
    if cluster_pv[ci] < 0.05:
        print(f"  cluster {ci}: p={cluster_pv[ci]:.4f}  bands={band_tag}  "
              f"n_pairs={int(mask.sum())}  sumT={sum_t:+.1f}  "
              f"({'F>NF' if sum_t > 0 else 'NF>F'})")

if not sig:
    smallest = float(np.min(cluster_pv)) if len(cluster_pv) else float("nan")
    print(f"  No significant clusters (smallest p={smallest:.3f}). If the primary"
          f" across-trials run was also null, this is robustness across estimators.")

pd.DataFrame(cluster_rows).to_csv(
    os.path.join(OUTPUT_DIR, "plv_overtime_cluster_results.csv"), index=False)
print(f"\nCluster table → {os.path.join(OUTPUT_DIR, 'plv_overtime_cluster_results.csv')}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Figures (t-map + participation), same as primary
# ═════════════════════════════════════════════════════════════════════════════
def significant_mask(band_idx: int) -> np.ndarray:
    m = np.zeros(n_pairs, bool)
    for ci in sig:
        m |= clusters[ci][band_idx]
    return m.reshape(n_channels, n_channels)


def plot_tmaps():
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, nb, figsize=(6 * nb, 5.5))
    axes = np.atleast_1d(axes)
    vmax = max(abs(T_map).max(), 1e-9)
    for col, band in enumerate(band_order):
        ax = axes[col]
        im = ax.imshow(T_map[col], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="upper")
        msk = significant_mask(col)
        if msk.any():
            yy, xx = np.where(msk)
            ax.scatter(xx, yy, s=4, marker="s", facecolors="none",
                       edgecolors="k", linewidths=0.3)
        ax.set_title(f"{band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)\n"
                     f"t(F−NF), over-time PLV; black = sig cluster", fontsize=10)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    fig.suptitle(f"HyPyP over-time inter-brain PLV t-map (friend − non-friend) · "
                 f"{PLV_CONDITIONS} · N={len(triad_ids)} triads", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "plv_overtime_tmap.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


def plot_participation():
    if not sig:
        print("Participation topomap skipped (no significant clusters).")
        return
    nb = len(FREQ_BANDS)
    fig, axes = plt.subplots(2, nb, figsize=(4.5 * nb, 8))
    axes = axes.reshape(2, nb)
    for col, band in enumerate(band_order):
        msk = significant_mask(col)
        deg_a = msk.sum(axis=1).astype(float)
        deg_b = msk.sum(axis=0).astype(float)
        for r, (deg, side) in enumerate([(deg_a, "head A"), (deg_b, "head B")]):
            ax = axes[r, col]
            vmax = max(deg.max(), 1e-9)
            mne.viz.plot_topomap(deg, info_ref, axes=ax, show=False,
                                 cmap="Reds", vlim=(0, vmax))
            ax.set_title(f"{band} · {side}\n# sig pairs / channel", fontsize=9)
    fig.suptitle(f"HyPyP over-time PLV — where sig friend≠non-friend pairs "
                 f"concentrate · {PLV_CONDITIONS}", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "plv_overtime_participation.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


plot_tmaps()
plot_participation()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Verification
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nVERIFICATION\n" + "=" * 70)

# PLV(x, x): feed one subject as BOTH brains; the inter-brain diagonal (a channel
# vs itself) has zero phase difference at every sample → over-time PLV = 1.
_subj = next(iter(subject_epochs))
_d = subject_epochs[_subj].get_data()
_self = hypyp_pair_matrices(_d, _d)[band_order[0]]
_diag_ok = np.allclose(np.diag(_self), 1.0, atol=1e-6)
print(f"  over-time PLV(x,x) diagonal mean = {np.diag(_self).mean():.6f} "
      f"(expect 1.0) … {'PASS' if _diag_ok else 'FAIL'}")

# Adjacency: A⊗A symmetric + self-inclusive (same checks as primary).
sym = (pair_adj != pair_adj.T).nnz == 0
diag_ok = (pair_adj.diagonal() > 0).all()
print(f"  pair adjacency symmetric={sym}, self-inclusive={bool(diag_ok)} … "
      f"{'PASS' if sym and diag_ok else 'FAIL'}")

print("\nDone.  Outputs in:", OUTPUT_DIR)
print("  • plv_overtime_cluster_results.csv  — any significant clusters?")
print("  • plv_overtime_tmap.png             — 64×64 t-map per band")
print("  • plv_overtime_participation.png    — where sig pairs concentrate")
print("  • plv_overtime_diff_<band>.npy      — per-triad difference matrices")
