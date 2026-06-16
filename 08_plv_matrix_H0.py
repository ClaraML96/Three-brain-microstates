"""
plv_matrix_pipeline.py — the matrix + cluster line (first runnable version)
═══════════════════════════════════════════════════════════════════════════════
Implements `plv-proposal-matrix.md`: the FULL cross-brain PLV matrix + the Dumas
(2010) inter-brain cluster-permutation test, for the friend / non-friend
contrast on the triadic force-game data.

This SUPERSEDES the analysis in `08_plv_v4.py` (homologous-channel + ROI
scalar). It REUSES v4's data-handling layer verbatim — file discovery,
friendship join, `selection`-based trial alignment, the N_MIN / EXCLUDE_TRIADS
filters — because that layer is correct and independent of how PLV is computed.
What changes is the measure (full 64×64 grid, not a scalar) and the statistics
(inter-brain cluster permutation, not a paired Wilcoxon on an ROI mean).

─────────────────────────────────────────────────────────────────────────────
WHY PURE MNE + NUMPY + SCIPY (not HyPyP) FOR THIS FIRST VERSION
─────────────────────────────────────────────────────────────────────────────
`hypyp-decision.md` adopts HyPyP as the production/cross-validation route.
This first *runnable* version deliberately does NOT depend on it, because:

  1. Zero new setup — runs in your existing MNE 1.6.1 environment. HyPyP needs
     a separate Python ≥3.11 venv; building that before seeing any output
     works against the goal of "run it and judge the output."
  2. The Dumas a/b/c inter-brain neighbourhood collapses to a Kronecker
     product of the single-head channel adjacency with itself (see
     build_pair_adjacency() — derivation in the docstring there). Fully
     transparent and toy-verifiable, versus a black-box call.
  3. It is the CORRECT paired null: a per-triad (friend − non-friend)
     difference fed to `permutation_cluster_1samp_test`, whose sign-flip
     permutation is exactly Maris & Oostenveld's within-subjects scheme
     (`plv-proposal-matrix.md` §6.5). HyPyP Fig. 3's "between-condition" shuffle is
     the independent-groups scheme and would be WRONG here.
  4. HyPyP remains the validation route: once its venv exists,
     `compute_sync(mode='plv')` + `metaconn_matrix_2brains` + `statscluster`
     should reproduce these numbers. An independent implementation HyPyP can
     check against is stronger than a single black-box run.

─────────────────────────────────────────────────────────────────────────────
PRE-REGISTERED ANALYSIS CHOICES (documented per user sign-off, 2026-06-10)
─────────────────────────────────────────────────────────────────────────────
  • PRIMARY CONDITION = "T3P" (continuous feedback). The only triadic
    condition with real-time mutual feedback — the substrate for
    interaction-driven inter-brain synchrony. T3Pn (feedback only at t=4 s; the
    0–4 s push is "blind") and pooled T3P+T3Pn are SECONDARY runs: change
    PLV_CONDITIONS and rerun. (`epo-fif-structure.md` §2.3.)
  • CLUSTER FAMILY = alpha-mu (8–12) + beta (13–30) ONLY. A-priori restriction
    grounded in BOTH prior papers (parent paper's alpha/beta group synchrony;
    Dumas's alpha-mu primary, beta secondary). Theta/gamma are exploratory and
    not added here. The two bands are corrected jointly (max-statistic across
    pairs × bands), so "found it in some band" cannot leak through.
  • TWO-SIDED TEST (CLUSTER_TAIL = 0). The directional hypothesis is genuinely
    ambiguous: Dumas's framing implies friends-more, but the PARENT PAPER on
    THIS dataset found NON-friends synchronise MORE (stronger ERD/ISC/group
    sync). Two-sided is the honest default. (Updates the matrix proposal §1b tail=1 assumption.)
  • CONTRAST, NOT ABSOLUTE LEVEL. This tests friend vs non-friend. The cluster
    statistic operates on the per-triad DIFFERENCE, where the √(π/4N)
    random-phase floor cancels — so no per-pair null subtraction is needed for
    inference. "Is there ANY sync above chance?" is the separate question v4
    already answered (null). See `plv-proposal-matrix.md` §0.

─────────────────────────────────────────────────────────────────────────────
REFERENCES
─────────────────────────────────────────────────────────────────────────────
  Dumas et al. 2010 (PLoS ONE 5(8):e12166) — full cross-brain PLV matrix, the
      a/b/c neighbourhood (p.5), |t|>2 cluster threshold, max-stat correction.
      → project/01-literature-review/lit-dumas.md
  Maris & Oostenveld 2007 — cluster permutation + within-subjects sign-flip.
      → raw/literature/Maris_Oostenveld_2007.pdf
  Li et al. 2025 (parent paper) — dataset, conditions, bands, the non-friend>
      friend direction. → project/01-literature-review/lit-preprint.md
  Lachaux et al. 1999 — PLV definition.
  Design: project/05-inter-brain-sync/plv-proposal-matrix.md (+ hypyp-decision.md)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import sparse
import mne
from mne.stats import permutation_cluster_1samp_test

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  EDIT PATHS FOR YOUR MACHINE
# ═════════════════════════════════════════════════════════════════════════════
# NOTE: paths below are from 08_plv_v4.py (Clara's machine). Point them at your
# copy of the PreprocessedEEGData folder and FG_overview_df_v2.pkl.

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
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

# ── Pre-registered choices (see header) ──────────────────────────────────────
PLV_CONDITIONS = ["T3P"]                      # PRIMARY. Secondary: ["T3Pn"], or both.
FREQ_BANDS     = {"alpha": (8, 12), "beta": (13, 30)}   # cluster family
PLV_TMIN, PLV_TMAX = 0.0, 4.0                 # the push phase
FILTER_ORDER   = 4

# ── Cluster-test parameters ──────────────────────────────────────────────────
CLUSTER_T_THRESHOLD = 2.0     # Dumas |t|>2 cluster-forming threshold
CLUSTER_TAIL        = 0       # two-sided (see header — direction is ambiguous)
N_PERMUTATIONS      = 5000
RNG_SEED            = 42

# ── Data-quality filters (inherited from v4) ─────────────────────────────────
N_MIN          = 30           # min selection-aligned trials per pair
EXCLUDE_TRIADS = [330]        # metadata bug, masked not fixed (08_plv-review §3)

# Time-chunk for the matrix PLV einsum (memory control; pure performance knob).
TIME_CHUNK = 500

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Pair labels from metadata   (reused from v4, extended with statuses)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading overview dataframe …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} subjects, {fg_df['Triad_id'].nunique()} triads\n")


def build_pair_labels(fg_df: pd.DataFrame) -> pd.DataFrame:
    """One row per within-triad pair. Carries each member's Friend_status so we
    can orient non-friend pairs consistently (friend = rows, non-friend = cols)."""
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
print("Pair label distribution (before triad-level aggregation):")
print(pair_df["pair_label"].value_counts().to_string(), "\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load epochs   (reused from v4)
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

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Phase extraction with caching + selection alignment   (reused from v4)
# ═════════════════════════════════════════════════════════════════════════════
_phase_cache: dict[tuple[int, str], np.ndarray] = {}


def get_phase(subj_id: int, band: str) -> np.ndarray:
    """(n_trials, n_channels, n_times_in_window). Filter full epoch, then crop."""
    key = (subj_id, band)
    if key in _phase_cache:
        return _phase_cache[key]
    fmin, fmax = FREQ_BANDS[band]
    data_full = subject_epochs[subj_id].get_data()
    b, a = butter(FILTER_ORDER, [fmin, fmax], btype="bandpass", fs=sfreq, output="ba")
    analytic = hilbert(filtfilt(b, a, data_full, axis=-1), axis=-1)
    phase = np.angle(analytic[:, :, t_mask])
    _phase_cache[key] = phase
    return phase


def align_by_selection(subj_a: int, subj_b: int):
    """idx_a, idx_b such that the two subjects' trials correspond to the SAME
    original task event (intersect on epochs.selection). See plv-workflow §3.3."""
    sel_a = subject_epochs[subj_a].selection
    sel_b = subject_epochs[subj_b].selection
    common = np.intersect1d(sel_a, sel_b)
    if len(common) == 0:
        return np.array([], int), np.array([], int), common
    return np.searchsorted(sel_a, common), np.searchsorted(sel_b, common), common

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — FULL cross-brain PLV matrix   (the matrix-line change)
#
# PLV[i,k] = mean_t | mean_trials exp(i (phi_A[i,t] - phi_B[k,t])) |
# for every electrode i on head A and k on head B. Across-trial PLV (matches
# v4's baseline finding: the random-phase floor is √(π/4·n_trials)).
#
# Implemented as a batched complex matmul over time (BLAS-fast), chunked over
# time to bound memory. cross_t[i,k] = (1/Ntr) Σ_tr zA[tr,i,t] · conj(zB[tr,k,t]).
# ═════════════════════════════════════════════════════════════════════════════
def matrix_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """phase_a, phase_b : (n_trials, n_channels, n_times) — already aligned.
    Returns (n_channels, n_channels): rows = head-A channels, cols = head-B."""
    assert phase_a.shape == phase_b.shape, "Phase arrays must be aligned."
    n_tr, n_ch, n_t = phase_a.shape
    za_t = np.exp(1j * phase_a).transpose(2, 1, 0)          # (n_t, n_ch, n_tr)
    zb_t = np.conj(np.exp(1j * phase_b).transpose(2, 0, 1)) # (n_t, n_tr, n_ch)
    acc = np.zeros((n_ch, n_ch))
    for s in range(0, n_t, TIME_CHUNK):
        e = min(s + TIME_CHUNK, n_t)
        cross = np.matmul(za_t[s:e], zb_t[s:e]) / n_tr      # (chunk, n_ch, n_ch)
        acc += np.abs(cross).sum(axis=0)
    return acc / n_t

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Per-triad friend / averaged-non-friend matrices, per band
#
# Orientation convention: non-friend pairs are oriented friend=rows,
# non-friend=cols (so averaging the two non-friend pairs is coherent — both
# share the non-friend C on the column axis). M(Y,X) = M(X,Y).T, so we transpose
# when the tuple has the non-friend in slot A.
# ═════════════════════════════════════════════════════════════════════════════
print("Computing full cross-brain PLV matrices per pair …\n")

# band -> triad_id -> {'friend': M, 'nonfriend': [M1, M2]}
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

    # orient so non-friend is on the column axis for non-friend pairs
    transpose = (row["pair_label"] == "non-friend" and row["status_A"] == "No")

    for band in FREQ_BANDS:
        pa = get_phase(sid_a, band)[idx_a]
        pb = get_phase(sid_b, band)[idx_b]
        M = matrix_plv(pa, pb)
        if transpose:
            M = M.T
        slot = mats[band].setdefault(tid, {"friend": [], "nonfriend": []})
        slot["friend" if row["pair_label"] == "friend" else "nonfriend"].append(M)
    print(f"  Triad {tid} {row['participant_A']}–{row['participant_B']} "
          f"({row['pair_label']}): n_aligned={len(common)}")

# Keep only triads with the complete 1-friend + 2-non-friend set, in BOTH bands.
triad_ids = sorted(
    tid for tid in mats[next(iter(FREQ_BANDS))]
    if all(len(mats[b][tid]["friend"]) == 1 and len(mats[b][tid]["nonfriend"]) == 2
           for b in FREQ_BANDS)
)
print(f"\n{len(triad_ids)} triads with complete pair sets (both bands)\n")

# Per-triad difference D = M_friend − mean(M_nonfriend), shape (n_triads, n_ch, n_ch)
diff_by_band, friend_mean, nf_mean = {}, {}, {}
for band in FREQ_BANDS:
    D = np.stack([
        mats[band][tid]["friend"][0] - np.mean(mats[band][tid]["nonfriend"], axis=0)
        for tid in triad_ids
    ])                                              # (n_triads, n_ch, n_ch)
    diff_by_band[band] = D
    friend_mean[band] = np.mean([mats[band][tid]["friend"][0] for tid in triad_ids], axis=0)
    nf_mean[band]     = np.mean([np.mean(mats[band][tid]["nonfriend"], axis=0)
                                 for tid in triad_ids], axis=0)
    np.save(os.path.join(OUTPUT_DIR, f"plv_diff_{band}.npy"), D)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Inter-brain a/b/c adjacency  =  A_chan ⊗ A_chan
#
# Derivation: pairs are p=(i,k). Dumas's three neighbourhood cases reduce to a
# single rule when the single-head adjacency A is SELF-INCLUSIVE (a channel is
# its own neighbour):  (i,k) ~ (i',k')  iff  i'∈neigh(i) AND k'∈neigh(k).
#   • both ends adjacent (i'≠i, k'≠k)          → Dumas case (a)
#   • shared head-A electrode (i'=i, k'∈neigh) → case (b)
#   • shared head-B electrode (k'=k, i'∈neigh) → case (c)
# Over the flattened index p = i*n_ch + k, that rule IS the Kronecker product
# A ⊗ A. (`metaconn_matrix_2brains` computes the same object inside HyPyP.)
# ═════════════════════════════════════════════════════════════════════════════
def build_single_head_adjacency(info) -> sparse.csr_matrix:
    A, names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    assert names == info["ch_names"], "adjacency channel order mismatch"
    A = (A + sparse.eye(A.shape[0])) > 0          # force self-inclusion
    return A.tocsr()


def build_pair_adjacency(A_chan: sparse.csr_matrix) -> sparse.csr_matrix:
    """A ⊗ A over flattened pairs p = i*n_ch + k (Dumas a/b/c)."""
    return (sparse.kron(A_chan, A_chan, format="csr") > 0).tocsr()


A_chan   = build_single_head_adjacency(info_ref)
pair_adj = build_pair_adjacency(A_chan)
# Joint (band × pair) adjacency — BLOCK-DIAGONAL across bands.
# alpha (8–12) and beta (13–30) are categorical, non-contiguous bands, NOT a
# continuous frequency axis: clusters must NOT be allowed to merge across them
# (an earlier version used combine_adjacency(n_bands, …), which chains the bands
# and lets an alpha blob fuse to the same pair in beta). Block-diagonal keeps
# clusters strictly within-band, while the max-statistic permutation STILL
# corrects FWER jointly across both bands — under H0 the running max is taken
# over every cluster in both blocks per permutation. Node order is band-major
# (band*n_pairs + pair), matching the (n_bands, n_pairs) layout that
# permutation_cluster_1samp_test flattens its test axes into.
adjacency = sparse.block_diag([pair_adj] * len(FREQ_BANDS), format="csr")
print(f"Adjacency built: {n_channels} ch → {pair_adj.shape[0]} pairs/band, "
      f"{adjacency.shape[0]} band×pair nodes "
      f"({len(FREQ_BANDS)} disjoint band blocks, no cross-band edges).\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Cluster-based permutation test (paired sign-flip across triads)
#
# X[obs, band, pair] = per-triad (friend − non-friend) difference. The
# one-sample t across triads tests H0: mean difference = 0 — i.e. the paired
# friend-vs-non-friend contrast — and the sign-flip permutation is exactly
# Maris & Oostenveld's within-subjects scheme.
# ═════════════════════════════════════════════════════════════════════════════
band_order = list(FREQ_BANDS)
X = np.stack([diff_by_band[b].reshape(len(triad_ids), -1) for b in band_order], axis=1)
print(f"Cluster test input X: {X.shape}  (n_triads, n_bands, n_pairs)")
print(f"  threshold |t|>{CLUSTER_T_THRESHOLD}, tail={CLUSTER_TAIL}, "
      f"{N_PERMUTATIONS} permutations …\n")

T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
    X, threshold=CLUSTER_T_THRESHOLD, n_permutations=N_PERMUTATIONS,
    tail=CLUSTER_TAIL, adjacency=adjacency, out_type="mask",
    seed=RNG_SEED, n_jobs=1, verbose=False,
)
# T_obs: (n_bands, n_pairs). Each clusters[ci] is a bool mask of the same shape.
n_pairs = n_channels * n_channels
T_map = T_obs.reshape(len(FREQ_BANDS), n_channels, n_channels)   # for plotting
sig = [ci for ci, p in enumerate(cluster_pv) if p < 0.05]
print("=" * 70)
print(f"CLUSTER RESULTS — condition {PLV_CONDITIONS}, {len(triad_ids)} triads")
print("=" * 70)
print(f"{len(clusters)} candidate clusters; {len(sig)} significant at p<0.05.\n")

cluster_rows = []
for ci in np.argsort(cluster_pv):                      # smallest p first
    mask = clusters[ci]                                # (n_bands, n_pairs) bool
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
        print(f"     e.g. {cluster_rows[-1]['example_pairs']}")

if not sig:
    print("  No significant clusters. (Consistent with v4's ROI null — but now"
          " tested over the full grid, both bands, with FWER control.)")

pd.DataFrame(cluster_rows).to_csv(
    os.path.join(OUTPUT_DIR, "plv_cluster_results.csv"), index=False)
print(f"\nCluster table → {os.path.join(OUTPUT_DIR, 'plv_cluster_results.csv')}")

# Persist the permutation null so the Level-1 inference histogram (Step 8c) can
# be redrawn without re-running the test. H0 is the max-statistic null: per
# permutation, the most extreme cluster mass over BOTH bands (two-sided, so the
# comparison is on |H0|). This is the array a reviewer asks for.
np.save(os.path.join(OUTPUT_DIR, "plv_H0.npy"), H0)
print(f"Permutation null H0 → {os.path.join(OUTPUT_DIR, 'plv_H0.npy')}  "
      f"({len(H0)} values)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Figures
#   (a) t-map heatmaps (friend−non-friend), 64×64 per band, significant pairs boxed
#   (b) participation topomaps: per channel, how many significant pairs it joins
#   (c) permutation-null histogram (Level-1 inference view) — H0 with the
#       observed largest cluster mass and the 95th-percentile line marked
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
                     f"t(F−NF); black = sig cluster", fontsize=10)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    fig.suptitle(f"Inter-brain PLV t-map (friend − non-friend) · {PLV_CONDITIONS} · "
                 f"N={len(triad_ids)} triads", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "plv_tmap_friend_vs_nonfriend.png")
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
        deg_a = msk.sum(axis=1).astype(float)     # head-A channel degree
        deg_b = msk.sum(axis=0).astype(float)     # head-B channel degree
        for r, (deg, side) in enumerate([(deg_a, "head A"), (deg_b, "head B")]):
            ax = axes[r, col]
            vmax = max(deg.max(), 1e-9)
            mne.viz.plot_topomap(deg, info_ref, axes=ax, show=False,
                                 cmap="Reds", vlim=(0, vmax))
            ax.set_title(f"{band} · {side}\n# sig pairs / channel", fontsize=9)
    fig.suptitle(f"Where significant friend≠non-friend pairs concentrate · "
                 f"{PLV_CONDITIONS}", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "plv_participation_topomap.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


def plot_null_histogram():
    """Level-1 inference view (stats_for_eeg_explainer §7.7): the max-statistic
    permutation null H0 as a histogram, with the observed largest cluster mass
    and the 95th-percentile line marked. ONE histogram — two-sided max-statistic
    over the alpha+beta family — so the marked value is the single largest real
    cluster overall (not one plot per band). For a null the observed line falls
    in the BULK, not the tail: the most direct display of 'our best blob is
    indistinguishable from chance'. Complements, does not replace, the N power
    caveat — it shows the observed was unremarkable against THIS null, not what a
    genuine small effect would have looked like."""
    if H0 is None or len(H0) == 0:
        print("Null histogram skipped (no H0).")
        return
    # tail=0: p-values compare |H0| (see _pval_from_histogram), so the null axis
    # is the absolute max cluster mass. Observed = largest |Σ t| over clusters.
    null = np.abs(H0)
    obs_stats = np.array([abs(float(T_obs[clusters[ci]].sum()))
                          for ci in range(len(clusters))]) if clusters else np.array([0.0])
    obs_max = float(obs_stats.max())
    p_obs   = float(min(cluster_pv)) if len(cluster_pv) else 1.0
    p95     = float(np.percentile(null, 95))
    pct     = float((null < obs_max).mean() * 100)   # observed's percentile in null

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null, bins=60, color="0.82", edgecolor="0.55", linewidth=0.3)
    ax.axvline(p95, color="0.30", linestyle="--", linewidth=1.2,
               label=f"95th percentile = {p95:.1f}")
    ax.axvline(obs_max, color="crimson", linewidth=2.2,
               label=f"observed max = {obs_max:.1f}  (p = {p_obs:.3f}, ~{pct:.0f}th pct)")
    ax.set_xlabel("max cluster mass  |Σ t|  per permutation")
    ax.set_ylabel("permutations")
    ax.set_title(f"Permutation null (H0) vs. observed · {PLV_CONDITIONS} · "
                 f"N={len(triad_ids)} triads · {len(H0)} perms\n"
                 f"two-sided max-statistic over the alpha + beta family",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "plv_null_histogram.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


plot_tmaps()
plot_participation()
plot_null_histogram()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Verification (plv-proposal-matrix §6). Cheap self-checks; print PASS/FAIL.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nVERIFICATION\n" + "=" * 70)

# (2) PLV(signal, itself) — diagonal must be 1.0.
_subj = next(iter(subject_epochs))
_p = get_phase(_subj, band_order[0])
_self = matrix_plv(_p, _p)
print(f"  PLV(x,x) diagonal mean = {np.diag(_self).mean():.6f}  "
      f"(expect 1.0) … {'PASS' if np.allclose(np.diag(_self), 1.0, atol=1e-6) else 'FAIL'}")

# (4) a/b/c adjacency on a toy 3-electrode line: neigh(0)={0,1} → pair (0,0) has
#     4 neighbours {(0,0),(0,1),(1,0),(1,1)}.
toy = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))  # self-incl line
toy_pair = (sparse.kron(toy, toy, format="csr") > 0).tocsr()
deg00 = toy_pair[0].toarray().sum()
print(f"  toy a/b/c: pair(0,0) degree = {int(deg00)} (expect 4) … "
      f"{'PASS' if deg00 == 4 else 'FAIL'}")

# (adjacency sanity) symmetric + self-inclusive on the real montage.
sym = (pair_adj != pair_adj.T).nnz == 0
diag_ok = (pair_adj.diagonal() > 0).all()
print(f"  pair adjacency symmetric={sym}, self-inclusive={bool(diag_ok)} … "
      f"{'PASS' if sym and diag_ok else 'FAIL'}")

# (3) Homologous-diagonal regression (proposal §6.3). The diagonal of the full
#     matmul PLV matrix — PLV between the SAME-named electrode on each head — must
#     reproduce v4's homologous-channel PLV. We recompute it here with an
#     independent, loop-free reference path (the canonical Lachaux across-trial
#     form) and assert the new matrix code agrees to numerical precision. This is
#     the regression test that links the matrix plumbing back to trusted v4.
def _homologous_plv_reference(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """Per-channel across-trial PLV between homologous electrodes (v4's measure).
    phase_*: (n_tr, n_ch, n_t) → (n_ch,)."""
    dphi = phase_a - phase_b
    return np.abs(np.mean(np.exp(1j * dphi), axis=0)).mean(axis=1)


_reg_pair = None
for _x, _y in itertools.combinations(list(subject_epochs), 2):
    _ia, _ib, _common = align_by_selection(_x, _y)
    if len(_common) >= 1:
        _reg_pair = (_x, _y, _ia, _ib)
        break
if _reg_pair is not None:
    _x, _y, _ia, _ib = _reg_pair
    _pa = get_phase(_x, band_order[0])[_ia]
    _pb = get_phase(_y, band_order[0])[_ib]
    _diag = np.diag(matrix_plv(_pa, _pb))
    _ref  = _homologous_plv_reference(_pa, _pb)
    _max_err = float(np.max(np.abs(_diag - _ref)))
    print(f"  homologous-diagonal regression: max|matmul − reference| = "
          f"{_max_err:.2e}  (expect ~0) … "
          f"{'PASS' if np.allclose(_diag, _ref, atol=1e-10) else 'FAIL'}")
else:
    print("  homologous-diagonal regression: SKIPPED (no overlapping trial pair)")

# (1) Trial-alignment hard-stop (proposal §6.1). align_by_selection must pair
#     ONLY trials sharing an original-event index, and must yield ZERO pairs on
#     disjoint selections — so a misaligned triad can never silently mispair.
#     Tested on toy selection arrays via .selection-only stand-ins.
class _SelOnly:
    def __init__(self, sel): self.selection = np.asarray(sel)


_TA, _TB = -101, -102
subject_epochs[_TA] = _SelOnly([0, 1, 2, 5, 9])
subject_epochs[_TB] = _SelOnly([1, 2, 3, 5, 8, 9])
_ia, _ib, _common = align_by_selection(_TA, _TB)
_aligned_ok = (
    np.array_equal(_common, [1, 2, 5, 9])
    and np.array_equal(subject_epochs[_TA].selection[_ia], _common)
    and np.array_equal(subject_epochs[_TB].selection[_ib], _common)
)
subject_epochs[_TB] = _SelOnly([100, 101, 102])      # disjoint from _TA
_ia2, _ib2, _common2 = align_by_selection(_TA, _TB)
_disjoint_ok = (len(_common2) == 0)
del subject_epochs[_TA], subject_epochs[_TB]
print(f"  trial-alignment: shared-index pairing={_aligned_ok}, "
      f"disjoint→empty={_disjoint_ok} … "
      f"{'PASS' if _aligned_ok and _disjoint_ok else 'FAIL'}")

print("\nDone.  Outputs in:", OUTPUT_DIR)
print("  • plv_cluster_results.csv      — the headline: any significant clusters?")
print("  • plv_tmap_friend_vs_nonfriend.png  — 64×64 t-map per band")
print("  • plv_participation_topomap.png     — where sig pairs concentrate")
print("  • plv_null_histogram.png        — permutation null vs. observed (Level-1)")
print("  • plv_H0.npy                    — the permutation null distribution")
print("  • plv_diff_<band>.npy           — per-triad difference matrices")
