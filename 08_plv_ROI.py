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

CONTRAST       = ("T3P", "T1P")          # RQ-GS primary. RQ-FB: ("T3P", "T3Pn"), ("T3P", "T1P")
COND_A, COND_B = CONTRAST
_tag           = f"{COND_A}_vs_{COND_B}"

# ── ROI restriction ──────────────────────────────────────────────────────────
# Whole analysis (phase extraction, PLV, adjacency, t-map, topomaps) is
# restricted to this channel subset. Order here defines row/col order of all
# n_channels × n_channels matrices below.
ROI = ["C3", "O1", "O2", "Oz"]
_roi_tag = "-".join(ROI)

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv_conditions_"
    + _tag + f"_ROI-{_roi_tag}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))

# ── Pre-registered analysis choices ──────────────────────────────────────────
FREQ_BANDS     = {"alpha": (8, 12), "beta": (13, 30)}   # cluster family
PLV_TMIN, PLV_TMAX = 0.0, 4.0            # the push phase
FILTER_ORDER   = 4

CLUSTER_T_THRESHOLD = 2.0                # Dumas |t|>2
CLUSTER_TAIL        = 0                  # two-sided
N_PERMUTATIONS      = 5000
RNG_SEED            = 42

MATCH_N        = True                    # equalise trial count per pair per cond

N_MIN          = 25                      # min matched trials per pair per condition
EXCLUDE_TRIADS = [] #[330]
TIME_CHUNK     = 500                     # matrix_plv memory knob (perf only)

rng = np.random.default_rng(RNG_SEED)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — pairs, NOT pair-types
# ═════════════════════════════════════════════════════════════════════════════
print("Loading overview dataframe (triad/subject structure only) …")
fg_df = pd.read_pickle(OVERVIEW_PKL)
print(f"  {len(fg_df)} subjects, {fg_df['Triad_id'].nunique()} triads\n")


def build_pairs(fg_df: pd.DataFrame) -> pd.DataFrame:
    """One row per within-triad pair (A-B, A-C, B-C), with each member's
    Subject_id. Structure only — no friendship column is read."""
    rows = []
    for triad_id, grp in fg_df.groupby("Triad_id"):
        members = grp.set_index("Participant")["Subject_id"]
        for p_a, p_b in itertools.combinations(sorted(members.index), 2):
            rows.append({
                "Triad_id": triad_id,
                "participant_A": p_a, "participant_B": p_b,
                "subj_A": int(members.loc[p_a]),
                "subj_B": int(members.loc[p_b]),
            })
    return pd.DataFrame(rows)


pair_df = build_pairs(fg_df)
print(f"{len(pair_df)} within-triad pairs across "
      f"{pair_df['Triad_id'].nunique()} triads\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — load epochs, KEEPING BOTH CONDITIONS SPLIT, restricted to ROI
# ═════════════════════════════════════════════════════════════════════════════
print(f"Found {len(EPOCH_FILES)} epoch files; keeping conditions {CONTRAST} split.")
print(f"Restricting analysis to ROI channels: {ROI}\n")

subject_epochs: dict[int, dict[str, mne.Epochs]] = {}
info_ref = None

for fpath in EPOCH_FILES:
    exp_id = os.path.basename(fpath).split("_")[0]
    match = fg_df[fg_df["Exp_id"] == exp_id]
    if match.empty:
        print(f"  WARNING: no metadata for {exp_id}, skipping.")
        continue
    subj_id = int(match["Subject_id"].iloc[0])
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    present = {c: epochs[c] for c in CONTRAST if c in epochs.event_id}
    if len(present) < 2:
        print(f"  {exp_id} (id={subj_id}): missing one of {CONTRAST} "
              f"(has {list(present)}), skipping.")
        continue
    missing_roi = [ch for ch in ROI if ch not in epochs.ch_names]
    if missing_roi:
        print(f"  {exp_id} (id={subj_id}): missing ROI channel(s) "
              f"{missing_roi}, skipping.")
        continue
    if subj_id in subject_epochs:
        print(f"  WARNING: Subject_id {subj_id} already loaded "
              f"(exp_id={exp_id}); overwriting. Investigate.")
    subject_epochs[subj_id] = present
    if info_ref is None:
        # Build a 4-channel reference Info, ordered exactly as ROI, from
        # whichever epochs object happens to load first.
        roi_picks_ref = mne.pick_channels(present[COND_A].info["ch_names"],
                                          include=ROI, ordered=True)
        info_ref = mne.pick_info(present[COND_A].info.copy(), roi_picks_ref)
    print(f"  {exp_id} (id={subj_id}): "
          + ", ".join(f"{c}={len(present[c])}" for c in CONTRAST))

print(f"\n{len(subject_epochs)} subjects with both conditions and full ROI\n")

sfreq      = info_ref["sfreq"]
times_full = next(iter(subject_epochs.values()))[COND_A].times
t_mask     = (times_full >= PLV_TMIN) & (times_full <= PLV_TMAX)
ch_names   = info_ref["ch_names"]          # now just ROI, in ROI order
n_channels = len(ch_names)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — phase extraction (cached, ROI-restricted) + selection alignment
# ═════════════════════════════════════════════════════════════════════════════
_phase_cache: dict[tuple, np.ndarray] = {}


def get_phase(subj_id: int, cond: str, band: str) -> np.ndarray:
    """(n_trials, n_channels, n_times_in_window) for one subject × condition ×
    band, restricted to ROI channels. Filter the ROI-only data, Hilbert, crop
    to the push window. Picks are recomputed per-subject/condition from each
    epochs object's own channel order, so this is robust even if channel
    ordering differs slightly across recordings."""
    key = (subj_id, cond, band)
    if key in _phase_cache:
        return _phase_cache[key]
    fmin, fmax = FREQ_BANDS[band]
    ep = subject_epochs[subj_id][cond]
    roi_picks = mne.pick_channels(ep.info["ch_names"], include=ROI, ordered=True)
    data_full = ep.get_data(picks=roi_picks)
    b, a = butter(FILTER_ORDER, [fmin, fmax], btype="bandpass", fs=sfreq, output="ba")
    analytic = hilbert(filtfilt(b, a, data_full, axis=-1), axis=-1)
    phase = np.angle(analytic[:, :, t_mask])
    _phase_cache[key] = phase
    return phase


def aligned_idx(subj_a: int, subj_b: int, cond: str):
    """idx_a, idx_b, n — trials of the two subjects that share an original event
    index WITHIN this condition (intersect epochs.selection)."""
    sel_a = subject_epochs[subj_a][cond].selection
    sel_b = subject_epochs[subj_b][cond].selection
    common = np.intersect1d(sel_a, sel_b)
    return (np.searchsorted(sel_a, common),
            np.searchsorted(sel_b, common), len(common))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — full cross-brain PLV matrix (now n_channels = len(ROI))
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

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — per-pair PLV in each condition on MATCHED trial counts,
# ═════════════════════════════════════════════════════════════════════════════
def subsample(idx_a, idx_b, n):
    """Randomly keep n of the aligned trials (counts-matching for the floor)."""
    if len(idx_a) <= n:
        return idx_a, idx_b
    keep = np.sort(rng.permutation(len(idx_a))[:n])
    return idx_a[keep], idx_b[keep]


def pair_contrast(subj_a, subj_b, band):
    """(M_A, M_B, n_used) — full cross-brain PLV for this pair in COND_A and
    COND_B, on a MATCHED trial count so √(π/4n) is identical and cancels in the
    difference. Returns None if the matched count is below N_MIN."""
    ia_A, ib_A, nA = aligned_idx(subj_a, subj_b, COND_A)
    ia_B, ib_B, nB = aligned_idx(subj_a, subj_b, COND_B)
    n = min(nA, nB)
    if n < N_MIN:
        return None
    if MATCH_N:
        ia_A, ib_A = subsample(ia_A, ib_A, n)
        ia_B, ib_B = subsample(ia_B, ib_B, n)
    M_A = matrix_plv(get_phase(subj_a, COND_A, band)[ia_A],
                     get_phase(subj_b, COND_A, band)[ib_A])
    M_B = matrix_plv(get_phase(subj_a, COND_B, band)[ia_B],
                     get_phase(subj_b, COND_B, band)[ib_B])
    return M_A, M_B, n


print(f"Computing per-pair PLV in {COND_A} & {COND_B} (MATCH_N={MATCH_N}) …\n")

# band -> triad_id -> list of per-pair matrices. Diff feeds the test; the two
# absolute accumulators (NEW in v2) feed the descriptive absolute-level maps.
pair_diffs: dict[str, dict[int, list]] = {b: {} for b in FREQ_BANDS}
pair_A:     dict[str, dict[int, list]] = {b: {} for b in FREQ_BANDS}
pair_B:     dict[str, dict[int, list]] = {b: {} for b in FREQ_BANDS}
matched_counts: list[int] = []           # for the Step-9 matched-N assertion

for _, row in pair_df.iterrows():
    tid = row["Triad_id"]
    sid_a, sid_b = row["subj_A"], row["subj_B"]
    if tid in EXCLUDE_TRIADS:
        continue
    if sid_a not in subject_epochs or sid_b not in subject_epochs:
        continue
    skip = False
    for band in FREQ_BANDS:
        res = pair_contrast(sid_a, sid_b, band)
        if res is None:
            skip = True
            break
        M_A, M_B, n = res
        pair_diffs[band].setdefault(tid, []).append(M_A - M_B)
        pair_A[band].setdefault(tid, []).append(M_A)
        pair_B[band].setdefault(tid, []).append(M_B)
        if band == next(iter(FREQ_BANDS)):
            matched_counts.append(n)
    if skip:
        print(f"  Triad {tid} {row['participant_A']}–{row['participant_B']}: "
              f"matched n < {N_MIN}, skipping pair.")
        continue
    print(f"  Triad {tid} {row['participant_A']}–{row['participant_B']}: "
          f"matched n={matched_counts[-1]}")

# Keep only triads with all 3 pairs present in BOTH bands → one diff per triad.
triad_ids = sorted(
    tid for tid in pair_diffs[next(iter(FREQ_BANDS))]
    if all(len(pair_diffs[b][tid]) == 3 for b in FREQ_BANDS)
)
print(f"\n{len(triad_ids)} triads with all 3 pairs in both bands\n")


diff_by_band, absA_by_band, absB_by_band = {}, {}, {}
for band in FREQ_BANDS:
    D = np.stack([np.mean(pair_diffs[band][tid], axis=0) for tid in triad_ids])
    A = np.stack([np.mean(pair_A[band][tid],     axis=0) for tid in triad_ids])
    B = np.stack([np.mean(pair_B[band][tid],     axis=0) for tid in triad_ids])
    diff_by_band[band], absA_by_band[band], absB_by_band[band] = D, A, B  # (n_tri,ch,ch)
    np.save(os.path.join(OUTPUT_DIR, f"plv_diff_{band}.npy"), D)
    np.save(os.path.join(OUTPUT_DIR, f"plv_abs_{COND_A}_{band}.npy"), A)
    np.save(os.path.join(OUTPUT_DIR, f"plv_abs_{COND_B}_{band}.npy"), B)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — inter-brain a/b/c adjacency  =  A_chan ⊗ A_chan  (ROI-sized)
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
print(f"Adjacency: {n_channels} ch (ROI={ROI}) → {pair_adj.shape[0]} pairs/band, "
      f"{adjacency.shape[0]} band×pair nodes "
      f"({len(FREQ_BANDS)} disjoint band blocks).\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — cluster-based permutation test (paired sign-flip across triads)
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
n_pairs = n_channels * n_channels
T_map = T_obs.reshape(len(FREQ_BANDS), n_channels, n_channels)
sig = [ci for ci, p in enumerate(cluster_pv) if p < 0.05]

print("=" * 70)
print(f"CLUSTER RESULTS — {COND_A} vs {COND_B}, {len(triad_ids)} triads, ROI={ROI}")
print("=" * 70)
print(f"{len(clusters)} candidate clusters; {len(sig)} significant at p<0.05.\n")

cluster_rows = []
for ci in np.argsort(cluster_pv):
    mask = clusters[ci]
    bands_in, pidx = np.where(mask)
    ii, kk = pidx // n_channels, pidx % n_channels
    sum_t  = float(T_obs[mask].sum())
    band_tag = "/".join(sorted({band_order[b] for b in np.unique(bands_in)}))
    direction = f"{COND_A}>{COND_B}" if sum_t > 0 else f"{COND_B}>{COND_A}"
    cluster_rows.append({
        "cluster": ci, "p_value": round(float(cluster_pv[ci]), 4),
        "n_pairs": int(mask.sum()), "sum_t": round(sum_t, 2),
        "bands": band_tag, "direction": direction,
        "example_pairs": "; ".join(
            f"{ch_names[i]}~{ch_names[k]}" for i, k in list(zip(ii, kk))[:8]),
    })
    if cluster_pv[ci] < 0.05:
        print(f"  cluster {ci}: p={cluster_pv[ci]:.4f}  bands={band_tag}  "
              f"n_pairs={int(mask.sum())}  sumT={sum_t:+.1f}  ({direction})")
        print(f"     e.g. {cluster_rows[-1]['example_pairs']}")

if not sig:
    print("  No significant clusters at p<0.05.")

pd.DataFrame(cluster_rows).to_csv(
    os.path.join(OUTPUT_DIR, "plv_cluster_results.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "plv_H0.npy"), H0)
print(f"\nCluster table → plv_cluster_results.csv")
print(f"Permutation null H0 → plv_H0.npy  ({len(H0)} values)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — figures
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
                     f"t({COND_A}−{COND_B}); black = sig cluster", fontsize=10)
        ax.set_xticks(range(n_channels)); ax.set_xticklabels(ch_names, rotation=90)
        ax.set_yticks(range(n_channels)); ax.set_yticklabels(ch_names)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    fig.suptitle(f"Inter-brain PLV t-map ({COND_A} − {COND_B}) · "
                 f"N={len(triad_ids)} triads · ROI={ROI}", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"plv_tmap_{_tag}.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


def plot_diff_maps():
    """Effect-size twin of the t-map: mean over triads of (M_A − M_B), in PLV
    units. Same RdBu_r/symmetric grammar as plot_tmaps so the two read side by
    side — t says 'consistent?', this says 'how big?'. No significance overlay:
    a surviving cluster (if any) is already marked on the t-map."""
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
                     f"mean ΔPLV ({COND_A}−{COND_B}); "
                     f"max|Δ|={abs(mean_diff[band]).max():.3f}", fontsize=10)
        ax.set_xticks(range(n_channels)); ax.set_xticklabels(ch_names, rotation=90)
        ax.set_yticks(range(n_channels)); ax.set_yticklabels(ch_names)
        ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, label="ΔPLV")
    fig.suptitle(f"Inter-brain PLV difference ({COND_A} − {COND_B}) · effect size "
                 f"in PLV units · N={len(triad_ids)} triads · ROI={ROI}",
                 fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"plv_diffmap_{_tag}.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


def plot_abs_maps():
    """Floor / sanity panel: grand-mean absolute cross-brain PLV per condition.
    DESCRIPTIVE ONLY — no significance overlay; inference lives on the difference.
    Each panel states its mean vs the across-trial floor √(π/4N). If the whole
    matrix sits at the floor, inter-brain coupling is essentially absent — which
    is itself a finding. N varies per pair (each matched to its own min ≥ N_MIN),
    so the reported floor uses the median matched count."""
    nb = len(FREQ_BANDS)
    gmA = {b: absA_by_band[b].mean(axis=0) for b in band_order}
    gmB = {b: absB_by_band[b].mean(axis=0) for b in band_order}
    vmax = max(max(m.max() for m in gmA.values()),
               max(m.max() for m in gmB.values()), 1e-9)
    n_med = int(np.median(matched_counts)) if matched_counts else N_MIN
    floor = float(np.sqrt(np.pi / (4 * n_med)))
    fig, axes = plt.subplots(2, nb, figsize=(5 * nb, 9))
    axes = axes.reshape(2, nb)
    for col, band in enumerate(band_order):
        for r, (gm, cond) in enumerate([(gmA[band], COND_A), (gmB[band], COND_B)]):
            ax = axes[r, col]
            im = ax.imshow(gm, cmap="magma", vmin=0, vmax=vmax,
                           aspect="equal", origin="upper")
            ax.set_title(f"{cond} · {band} "
                         f"({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)\n"
                         f"mean={gm.mean():.3f}  (floor≈{floor:.3f})", fontsize=9)
            ax.set_xticks(range(n_channels)); ax.set_xticklabels(ch_names, rotation=90)
            ax.set_yticks(range(n_channels)); ax.set_yticklabels(ch_names)
            ax.set_xlabel("head-B channel"); ax.set_ylabel("head-A channel")
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, label="PLV")
    fig.suptitle(f"Absolute inter-brain PLV per condition · floor √(π/4N) ≈ "
                 f"{floor:.3f} at N≈{n_med} trials · descriptive, not inferential "
                 f"· ROI={ROI}", fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"plv_absmap_{_tag}.png")
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
        for r, (deg, side) in enumerate([(msk.sum(axis=1).astype(float), "head A"),
                                         (msk.sum(axis=0).astype(float), "head B")]):
            ax = axes[r, col]
            mne.viz.plot_topomap(deg, info_ref, axes=ax, show=False,
                                 cmap="Reds", vlim=(0, max(deg.max(), 1e-9)))
            ax.set_title(f"{band} · {side}\n# sig pairs / channel", fontsize=9)
    fig.suptitle(f"Where significant {COND_A}≠{COND_B} pairs concentrate · ROI={ROI}",
                 fontweight="bold")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"plv_participation_{_tag}.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


def plot_null_histogram():
    """Level-1 inference view: the max-statistic permutation null with the
    observed largest cluster mass and the 95th-percentile line."""
    if H0 is None or len(H0) == 0:
        print("Null histogram skipped (no H0).")
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
    ax.set_title(f"Permutation null (H0) vs. observed · {COND_A} vs {COND_B} · "
                 f"N={len(triad_ids)} triads · {len(H0)} perms · ROI={ROI}\n"
                 f"two-sided max-statistic over the alpha + beta family", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"plv_null_histogram_{_tag}.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fn}")


plot_tmaps()
plot_diff_maps()
plot_abs_maps()
plot_participation()
plot_null_histogram()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — verification
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nVERIFICATION\n" + "=" * 70)

print(f"  (V0) ROI restriction: ch_names = {ch_names} (expect {ROI}) … "
      f"{'PASS' if ch_names == ROI else 'FAIL'}")

_subj = next(iter(subject_epochs))
_p = get_phase(_subj, COND_A, band_order[0])
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


subject_epochs[-101] = {COND_A: _SelOnly([0, 1, 2, 5, 9]),
                        COND_B: _SelOnly([0, 1, 2, 5, 9])}
subject_epochs[-102] = {COND_A: _SelOnly([1, 2, 3, 5, 8, 9]),
                        COND_B: _SelOnly([100, 101, 102])}
_, _, n_share = aligned_idx(-101, -102, COND_A)
_, _, n_disjoint = aligned_idx(-101, -102, COND_B)
del subject_epochs[-101], subject_epochs[-102]
print(f"  (V4) trial-alignment: shared={n_share} (expect 4), "
      f"disjoint={n_disjoint} (expect 0) … "
      f"{'PASS' if n_share == 4 and n_disjoint == 0 else 'FAIL'}")


if MATCH_N:
    _ok = len(matched_counts) > 0 and all(c >= N_MIN for c in matched_counts)
    print(f"  (V5) matched-N: {len(matched_counts)} pairs, all ≥ {N_MIN} "
          f"(min={min(matched_counts) if matched_counts else 'NA'}) … "
          f"{'PASS' if _ok else 'FAIL'}")
else:
    print("  (V5) matched-N: SKIPPED (MATCH_N=False — floor will NOT cancel!)")


print("  (V6) floor self-test: run CONTRAST=(X,X) separately → diff ≈ 0 (manual).")

_b0   = band_order[0]
_lhs  = diff_by_band[_b0].mean(axis=0)
_rhs  = absA_by_band[_b0].mean(axis=0) - absB_by_band[_b0].mean(axis=0)
_dev  = float(np.abs(_lhs - _rhs).max())
print(f"  (V7) diff/abs consistency: max|Δ| = {_dev:.2e} (expect ~0) … "
      f"{'PASS' if _dev < 1e-9 else 'FAIL'}")

print("\nDone.  Outputs in:", OUTPUT_DIR)
print(f"  • ROI = {ROI}  ({n_channels}×{n_channels} matrices instead of full montage)")
print(f"  • plv_cluster_results.csv          — any significant clusters?")
print(f"  • plv_tmap_{_tag}.png              — {n_channels}×{n_channels} t-map per band (inference)")
print(f"  • plv_diffmap_{_tag}.png           — {n_channels}×{n_channels} mean ΔPLV per band (effect size)")
print(f"  • plv_absmap_{_tag}.png            — abs PLV per condition vs floor (sanity)")
print(f"  • plv_participation_{_tag}.png     — where sig pairs concentrate")
print(f"  • plv_null_histogram_{_tag}.png    — permutation null vs observed")
print(f"  • plv_H0.npy / plv_diff_<band>.npy — null + per-triad differences")
print(f"  • plv_abs_<cond>_<band>.npy        — per-triad absolute PLV (both conds)")