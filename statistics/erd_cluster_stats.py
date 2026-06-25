"""erd_cluster_stats.py — shared compute layer for the single-brain ERD condition statistics.

Imported by both 07_permutation.py (cluster test + descriptive figures) and
07_permutation_violin.py (effect-size companion). Owns everything that must be
*identical* between the two so the figure can never silently disagree with the
inference: the data load, the Morlet TFR, subject-ID-matched pairing, the
spatio-temporal cluster permutation test, and the a-priori-window extractor for the
parametric companion.

Separation of concerns: this module owns data + compute + statistics; the two
scripts own only presentation (labels, colours, figure layout).

LO5 — two legs, kept independent:
  • NON-PARAMETRIC : spatio_temporal_cluster_1samp_test over the 4-channel ROI and
    the full 0–4 s window (`run_cluster`). No window assumption; controls the
    family-wise error across time and space within each contrast×band test.
  • PARAMETRIC     : paired t + Cohen's dz on a per-subject ERD value averaged in a
    *fixed, a-priori* ROI×time box (`apriori_window_values` + `paired_parametric`).
    Because the box is specified in advance — NOT read off the significant cluster —
    these statistics are not selection-biased (no double dipping).

The earlier draft summarised each subject by the mean ERD *inside the significant
cluster*; that is circular (the cluster was chosen for having a large A−B gap) and
has been removed in favour of the a-priori window here.
"""

import os
import glob
import numpy as np
from scipy import stats as scipy_stats
import mne
from mne.stats import spatio_temporal_cluster_1samp_test

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)


def epoch_files():
    """Sorted list of the parent paper's preprocessed epoch files."""
    return sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))


# ─────────────────────────────────────────────────────────────────────────────
# ROI / BANDS / TFR  (must match the ERD visualisation pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────
CHANNELS = ["C3", "O1", "Oz", "O2"]
FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}
FOI = np.linspace(1, 30, 30, dtype=int)
N_CYCLES = 3 + 0.5 * FOI
BASELINE = (-0.25, 0)            # seconds
PLOT_TMIN, PLOT_TMAX = 0.0, 4.0  # analysis / cluster-test window

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER PERMUTATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
N_PERMUTATIONS = 5000   # final value (draft was 1024); see perm_p_floor for the
                        # 2**N sign-flip ceiling at small subject counts
ALPHA_CLUSTER  = 0.05   # cluster-forming threshold (two-tailed p)
P_ACCEPT       = 0.05   # cluster significance level
SEED           = 42

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CONTRASTS — each its own pre-registered family
# ─────────────────────────────────────────────────────────────────────────────
CONTRASTS = [
    ("solo_feedback",    "T1P",  "T1Pn", "Solo: With vs. No Feedback"),
    ("trio_feedback",    "T3P",  "T3Pn", "Trio: With vs. No Feedback"),
    ("solo_vs_trio_fb",  "T1P",  "T3P",  "Solo vs. Trio (With Feedback)"),
    ("solo_vs_trio_nfb", "T1Pn", "T3Pn", "Solo vs. Trio (No Feedback)"),
]
SOLO_TRIO_KEYS = ("T1P", "T1Pn", "T3P", "T3Pn")

# ─────────────────────────────────────────────────────────────────────────────
# MULTIPLE-COMPARISON STANCE  (pre-registered; declare it, don't bury it)
# ─────────────────────────────────────────────────────────────────────────────
MULTIPLE_COMPARISONS = (
    "Family structure: each of the 4 contrasts is a separate pre-registered "
    "family (a distinct scientific question). The cluster permutation test "
    "controls the family-wise error rate across time AND space within each "
    "contrast x band test. The remaining multiplicity is the 2 frequency bands "
    "per contrast: band is treated as a within-family factor and both bands are "
    "reported jointly without pooling p. No correction is applied across the 4 "
    "contrasts (independent questions). This is a pre-registration choice — if "
    "the supervisor prefers band as a family boundary, apply Bonferroni/FDR "
    "across the 2 bands per contrast."
)

# ─────────────────────────────────────────────────────────────────────────────
# A-PRIORI ROI + WINDOW for the PARAMETRIC companion (fixed in advance)
# ─────────────────────────────────────────────────────────────────────────────
# The per-subject value feeding the paired t / Cohen's d is the mean ERD in this
# fixed (ROI x time) box. It is chosen in advance — NOT from the significant
# cluster — so the parametric statistics are not selection-biased.
#
# Default ROI = the full analysis ROI (same channels as the cluster test), so the
# only pre-registered choice this leg adds beyond the cluster test is the time
# window. A more physiologically targeted alternative (occipital alpha / central
# beta) is left commented for a deliberate, separately-justified switch.
APRIORI_WINDOW = (0.5, 3.5)   # s — sustained push; drops onset/offset transients
APRIORI_ROI = {
    "alpha": CHANNELS,
    "beta":  CHANNELS,
    # "alpha": ["O1", "Oz", "O2"],   # occipital alpha ERD
    # "beta":  ["C3"],               # sensorimotor mu/beta ERD
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def subject_id(path):
    """Subject id = filename prefix before '_FG_preprocessed-epo.fif'."""
    return os.path.basename(path).split("_FG_preprocessed-epo.fif")[0]


def load_group_tfrs(channels=CHANNELS, verbose=True):
    """Per-subject baseline-corrected % ERD TFRs, keyed by condition then subject id.

    Parameters
    ----------
    channels : list[str] | None
        ROI channels to keep. ``None`` keeps **all EEG channels** (whole-scalp
        variant — used by 07_permutation_wholescalp.py). The default 4-channel
        ROI is unchanged, so the ROI-based scripts (07_permutation.py, the violin
        companion) keep their v3 behaviour exactly.

    Returns
    -------
    group_tfr : dict[str, dict[str, mne.time_frequency.AverageTFR]]
        group_tfr[condition][subject_id] = AverageTFR  (% signal change)
    info_ref : mne.Info
        Channels in `channels` order (from the first file read); all EEG channels
        when ``channels is None``.
    """
    files = epoch_files()
    if verbose:
        print(f"Found {len(files)} epoch files\n")

    group_tfr: dict[str, dict[str, "mne.time_frequency.AverageTFR"]] = {}
    info_ref = None

    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f)
        sid = subject_id(f)
        if verbose:
            print(f"Processing: {os.path.basename(f)}  (subject {sid})")

        epochs = mne.read_epochs(f, preload=True)
        if channels is None:
            epochs.pick("eeg")   # whole-scalp: keep all 64 EEG channels
        else:
            available = [ch for ch in channels if ch in epochs.ch_names]
            epochs.pick(available)

        if info_ref is None:
            info_ref = epochs.info.copy()

        for cond in [k for k in epochs.event_id if k in SOLO_TRIO_KEYS]:
            tfr = epochs[cond].compute_tfr(
                method="morlet", freqs=FOI, n_cycles=N_CYCLES,
                return_itc=False, average=False,
            )
            tfr_avg = tfr.average()
            tfr_avg.apply_baseline(BASELINE, mode="percent")
            tfr_avg.data *= 100  # → % signal change
            group_tfr.setdefault(cond, {})[sid] = tfr_avg

    return group_tfr, info_ref


def reference_times(group_tfr):
    """The (shared) time axis, read from an arbitrary subject/condition."""
    cond = next(iter(group_tfr))
    sid = next(iter(group_tfr[cond]))
    return group_tfr[cond][sid].times


def build_adjacency(info_ref):
    """Channel adjacency from the montage, in info_ref channel order."""
    adjacency, _ = mne.channels.find_ch_adjacency(info_ref, ch_type="eeg")
    return adjacency


# ─────────────────────────────────────────────────────────────────────────────
# BAND AVERAGING + SUBJECT-ID-MATCHED DIFFERENCE ARRAY
# ─────────────────────────────────────────────────────────────────────────────
def band_spatiotemporal(tfr, fmin, fmax, t_mask):
    """Collapse a TFR over a band and crop in time → (n_times_masked, n_channels)."""
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    band = tfr.data[:, f_mask, :].mean(axis=1)   # (n_ch, n_times)
    return band[:, t_mask].T                      # (n_times, n_ch)


def matched_subjects(group_tfr, cond_a, cond_b):
    """Subject ids present in BOTH conditions, sorted (the paired sample)."""
    return sorted(set(group_tfr[cond_a]) & set(group_tfr[cond_b]))


def build_X(group_tfr, cond_a, cond_b, fmin, fmax, times_full,
            tmin=PLOT_TMIN, tmax=PLOT_TMAX):
    """Subject-ID-matched paired difference array for the cluster test.

    Pairing is by subject id: a subject missing either condition is *dropped*,
    never silently misaligned (the old positional `[:n]` pairing could subtract
    two different people). Row i of A and B is therefore guaranteed the same
    subject.

    Returns
    -------
    X    : (n_subj, n_times, n_ch)  difference A − B
    st_a : (n_subj, n_times, n_ch)  condition A values
    st_b : (n_subj, n_times, n_ch)  condition B values
    sids : list[str]                matched subject ids, in row order
    """
    sids = matched_subjects(group_tfr, cond_a, cond_b)
    t_mask = (times_full >= tmin) & (times_full <= tmax)
    st_a = np.stack([band_spatiotemporal(group_tfr[cond_a][s], fmin, fmax, t_mask)
                     for s in sids])
    st_b = np.stack([band_spatiotemporal(group_tfr[cond_b][s], fmin, fmax, t_mask)
                     for s in sids])
    return st_a - st_b, st_a, st_b, sids


# ─────────────────────────────────────────────────────────────────────────────
# NON-PARAMETRIC: spatio-temporal cluster permutation test
# ─────────────────────────────────────────────────────────────────────────────
def run_cluster(X, adjacency):
    """Paired spatio-temporal cluster permutation test on X = A − B.

    Cluster-forming threshold is the two-tailed α critical t (df = n_obs − 1); the
    cluster p-value itself comes from the sign-flip permutation null, so no
    t-distribution assumption is load-bearing for inference.
    """
    n_obs = X.shape[0]
    df = n_obs - 1
    threshold = scipy_stats.t.ppf(1.0 - ALPHA_CLUSTER / 2.0, df=df)
    T_obs, clusters, cluster_p, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=N_PERMUTATIONS,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        out_type="mask",
        seed=SEED,
        n_jobs=1,
        verbose=False,
    )
    return T_obs, clusters, cluster_p, H0


def significant_clusters(clusters, cluster_p, T_obs):
    """Union the p < P_ACCEPT clusters into one boolean mask + a list of (mask, p)."""
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    sig = []
    for c, p in zip(clusters, cluster_p):
        if p < P_ACCEPT:
            sig_mask |= c
            sig.append((c, p))
    return sig_mask, sig


def perm_p_floor(n_subjects):
    """Approx smallest achievable cluster p: ~1 / min(n_perm, 2**N) sign-flips.

    With a small subject N the permutation ceiling is 2**N sign-flips, which can be
    below N_PERMUTATIONS — report this alongside any p so a 'p = 0.0002' is read
    against what was actually achievable.
    """
    return 1.0 / min(N_PERMUTATIONS, 2 ** n_subjects)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC: a-priori-window per-subject value + paired t / Cohen's dz
# ─────────────────────────────────────────────────────────────────────────────
def apriori_window_values(group_tfr, cond, band_name, info_ref, times_full, sids):
    """Per-subject mean ERD in the fixed a-priori (ROI × band × time-window) box.

    Independent of any cluster mask → a legitimate, non-circular basis for a
    parametric paired test and Cohen's d. `sids` fixes subject order — pass the
    matched list from `build_X` so A and B align.
    """
    ch_names = info_ref.ch_names
    roi_idx = np.array([ch_names.index(c) for c in APRIORI_ROI[band_name]
                        if c in ch_names])
    fmin, fmax = FREQ_BANDS[band_name]
    t0, t1 = APRIORI_WINDOW

    out = np.empty(len(sids))
    for i, s in enumerate(sids):
        tfr = group_tfr[cond][s]
        f_idx = np.where((tfr.freqs >= fmin) & (tfr.freqs <= fmax))[0]
        t_idx = np.where((tfr.times >= t0) & (tfr.times <= t1))[0]
        # tfr.data: (n_ch, n_freq, n_times) → mean over ROI × band × window
        out[i] = tfr.data[np.ix_(roi_idx, f_idx, t_idx)].mean()
    return out


def paired_parametric(a_vals, b_vals):
    """Paired t-test and paired Cohen's dz on the a-priori-window per-subject values.

    Not selection-biased (the window is fixed in advance), so unlike a statistic
    computed inside the significant cluster these may be reported as the contrast's
    parametric test and effect size.

    Returns
    -------
    t_stat, p_value, dz
    """
    diff = a_vals - b_vals
    t_stat, p_value = scipy_stats.ttest_rel(a_vals, b_vals)
    dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return t_stat, p_value, dz
