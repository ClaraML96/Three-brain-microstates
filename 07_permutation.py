"""
statistics_friends_vs_nonfriends.py
=====================================
Statistical comparison of ERD/ERS between FRIENDS and NON-FRIENDS.

Learning objective covered:
  "Statistical comparisons of conditions using parametric tests
   or non-parametric permutation tests"

Structure
---------
STEP 1  Load TFRs per participant, labelled by friendship role
STEP 2  Extract scalar ERD scores (time × channel average per subject)
STEP 3  PARAMETRIC TESTS
          3a. Paired-samples t-test (scipy)       — one value per subject
          3b. Cohen's d effect size
          3c. Results table printed to console
STEP 4  NON-PARAMETRIC CLUSTER PERMUTATION TEST  (MNE)
          Retains full time × channel resolution
          Contrasts: friends vs non-friends
          Conditions: interactive with-feedback (Duo + Trio collapsed)
STEP 5  PLOTS
          Fig A — Violin + strip plot of scalar ERD (parametric result)
          Fig B — Mean ± SEM time-courses with cluster shading
          Fig C — Channel-averaged T-statistic with cluster shading

Friendship labelling
---------------------
Each .fif file corresponds to one participant in one experimental session.
Participants are labelled by role:
  - parts 1 & 2 are the two FRIENDS in each triad
  - part 3      is the NON-FRIEND

Adjust FRIEND_PARTS / NONFRIEND_PARTS below if your naming differs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import mne
from mne.stats import permutation_cluster_1samp_test

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet"
    r"\Skrivebord\DTU\Human Centeret Artificial Intelligence"
    r"\Thesis\data\ica_cleaned"
)
OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet"
    r"\Skrivebord\DTU\Human Centeret Artificial Intelligence"
    r"\Thesis\figures\statistics"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parts 1 & 2 are friends; part 3 is the non-friend
FRIEND_PARTS    = [1, 2]
NONFRIEND_PARTS = [3]

# All participant IDs
PIDS = ["301", "302", "303", "304"]

# Morlet parameters (identical to your ERD script)
FOI      = np.linspace(1, 30, 30, dtype=int)
N_CYCLES = 3 + 0.5 * FOI
BASELINE = (-0.25, 0)

# Analysis time window (post-stimulus)
TMIN, TMAX = 0.0, 4.0

# Frequency bands of interest
FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

# Interactive WITH-FEEDBACK conditions (duo + trio collapsed per paper)
# Adjust condition labels to match your epoch event_id if needed
INTERACTIVE_CONDITIONS = ["Condition_2", "Condition_4",   # duo with feedback
                          "Condition_8"]                  # trio with feedback

N_PERMUTATIONS = 1000   # paper uses 1000 for ERD cluster tests

# ============================================================
# STEP 1 — Load TFRs, separated by friendship role
# ============================================================
# friend_tfr[condition]    = list of AverageTFR (one per friend participant)
# nonfriend_tfr[condition] = list of AverageTFR (one per non-friend participant)

friend_tfr    = {c: [] for c in INTERACTIVE_CONDITIONS}
nonfriend_tfr = {c: [] for c in INTERACTIVE_CONDITIONS}

for pid in PIDS:
    for part in [1, 2, 3]:
        epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif")
        if not os.path.exists(epoch_file):
            print(f"  File not found, skipping: {epoch_file}")
            continue

        role = "friend" if part in FRIEND_PARTS else "nonfriend"
        print(f"\nLoading {pid} part {part}  [{role}]")
        epochs = mne.read_epochs(epoch_file, preload=True)

        for condition in INTERACTIVE_CONDITIONS:
            if condition not in epochs.event_id:
                print(f"  Condition '{condition}' not in file — skipping")
                continue

            tfr = epochs[condition].compute_tfr(
                method="morlet",
                freqs=FOI,
                n_cycles=N_CYCLES,
                return_itc=False,
                average=False,
            )
            tfr_avg = tfr.average()
            tfr_avg.apply_baseline(BASELINE, mode="percent")
            tfr_avg.data *= 100

            if role == "friend":
                friend_tfr[condition].append(tfr_avg)
            else:
                nonfriend_tfr[condition].append(tfr_avg)

print("\nAll TFRs loaded.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def band_time_power(tfr_list, fmin, fmax, tmin, tmax):
    """
    Extract power averaged over [fmin, fmax] Hz and cropped to [tmin, tmax] s.
    Returns array of shape (n_subjects, n_channels, n_times).
    """
    out = []
    for tfr in tfr_list:
        f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
        t_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
        out.append(tfr.data[:, f_mask, :][:, :, t_mask].mean(axis=1))
    return np.array(out)   # (n_subjects, n_channels, n_times)


def scalar_erd(tfr_list, fmin, fmax, tmin, tmax):
    """
    Single scalar ERD value per subject:
    average over frequency band, all channels, and full time window.
    Returns 1-D array of shape (n_subjects,).
    """
    arr = band_time_power(tfr_list, fmin, fmax, tmin, tmax)
    return arr.mean(axis=(1, 2))   # (n_subjects,)


def cohens_d(a, b):
    """Paired Cohen's d = mean(diff) / std(diff)."""
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def cluster_time_mask(cluster, n_ch, n_t):
    """
    MNE returns clusters as a tuple containing flat indices into
    (n_channels * n_times).  Convert to a 1-D boolean time mask.
    """
    flat_idx = cluster[0]
    mask_2d = np.zeros((n_ch, n_t), dtype=bool)
    mask_2d.flat[flat_idx] = True
    return mask_2d.any(axis=0)   # (n_times,)


def collapse_conditions(tfr_dict, conditions):
    """
    Pool TFR lists across several conditions into one list per subject.
    Because each subject contributes one TFR per condition we average them
    so each subject remains one observation.
    """
    # Find subjects present in ALL requested conditions
    n_per_cond = [len(tfr_dict[c]) for c in conditions if c in tfr_dict]
    if not n_per_cond:
        return []
    n_sub = min(n_per_cond)

    collapsed = []
    for i in range(n_sub):
        # Average TFR data across conditions for subject i
        stacked = np.stack(
            [tfr_dict[c][i].data for c in conditions if c in tfr_dict],
            axis=0
        ).mean(axis=0)
        # Clone first TFR object, replace data
        tfr_mean = tfr_dict[conditions[0]][i].copy()
        tfr_mean.data = stacked
        collapsed.append(tfr_mean)
    return collapsed


# Collapse duo + trio with-feedback into one observation per subject
friend_collapsed    = collapse_conditions(friend_tfr,    INTERACTIVE_CONDITIONS)
nonfriend_collapsed = collapse_conditions(nonfriend_tfr, INTERACTIVE_CONDITIONS)

print(f"\nFriends    : {len(friend_collapsed)} observations")
print(f"Non-friends: {len(nonfriend_collapsed)} observations")

# ============================================================
# STEP 3 — PARAMETRIC TESTS: paired t-test + Cohen's d
# ============================================================
# Each subject contributes one scalar ERD value (mean over time,
# channels, and frequency band).  We pair friends and non-friends
# within the same triad (one friend vs the non-friend from the same triad).
# If counts are unequal we truncate to the smaller group.

print("\n" + "=" * 65)
print("PARAMETRIC TESTS — Paired t-test: Friends vs Non-friends")
print("=" * 65)

parametric_results = {}

for band_name, (fmin, fmax) in FREQ_BANDS.items():
    erd_f  = scalar_erd(friend_collapsed,    fmin, fmax, TMIN, TMAX)
    erd_nf = scalar_erd(nonfriend_collapsed, fmin, fmax, TMIN, TMAX)

    # Match sample sizes (pair within triad)
    n = min(len(erd_f), len(erd_nf))
    erd_f  = erd_f[:n]
    erd_nf = erd_nf[:n]

    # Paired t-test (two-tailed)
    t_stat, p_val = stats.ttest_rel(erd_f, erd_nf)

    # Effect size
    d = cohens_d(erd_f, erd_nf)

    # 95 % CI on the mean difference
    diff   = erd_f - erd_nf
    se     = stats.sem(diff)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1,
                                     loc=diff.mean(), scale=se)

    parametric_results[band_name] = {
        "erd_friends":    erd_f,
        "erd_nonfriends": erd_nf,
        "t":  t_stat,
        "p":  p_val,
        "d":  d,
        "ci": (ci_lo, ci_hi),
        "n":  n,
    }

    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
          else ("*" if p_val < 0.05 else "n.s."))

    print(f"\n  Band : {band_name.upper()}  ({fmin}–{fmax} Hz)")
    print(f"  N    : {n} pairs")
    print(f"  Mean ERD — Friends    : {erd_f.mean():.2f} %")
    print(f"  Mean ERD — Non-friends: {erd_nf.mean():.2f} %")
    print(f"  t({n-1}) = {t_stat:.3f},  p = {p_val:.4f}  {sig}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  95% CI on difference: [{ci_lo:.3f}, {ci_hi:.3f}]")

# ============================================================
# STEP 4 — NON-PARAMETRIC CLUSTER PERMUTATION TEST
# Contrasts friends vs non-friends in the full channel × time space
# Uses MNE permutation_cluster_1samp_test on the difference array
# (friends − non-friends), testing whether it differs from zero
# ============================================================

print("\n" + "=" * 65)
print("NON-PARAMETRIC CLUSTER PERMUTATION — Friends vs Non-friends")
print("=" * 65)

cluster_results = {}

for band_name, (fmin, fmax) in FREQ_BANDS.items():
    print(f"\n  Band: {band_name.upper()}  ({fmin}–{fmax} Hz)")

    data_f  = band_time_power(friend_collapsed,    fmin, fmax, TMIN, TMAX)
    data_nf = band_time_power(nonfriend_collapsed, fmin, fmax, TMIN, TMAX)

    n = min(len(data_f), len(data_nf))
    data_f  = data_f[:n]
    data_nf = data_nf[:n]

    # Contrast: (n_subjects, n_channels, n_times)
    contrast = data_f - data_nf

    T_obs, clusters, p_vals, H0 = permutation_cluster_1samp_test(
        contrast,
        n_permutations=N_PERMUTATIONS,
        threshold=2.0,    # cluster-forming T threshold (matches paper)
        tail=0,           # two-tailed
        n_jobs=1,
        verbose=False,
    )

    cluster_results[band_name] = {
        "T_obs":    T_obs,
        "clusters": clusters,
        "p_vals":   p_vals,
        "contrast": contrast,
        "n":        n,
    }

    n_ch, n_t = T_obs.shape
    times_crop = np.linspace(TMIN, TMAX, n_t)

    sig_clusters = [(c, p) for c, p in zip(clusters, p_vals) if p < 0.05]
    print(f"  Total clusters found : {len(clusters)}")
    print(f"  Significant (p<0.05) : {len(sig_clusters)}")

    for i, (c, p) in enumerate(sig_clusters):
        tm = cluster_time_mask(c, n_ch, n_t)
        t_idx = np.where(tm)[0]
        print(f"    Cluster {i+1}: p = {p:.4f}, "
              f"span ≈ {times_crop[t_idx.min()]:.2f}–"
              f"{times_crop[t_idx.max()]:.2f} s")

# ============================================================
# STEP 5 — PLOTS
# ============================================================

# --- Colour palette ---
C_FRIEND    = "#2166ac"   # blue
C_NONFRIEND = "#d6604d"   # red-orange

# Get the time axis from data (same for both bands)
_sample_tfr = friend_collapsed[0]
_t_mask     = (_sample_tfr.times >= TMIN) & (_sample_tfr.times <= TMAX)
TIMES_CROP  = _sample_tfr.times[_t_mask]

# ------------------------------------------------------------------
# FIGURE A — Violin + strip plot of scalar ERD  (parametric result)
# ------------------------------------------------------------------
fig_a, axes_a = plt.subplots(1, 2, figsize=(9, 5), sharey=False)
fig_a.suptitle(
    "ERD Magnitude: Friends vs Non-friends\n"
    "(paired t-test, time–channel average over interactive trials with feedback)",
    fontsize=11, fontweight="bold",
)

for ax, (band_name, (fmin, fmax)) in zip(axes_a, FREQ_BANDS.items()):
    res = parametric_results[band_name]
    erd_f  = res["erd_friends"]
    erd_nf = res["erd_nonfriends"]

    # Violin plots
    parts = ax.violinplot(
        [erd_f, erd_nf],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )
    parts["bodies"][0].set_facecolor(C_FRIEND)
    parts["bodies"][0].set_alpha(0.45)
    parts["bodies"][1].set_facecolor(C_NONFRIEND)
    parts["bodies"][1].set_alpha(0.45)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Individual data points (jittered)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.06, 0.06, size=len(erd_f))
    ax.scatter(0 + jitter, erd_f,  color=C_FRIEND,    s=28, alpha=0.8,
               zorder=3)
    ax.scatter(1 + jitter, erd_nf, color=C_NONFRIEND, s=28, alpha=0.8,
               zorder=3)

    # Significance bracket
    y_max  = max(erd_f.max(), erd_nf.max())
    y_top  = y_max + abs(y_max) * 0.12
    p      = res["p"]
    sig_lbl = (
        "***" if p < 0.001 else
        "**"  if p < 0.01  else
        "*"   if p < 0.05  else
        f"n.s.\np={p:.3f}"
    )
    ax.plot([0, 0, 1, 1], [y_top * 0.92, y_top, y_top, y_top * 0.92],
            lw=1.2, color="k")
    ax.text(0.5, y_top * 1.02, sig_lbl, ha="center", va="bottom", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Friends", "Non-friends"], fontsize=10)
    ax.set_ylabel("ERD (%)", fontsize=9)
    ax.set_title(
        f"{band_name.capitalize()} ({fmin}–{fmax} Hz)\n"
        f"t({res['n']-1})={res['t']:.2f}, p={res['p']:.3f}, d={res['d']:.2f}",
        fontsize=9,
    )
    ax.axhline(0, color="k", lw=0.7, ls="--")

patch_f  = mpatches.Patch(color=C_FRIEND,    alpha=0.6, label="Friends")
patch_nf = mpatches.Patch(color=C_NONFRIEND, alpha=0.6, label="Non-friends")
fig_a.legend(handles=[patch_f, patch_nf], loc="upper right", fontsize=9)
fig_a.tight_layout()
path_a = os.path.join(OUTPUT_DIR, "stat_A_violin_ttest.png")
fig_a.savefig(path_a, dpi=300, bbox_inches="tight")
plt.close(fig_a)
print(f"\nSaved: {path_a}")

# ------------------------------------------------------------------
# FIGURE B — Mean ± SEM time-courses with cluster shading
#            (one subplot per band, friends vs non-friends)
# ------------------------------------------------------------------
fig_b, axes_b = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig_b.suptitle(
    "ERD Time-courses: Friends vs Non-friends\n"
    "(mean ± SEM, channel average, interactive trials with feedback)",
    fontsize=11, fontweight="bold",
)

for ax, (band_name, (fmin, fmax)) in zip(axes_b, FREQ_BANDS.items()):
    res = cluster_results[band_name]
    n_ch_plot = res["T_obs"].shape[0]
    n_t_plot  = res["T_obs"].shape[1]

    data_f  = band_time_power(friend_collapsed,    fmin, fmax, TMIN, TMAX)
    data_nf = band_time_power(nonfriend_collapsed, fmin, fmax, TMIN, TMAX)
    n = min(len(data_f), len(data_nf))
    data_f  = data_f[:n]
    data_nf = data_nf[:n]

    # Channel-average → (n_subjects, n_times)
    ts_f  = data_f.mean(axis=1)
    ts_nf = data_nf.mean(axis=1)

    mean_f,  sem_f  = ts_f.mean(0),  ts_f.std(0)  / np.sqrt(n)
    mean_nf, sem_nf = ts_nf.mean(0), ts_nf.std(0) / np.sqrt(n)

    ax.plot(TIMES_CROP, mean_f,  color=C_FRIEND,    lw=1.8, label="Friends")
    ax.fill_between(TIMES_CROP, mean_f  - sem_f,  mean_f  + sem_f,
                    alpha=0.25, color=C_FRIEND)
    ax.plot(TIMES_CROP, mean_nf, color=C_NONFRIEND, lw=1.8, label="Non-friends")
    ax.fill_between(TIMES_CROP, mean_nf - sem_nf, mean_nf + sem_nf,
                    alpha=0.25, color=C_NONFRIEND)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Power (%)", fontsize=9)
    ax.set_title(f"{band_name.capitalize()} ({fmin}–{fmax} Hz)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)

    # Shade significant clusters
    for cluster, p_val in zip(res["clusters"], res["p_vals"]):
        if p_val < 0.05:
            tm = cluster_time_mask(cluster, n_ch_plot, n_t_plot)
            t_idx = np.where(tm)[0]
            if len(t_idx):
                ax.axvspan(TIMES_CROP[t_idx.min()], TIMES_CROP[t_idx.max()],
                           alpha=0.15, color="gray",
                           label=f"cluster p={p_val:.3f}")

axes_b[-1].set_xlabel("Time (s)", fontsize=9)
for ax in axes_b:
    ax.set_xlim(TMIN, TMAX)

fig_b.tight_layout()
path_b = os.path.join(OUTPUT_DIR, "stat_B_timecourse_cluster.png")
fig_b.savefig(path_b, dpi=300, bbox_inches="tight")
plt.close(fig_b)
print(f"Saved: {path_b}")

# ------------------------------------------------------------------
# FIGURE C — Channel-averaged T-statistic with cluster shading
# ------------------------------------------------------------------
fig_c, axes_c = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig_c.suptitle(
    "Cluster Permutation T-statistic: Friends vs Non-friends\n"
    "(channel average, threshold |T| = 2, 1000 permutations)",
    fontsize=11, fontweight="bold",
)

for ax, (band_name, (fmin, fmax)) in zip(axes_c, FREQ_BANDS.items()):
    res = cluster_results[band_name]
    T_obs = res["T_obs"]
    n_ch_plot, n_t_plot = T_obs.shape

    T_mean = T_obs.mean(axis=0)
    ax.plot(TIMES_CROP, T_mean, color="dimgray", lw=1.8)
    ax.axhline(0,   color="k",      lw=0.8, ls="--")
    ax.axhline( 2,  color="orange", lw=0.9, ls=":", label="threshold ±2")
    ax.axhline(-2,  color="orange", lw=0.9, ls=":")
    ax.set_ylabel("T value", fontsize=9)
    ax.set_title(f"{band_name.capitalize()} ({fmin}–{fmax} Hz)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)

    for cluster, p_val in zip(res["clusters"], res["p_vals"]):
        if p_val < 0.05:
            tm = cluster_time_mask(cluster, n_ch_plot, n_t_plot)
            t_idx = np.where(tm)[0]
            if len(t_idx):
                ax.axvspan(TIMES_CROP[t_idx.min()], TIMES_CROP[t_idx.max()],
                           alpha=0.20, color="orange",
                           label=f"p={p_val:.3f}")

axes_c[-1].set_xlabel("Time (s)", fontsize=9)
for ax in axes_c:
    ax.set_xlim(TMIN, TMAX)

fig_c.tight_layout()
path_c = os.path.join(OUTPUT_DIR, "stat_C_tstat_cluster.png")
fig_c.savefig(path_c, dpi=300, bbox_inches="tight")
plt.close(fig_c)
print(f"Saved: {path_c}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Band':<8} {'Test':<30} {'Statistic':<18} {'p':<10} {'Effect'}")
print("-" * 70)
for band_name in FREQ_BANDS:
    pr = parametric_results[band_name]
    cr = cluster_results[band_name]
    sig_cls = [(c, p) for c, p in zip(cr["clusters"], cr["p_vals"]) if p < 0.05]

    print(f"{band_name:<8} {'Paired t-test':<30} "
          f"t={pr['t']:+.3f}        "
          f"p={pr['p']:.4f}   "
          f"d={pr['d']:.3f}")
    print(f"{'':8} {'Cluster permutation':<30} "
          f"{len(sig_cls)} sig. cluster(s)   "
          f"—          "
          f"—")
print("=" * 70)