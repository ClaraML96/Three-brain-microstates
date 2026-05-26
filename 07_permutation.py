import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import false_discovery_control   # scipy >= 1.11
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_cluster_test

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

# Friendship roles
FRIEND_PARTS    = [1, 2]
NONFRIEND_PARTS = [3]
PIDS = ["301", "302", "303", "304"]

# Morlet parameters (consistent across all scripts)
FOI      = np.linspace(1, 30, 30, dtype=int)
N_CYCLES = 3 + 0.5 * FOI
BASELINE = (-0.25, 0)
TMIN, TMAX = 0.0, 4.0

FREQ_BANDS = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

# Conditions to analyse
# Each entry: (label, [condition keys to average], row_label, col_label)
CONDITIONS = [
    ("Solo_NoFB",   ["Condition_1"],          "No Feedback",   "Solo interactions"),
    ("Solo_WithFB", ["Condition_0"],          "With Feedback", "Solo interactions"),
    ("Trio_NoFB",   ["Condition_9"],          "No Feedback",   "Triadic interactions"),
    ("Trio_WithFB", ["Condition_8"],          "With Feedback", "Triadic interactions"),
]

ALL_COND_KEYS = list({k for _, keys, _, _ in CONDITIONS for k in keys})

N_PERMUTATIONS = 1000
T_THRESHOLD    = 2.0

# Colours matching the paper figure
C_FRIEND    = "#d6604d"   # orange  (Friends)
C_NONFRIEND = "#4393c3"   # blue    (Non-friends)

# ============================================================
# STEP 1 — Load TFRs
# ============================================================
friend_tfr    = {c: [] for c in ALL_COND_KEYS}
nonfriend_tfr = {c: [] for c in ALL_COND_KEYS}

for pid in PIDS:
    for part in [1, 2, 3]:
        epoch_file = os.path.join(
            DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif"
        )
        if not os.path.exists(epoch_file):
            print(f"  Missing: {epoch_file}")
            continue

        role = "friend" if part in FRIEND_PARTS else "nonfriend"
        print(f"Loading {pid} part {part}  [{role}]")
        epochs = mne.read_epochs(epoch_file, preload=True)

        for cond in ALL_COND_KEYS:
            if cond not in epochs.event_id:
                print(f"  '{cond}' not found — skipping")
                continue

            tfr = epochs[cond].compute_tfr(
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
                friend_tfr[cond].append(tfr_avg)
            else:
                nonfriend_tfr[cond].append(tfr_avg)

print("\nAll TFRs loaded.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def band_time_power(tfr_list, fmin, fmax, tmin, tmax):
    """(n_subjects, n_channels, n_times) averaged over frequency band."""
    out = []
    for tfr in tfr_list:
        f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
        t_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
        out.append(tfr.data[:, f_mask, :][:, :, t_mask].mean(axis=1))
    return np.array(out)


def scalar_erd(tfr_list, fmin, fmax, tmin, tmax):
    """Single scalar ERD value per subject (mean over ch, time, freq)."""
    arr = band_time_power(tfr_list, fmin, fmax, tmin, tmax)
    return arr.mean(axis=(1, 2))


def cohens_d_ind(a, b):
    """Independent samples Cohen's d (pooled SD)."""
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / (pooled_sd + 1e-12)


def cluster_time_mask(cluster, n_ch, n_t):
    """Convert MNE flat-index cluster to boolean time mask."""
    flat_idx = cluster[0]
    mask_2d  = np.zeros((n_ch, n_t), dtype=bool)
    mask_2d.flat[flat_idx] = True
    return mask_2d.any(axis=0)


def collapse_conditions(tfr_dict, cond_keys):
    """
    Average TFRs across several condition keys, one observation per subject.
    Subjects must be present in all keys.
    """
    available = [tfr_dict[k] for k in cond_keys if k in tfr_dict and tfr_dict[k]]
    if not available:
        return []
    n_sub = min(len(lst) for lst in available)
    collapsed = []
    for i in range(n_sub):
        stacked = np.stack(
            [tfr_dict[k][i].data for k in cond_keys if k in tfr_dict],
            axis=0,
        ).mean(axis=0)
        tfr_mean      = tfr_dict[cond_keys[0]][i].copy()
        tfr_mean.data = stacked
        collapsed.append(tfr_mean)
    return collapsed


# ============================================================
# STEP 2 — Run cluster permutation for every condition × band
# ============================================================
results = {}   # keyed by (cond_label, band_name)

all_p_scalar = []   # collect scalar t-test p-values for FDR

for cond_label, cond_keys, row_lbl, col_lbl in CONDITIONS:
    f_col  = collapse_conditions(friend_tfr,    cond_keys)
    nf_col = collapse_conditions(nonfriend_tfr, cond_keys)
    n_f  = len(f_col)
    n_nf = len(nf_col)

    if n_f < 1 or n_nf < 1:
        print(f"\n  [{cond_label}] Not enough data (n_friend={n_f}, n_nonfriend={n_nf}), skipping.")
        continue

    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        key = (cond_label, band_name)
        print(f"\n  [{cond_label}] [{band_name}]  n_friend={n_f}, n_nonfriend={n_nf}")

        # Scalar ERD for violin plot + independent samples t-test
        erd_f  = scalar_erd(f_col,  fmin, fmax, TMIN, TMAX)
        erd_nf = scalar_erd(nf_col, fmin, fmax, TMIN, TMAX)
        t_stat, p_scalar = stats.ttest_ind(erd_f, erd_nf, equal_var=False)
        cohens_d = cohens_d_ind(erd_f, erd_nf) 

        # Full (n_sub, n_ch, n_t) arrays for cluster test
        data_f  = band_time_power(f_col,  fmin, fmax, TMIN, TMAX)
        data_nf = band_time_power(nf_col, fmin, fmax, TMIN, TMAX)
        
        T_obs, clusters, p_vals, _ = permutation_cluster_test(
            [data_f, data_nf],          # list of two groups, unequal n is fine
            n_permutations=N_PERMUTATIONS,
            threshold=T_THRESHOLD,
            tail=0,
            n_jobs=1,
            verbose=False,
        )

        n_ch, n_t = T_obs.shape
        times_crop = np.linspace(TMIN, TMAX, n_t)

        results[key] = dict(
            cond_label  = cond_label,
            band_name   = band_name,
            row_lbl     = row_lbl,
            col_lbl     = col_lbl,
            n_f         = n_f,
            n_nf        = n_nf,
            erd_f       = erd_f,
            erd_nf      = erd_nf,
            t_scalar    = t_stat,
            p_scalar    = p_scalar,
            cohens_d    = cohens_d,
            T_obs       = T_obs,
            clusters    = clusters,
            p_vals      = p_vals,
            times_crop  = times_crop,
            data_f      = data_f,
            data_nf     = data_nf,
        )
        all_p_scalar.append((key, p_scalar))
        print(f"    independent t: t={t_stat:.3f}, p={p_scalar:.4f}, Cohen's d={cohens_d:.3f}")
        sig = sum(p < 0.05 for p in p_vals)
        print(f"    clusters: {len(clusters)} total, {sig} significant")

# ============================================================
# STEP 3 — FDR correction across all condition × band p-values
# ============================================================
if all_p_scalar:
    keys_ordered = [k for k, _ in all_p_scalar]
    raw_p        = np.array([p for _, p in all_p_scalar])
    try:
        fdr_p = false_discovery_control(raw_p, method="bh")
    except Exception:
        # scipy < 1.11 fallback — Benjamini-Hochberg manually
        n_tests = len(raw_p)
        order   = np.argsort(raw_p)
        fdr_p   = np.empty(n_tests)
        fdr_p[order] = raw_p[order] * n_tests / (np.arange(n_tests) + 1)
        fdr_p = np.minimum.accumulate(fdr_p[::-1])[::-1]
        fdr_p = np.minimum(fdr_p, 1.0)

    for k, pf in zip(keys_ordered, fdr_p):
        results[k]["p_fdr"] = pf
        print(f"  {k}: raw p={results[k]['p_scalar']:.4f}, FDR p={pf:.4f}")

# ============================================================
# STEP 4 — PLOTS  (one figure per frequency band)
#          Layout mirrors Fig. 2:  rows=feedback, cols=condition
# ============================================================

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return f"p={p:.3f}"


def half_violin(ax, data, pos, color, side="left", alpha=0.45, bw=0.15):
    """Draw a one-sided (half) violin at x=pos."""
    from scipy.stats import gaussian_kde
    if len(data) < 2:
        return
    kde  = gaussian_kde(data, bw_method=bw)
    ys   = np.linspace(data.min() - 0.05, data.max() + 0.05, 200)
    xs   = kde(ys)
    xs   = xs / xs.max() * 0.35   # normalise width
    if side == "left":
        ax.fill_betweenx(ys, pos - xs, pos, color=color, alpha=alpha)
    else:
        ax.fill_betweenx(ys, pos, pos + xs, color=color, alpha=alpha)


ROW_LABELS = ["No Feedback", "With Feedback"]
COL_LABELS_SOLO = ["Solo interactions"]
COL_LABELS_TRIO = ["Triadic interactions"]

# Map condition label → (row_idx, col_idx)
COND_GRID = {
    "Solo_NoFB":   (0, 0),
    "Solo_WithFB": (1, 0),
    "Trio_NoFB":   (0, 1),
    "Trio_WithFB": (1, 1),
}
COL_TITLES = ["Solo interactions\n", "Triadic interactions\n"]

rng = np.random.default_rng(42)

for band_name, (fmin, fmax) in FREQ_BANDS.items():
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9),
        sharey="row",
        gridspec_kw={"hspace": 0.42, "wspace": 0.18},
    )
    fig.suptitle(
        f"Friends vs Non-friends — {band_name.capitalize()} band ",
        # f"({fmin}–{fmax} Hz)\n"
        # r"ERD (% change from baseline), cluster permutation, FDR-corrected",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # Column headers
    for col, title in enumerate(COL_TITLES):
        color = "#6a0dad" if col == 0 else "#228b22"   
        axes[0, col].set_title(title, fontsize=11, fontweight="bold",
                               color=color, pad=8)

    # Row labels (y-axis)
    for row, lbl in enumerate(ROW_LABELS):
        axes[row, 0].set_ylabel(
            f"{lbl}", fontsize=9, labelpad=6
        )

    for cond_label, _, _, _ in CONDITIONS:
        row_idx, col_idx = COND_GRID[cond_label]
        key = (cond_label, band_name)
        if key not in results:
            continue

        res = results[key]
        ax  = axes[row_idx, col_idx]

        erd_f  = res["erd_f"]
        erd_nf = res["erd_nf"]
        p_use  = res.get("p_fdr", res["p_scalar"])   # prefer FDR

        # Half violins (split at centre)
        half_violin(ax, erd_f,  0.5, C_FRIEND,    side="left")
        half_violin(ax, erd_nf, 0.5, C_NONFRIEND, side="right")

        # Jittered strip plots
        jf  = rng.uniform(-0.15, -0.02, size=len(erd_f))
        jnf = rng.uniform( 0.02,  0.15, size=len(erd_nf))
        ax.scatter(0.5 + jf,  erd_f,  color=C_FRIEND,    s=22,
                   alpha=0.80, zorder=3, edgecolors="none")
        ax.scatter(0.5 + jnf, erd_nf, color=C_NONFRIEND, s=22,
                   alpha=0.80, zorder=3, edgecolors="none")

        # Mean ± SEM markers (black diamonds, matching Fig 2)
        for val_arr, xpos in [(erd_f, 0.34), (erd_nf, 0.66)]:
            mean_v = val_arr.mean()
            sem_v  = val_arr.std(ddof=1) / np.sqrt(len(val_arr))
            ax.errorbar(xpos, mean_v, yerr=sem_v,
                        fmt="D", color="black", ms=5, capsize=3, lw=1.4,
                        zorder=5)

        # Reference line at 0
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

        # Significance bracket
        y_vals  = np.concatenate([erd_f, erd_nf])
        y_span  = y_vals.max() - y_vals.min()
        y_top   = y_vals.max() + y_span * 0.14
        slabel  = sig_label(p_use)
        ax.plot([0.34, 0.34, 0.66, 0.66],
                [y_top * 0.93, y_top, y_top, y_top * 0.93],
                lw=1.2, color="k")
        ax.text(0.50, y_top * 1.015, slabel,
                ha="center", va="bottom", fontsize=10)

        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)

        # Annotate n
        ax.text(0.98, 0.02, f"N = {res['n_f'] + res['n_nf']}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="dimgray")

    # Shared legend
    patch_f  = mpatches.Patch(color=C_FRIEND,    alpha=0.75, label="Friends")
    patch_nf = mpatches.Patch(color=C_NONFRIEND, alpha=0.75, label="Non-friend")
    fig.legend(handles=[patch_f, patch_nf],
               loc="upper right", fontsize=9,
               framealpha=0.85, edgecolor="none",
               bbox_to_anchor=(1.0, 0.98))

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR,
                            f"cluster_friends_vs_nonfriends_{band_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

# ============================================================
# STEP 5 — SUMMARY TABLE
# ============================================================
print("\n" + "=" * 100)
print(f"{'Condition':<18} {'Band':<7} {'n_f':>3} {'n_nf':>3}  "
      f"{'t':>7}  {'p_raw':>7}  {'p_FDR':>7}  "
      f"{'Cohen\'s d':>9}  {'Sig.clusters':>14}")
print("-" * 100)
for cond_label, _, _, _ in CONDITIONS:
    for band_name in FREQ_BANDS:
        key = (cond_label, band_name)
        if key not in results:
            continue
        res     = results[key]
        sig_cls = sum(p < 0.05 for p in res["p_vals"])
        p_fdr   = res.get("p_fdr", float("nan"))
        print(
            f"{cond_label:<18} {band_name:<7} {res['n_f']:>3} {res['n_nf']:>3}  "
            f"{res['t_scalar']:>+7.3f}  "
            f"{res['p_scalar']:>7.4f}  "
            f"{p_fdr:>7.4f}  "
            f"{res['cohens_d']:>+9.3f}  "
            f"{sig_cls:>14}"
        )
print("=" * 100)