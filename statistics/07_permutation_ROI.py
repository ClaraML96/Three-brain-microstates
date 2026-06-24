"""07_permutation.py — single-brain ERD spatio-temporal cluster permutation test.

NON-PARAMETRIC leg of LO5: for each of four paired condition contrasts × two bands,
does baseline-relative alpha/beta power (ERD) differ across the 0–4 s push, and
where in time/space? Descriptive figures (per-channel timecourses, cluster-mass
topomap) illustrate the one inferential output — the cluster p-values.

The data load, TFR, subject-ID-matched pairing, and the cluster test all live in
`erd_cluster_stats.py`, shared with the effect-size companion 07_permutation_violin.py
so the figure can never disagree with the inference. This script owns presentation
+ the descriptive plots only.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

import erd_cluster_stats as ec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (presentation only — compute config lives in erd_cluster_stats)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\cluster_perm"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

COND_LABELS = {
    "T1P":  "Solo — With Feedback",
    "T1Pn": "Solo — No Feedback",
    "T3P":  "Trio — With Feedback",
    "T3Pn": "Trio — No Feedback",
}
COLORS = {"a": "firebrick", "b": "steelblue"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load per-subject baseline-corrected TFRs (keyed by subject id)
# ─────────────────────────────────────────────────────────────────────────────
group_tfr, info_ref = ec.load_group_tfrs()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Channel adjacency from the montage
# ─────────────────────────────────────────────────────────────────────────────
adjacency = ec.build_adjacency(info_ref)
ch_names = info_ref.ch_names
print(f"\nChannel adjacency built for: {ch_names}")
print(f"Adjacency matrix shape: {adjacency.shape}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run all contrasts × bands
# ─────────────────────────────────────────────────────────────────────────────
times_full = ec.reference_times(group_tfr)
t_mask = (times_full >= ec.PLOT_TMIN) & (times_full <= ec.PLOT_TMAX)
times_plot = times_full[t_mask]

results = {}  # (contrast_label, band_name) → dict

for contrast_label, cond_a, cond_b, contrast_title in ec.CONTRASTS:
    if cond_a not in group_tfr or cond_b not in group_tfr:
        print(f"Skipping {contrast_label}: missing condition data.")
        continue

    for band_name, (fmin, fmax) in ec.FREQ_BANDS.items():
        print(f"\n{'='*60}")
        print(f"Contrast: {contrast_title}  |  {band_name} ({fmin}–{fmax} Hz)")

        X, st_a, st_b, sids = ec.build_X(
            group_tfr, cond_a, cond_b, fmin, fmax, times_full,
        )
        n_subjects = X.shape[0]
        print(f"N matched subjects = {n_subjects},  "
              f"data shape = {X.shape}  (subj × times × channels)")

        T_obs, clusters, cluster_p, H0 = ec.run_cluster(X, adjacency)
        sig_mask, sig_clusters = ec.significant_clusters(clusters, cluster_p, T_obs)

        results[(contrast_label, band_name)] = dict(
            T_obs=T_obs,
            clusters=clusters,
            cluster_p=cluster_p,
            sig_mask=sig_mask,
            sig_clusters=sig_clusters,
            mean_a=st_a.mean(axis=0),
            sem_a=st_a.std(axis=0) / np.sqrt(n_subjects),
            mean_b=st_b.mean(axis=0),
            sem_b=st_b.std(axis=0) / np.sqrt(n_subjects),
            n_subjects=n_subjects,
            sids=sids,
            p_floor=ec.perm_p_floor(n_subjects),
            contrast_title=contrast_title,
            cond_a=cond_a,
            cond_b=cond_b,
            fmin=fmin,
            fmax=fmax,
        )

        print(f"Significant spatio-temporal clusters (p < {ec.P_ACCEPT}): "
              f"{len(sig_clusters)}   "
              f"[permutation p floor ≈ {ec.perm_p_floor(n_subjects):.2g}]")
        for i, (c, p) in enumerate(sig_clusters):
            t_idx, ch_idx = np.where(c)
            t_span = times_plot[t_idx]
            chans = [ch_names[j] for j in np.unique(ch_idx)]
            print(f"  Cluster {i+1}: {t_span.min():.3f}–{t_span.max():.3f} s  "
                  f"| channels: {chans}  | p = {p:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Per-channel ERD timecourses with cluster shading (descriptive)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_channel(contrast_label, cond_a, cond_b, contrast_title):
    bands = list(ec.FREQ_BANDS.keys())
    n_ch = len(ch_names)
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
        f"Per-channel ERD/ERS  |  gold = significant cluster (p < {ec.P_ACCEPT})",
        fontsize=12, fontweight="bold",
    )

    for col, band_name in enumerate(bands):
        key = (contrast_label, band_name)
        if key not in results:
            continue
        r = results[key]

        for row, ch in enumerate(ch_names):
            ax = axes[row, col]
            ci = ch_names.index(ch)

            ma = r["mean_a"][:, ci]
            sa = r["sem_a"][:, ci]
            mb = r["mean_b"][:, ci]
            sb = r["sem_b"][:, ci]
            sig = r["sig_mask"][:, ci]

            ax.plot(times_plot, ma, lw=1.5, color=COLORS["a"],
                    label=COND_LABELS.get(cond_a, cond_a))
            ax.fill_between(times_plot, ma - sa, ma + sa,
                            alpha=0.15, color=COLORS["a"])

            ax.plot(times_plot, mb, lw=1.5, color=COLORS["b"],
                    label=COND_LABELS.get(cond_b, cond_b))
            ax.fill_between(times_plot, mb - sb, mb + sb,
                            alpha=0.15, color=COLORS["b"])

            if sig.any():
                changes = np.diff(sig.astype(int), prepend=0, append=0)
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                for s, e in zip(starts, ends):
                    ax.axvspan(times_plot[s], times_plot[e - 1],
                               color="gold", alpha=0.40)

            ax.axhline(0,   color="k",    lw=0.6, ls="--")
            ax.axvline(0.0, color="gray", lw=0.6, ls=":")
            ax.set_xlim(ec.PLOT_TMIN, ec.PLOT_TMAX)
            ax.set_ylabel(f"{ch}\n(%)", fontsize=8)

            if row == 0:
                ax.set_title(
                    f"{band_name.capitalize()} ({r['fmin']}–{r['fmax']} Hz)\n"
                    f"N = {r['n_subjects']}",
                    fontsize=10,
                )
            if row == n_ch - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, framealpha=0.8, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"cluster_perm_perchannel_{contrast_label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Topographic cluster-mass map (descriptive)
# ─────────────────────────────────────────────────────────────────────────────
def plot_topomap(contrast_label, cond_a, cond_b, contrast_title):
    bands = list(ec.FREQ_BANDS.keys())
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

        t_mass = np.zeros(len(ch_names))
        if r["sig_mask"].any():
            for ci in range(len(ch_names)):
                ch_sig = r["sig_mask"][:, ci]
                t_mass[ci] = np.abs(r["T_obs"][ch_sig, ci]).sum()

        mne.viz.plot_topomap(
            t_mass, info_ref, axes=ax, show=False, cmap="Reds",
            vlim=(0, t_mass.max() if t_mass.max() > 0 else 1),
        )
        ax.set_title(
            f"{band_name.capitalize()} ({r['fmin']}–{r['fmax']} Hz)\n"
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
for contrast_label, cond_a, cond_b, contrast_title in ec.CONTRASTS:
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
print(f"\nN_PERMUTATIONS = {ec.N_PERMUTATIONS}  |  cluster-forming α = {ec.ALPHA_CLUSTER} "
      f"(two-tailed)  |  cluster significance p < {ec.P_ACCEPT}")
print("\n" + ec.MULTIPLE_COMPARISONS)
for (contrast_label, band_name), r in results.items():
    print(f"\n{r['contrast_title']}  |  {band_name.capitalize()} "
          f"({r['fmin']}–{r['fmax']} Hz)  |  N = {r['n_subjects']}  "
          f"|  p floor ≈ {r['p_floor']:.2g}")
    print(f"  Cond A: {COND_LABELS.get(r['cond_a'], r['cond_a'])}")
    print(f"  Cond B: {COND_LABELS.get(r['cond_b'], r['cond_b'])}")
    if r["sig_clusters"]:
        for i, (c, p) in enumerate(r["sig_clusters"]):
            t_idx, ch_idx = np.where(c)
            t_span = times_plot[t_idx]
            chans = [ch_names[j] for j in np.unique(ch_idx)]
            print(f"  ✓ Cluster {i+1}: {t_span.min():.3f}–{t_span.max():.3f} s  "
                  f"| channels: {chans}  | p = {p:.4f}")
    else:
        print(f"  ✗ No significant clusters (p < {ec.P_ACCEPT})")

print("\nDone.")
