"""07_permutation_wholescalp.py — whole-scalp single-brain ERD cluster permutation test.

WHOLE-SCALP VARIANT of 07_permutation.py, built in response to supervisor
feedback (#5, 2026-06-24): "use all channels — the current violin is
hypothesis-driven." The original ROI-confirmatory leg ran the cluster test over a
4-channel a-priori ROI (C3, O1, Oz, O2, from the parent paper) and reported a
per-subject effect size on that ROI (the violin). The criticism is that the ROI is
a cherry-pick; the honest object is the *exploratory* whole-scalp test, which is
already FWER-corrected over space by construction.

This variant therefore:
  • loads ALL 64 EEG channels (ec.load_group_tfrs(channels=None)) and builds the
    full-montage adjacency, so the spatio-temporal cluster test controls the
    family-wise error across the whole scalp AND time within each contrast x band;
  • drops the violin / a-priori-window magnitude companion entirely (decision
    B=wait, 2026-06-24): no per-subject magnitude figure for now;
  • presents results with a GUARDED cluster topomap — the pattern ported from the
    PLV script's plot_participation(): when a contrast returns a significant
    cluster, draw a plot_topomap of the cluster channels (per-channel summed |T|,
    significant channels masked/highlighted); when it returns NONE, omit the figure
    and report the null in text. An empty/all-grey map would otherwise read as "we
    looked and found structure" when the honest result is null.

Lineage: this is a SEPARATE variant, not an in-place edit. 07_permutation.py and
07_permutation_violin.py keep their v3 (4-channel ROI) behaviour exactly — the only
shared-module change is the additive `channels=None` branch in load_group_tfrs,
whose default leaves the ROI scripts untouched.

The data load, TFR, subject-ID-matched pairing, and cluster test all live in
erd_cluster_stats.py. This script owns presentation only.

PARENT-PAPER DIVERGENCE (state, don't bury): the parent paper analysed the
4-channel ROI. Whole-scalp departs from that on purpose — it trades the ROI's clean
per-subject magnitude for an unbiased spatial search. Flag the divergence in the
write-up.
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
    r"\Human Centeret Artificial Intelligence\Thesis\figures\cluster_perm_wholescalp"
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
# STEP 1 — Load per-subject baseline-corrected TFRs over ALL EEG channels
# ─────────────────────────────────────────────────────────────────────────────
group_tfr, info_ref = ec.load_group_tfrs(channels=None)   # None → whole scalp

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Whole-montage channel adjacency
# ─────────────────────────────────────────────────────────────────────────────
adjacency = ec.build_adjacency(info_ref)
ch_names = info_ref.ch_names
print(f"\nWHOLE-SCALP cluster test over {len(ch_names)} channels")
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
                  f"| {len(chans)} channels  | p = {p:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for the presentation layer
# ─────────────────────────────────────────────────────────────────────────────
def sig_channel_mask(r):
    """Boolean (n_ch,): channels that sit in ANY significant cluster timepoint."""
    return r["sig_mask"].any(axis=0)


def per_channel_tmass(r):
    """(n_ch,): per channel, summed |T_obs| over its significant timepoints.

    Zero for channels not in any significant cluster. This is the single-head
    analogue of plot_participation()'s degree map: 'how much cluster mass sits on
    this electrode'.
    """
    t_mass = np.zeros(len(ch_names))
    if r["sig_mask"].any():
        for ci in range(len(ch_names)):
            ch_sig = r["sig_mask"][:, ci]
            t_mass[ci] = np.abs(r["T_obs"][ch_sig, ci]).sum()
    return t_mass


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GUARDED cluster topomap (ported plot_participation pattern)
#          One figure per contrast, alpha + beta side by side.
#          Skips a band-cell with no significant cluster; skips the whole figure
#          if NEITHER band has one (honest null → reported in text, not a grey map).
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_topomap(contrast_label, contrast_title):
    bands = list(ec.FREQ_BANDS.keys())
    have_any = any(
        results.get((contrast_label, b), {}).get("sig_clusters")
        for b in bands
    )
    if not have_any:
        print(f"[topomap skipped] {contrast_title}: no significant cluster in any "
              f"band — null reported in text, no empty map drawn.")
        return

    n_bands = len(bands)
    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        f"Whole-scalp cluster T-mass: {contrast_title}\n"
        f"(Σ|T| within significant cluster per channel; ● = cluster channel)",
        fontsize=11, fontweight="bold",
    )

    for ax, band_name in zip(axes, bands):
        key = (contrast_label, band_name)
        r = results.get(key)
        if r is None or not r["sig_clusters"]:
            # Guard at the band level too: don't draw a flat map for a null band.
            ax.axis("off")
            ax.set_title(
                f"{band_name.capitalize()} — no cluster",
                fontsize=10,
            )
            continue

        t_mass = per_channel_tmass(r)
        mask = sig_channel_mask(r)
        mne.viz.plot_topomap(
            t_mass, info_ref, axes=ax, show=False, cmap="Reds",
            vlim=(0, t_mass.max()),
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="k",
                             markeredgecolor="k", markersize=4),
        )
        n_sig = int(mask.sum())
        pmin = min(p for _, p in r["sig_clusters"])
        ax.set_title(
            f"{band_name.capitalize()} ({r['fmin']}–{r['fmax']} Hz)\n"
            f"{n_sig}/{len(ch_names)} ch in cluster · p={pmin:.4f} · N={r['n_subjects']}",
            fontsize=9,
        )

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"wholescalp_topomap_{contrast_label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GUARDED cluster-mean timecourse (descriptive temporal story)
#          Mean ERD over the channels in the significant cluster, A vs B, with the
#          cluster time-extent shaded. Replaces the 4-channel per-channel grid,
#          which is unreadable at 64 channels. Purely descriptive illustration of
#          an already-significant cluster (no new statistic computed off it).
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_mean_timecourse(contrast_label, cond_a, cond_b, contrast_title):
    bands = list(ec.FREQ_BANDS.keys())
    have_any = any(
        results.get((contrast_label, b), {}).get("sig_clusters")
        for b in bands
    )
    if not have_any:
        return  # null already reported by the topomap guard

    n_bands = len(bands)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 3.6), sharex=True)
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        f"Cluster-mean ERD/ERS: {contrast_title}\n"
        f"mean over significant-cluster channels  |  gold = cluster time-extent",
        fontsize=11, fontweight="bold",
    )

    for ax, band_name in zip(axes, bands):
        key = (contrast_label, band_name)
        r = results.get(key)
        if r is None or not r["sig_clusters"]:
            ax.axis("off")
            ax.set_title(f"{band_name.capitalize()} — no cluster", fontsize=10)
            continue

        ch_in = sig_channel_mask(r)
        ma = r["mean_a"][:, ch_in].mean(axis=1)
        mb = r["mean_b"][:, ch_in].mean(axis=1)
        # SEM across the cluster channels' subject-SEMs (descriptive band only).
        sa = r["sem_a"][:, ch_in].mean(axis=1)
        sb = r["sem_b"][:, ch_in].mean(axis=1)

        ax.plot(times_plot, ma, lw=1.6, color=COLORS["a"],
                label=COND_LABELS.get(cond_a, cond_a))
        ax.fill_between(times_plot, ma - sa, ma + sa, alpha=0.15, color=COLORS["a"])
        ax.plot(times_plot, mb, lw=1.6, color=COLORS["b"],
                label=COND_LABELS.get(cond_b, cond_b))
        ax.fill_between(times_plot, mb - sb, mb + sb, alpha=0.15, color=COLORS["b"])

        # Shade the union time-extent of the significant cluster(s).
        sig_t = r["sig_mask"].any(axis=1)
        if sig_t.any():
            changes = np.diff(sig_t.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                ax.axvspan(times_plot[s], times_plot[e - 1], color="gold", alpha=0.35)

        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0.0, color="gray", lw=0.6, ls=":")
        ax.set_xlim(ec.PLOT_TMIN, ec.PLOT_TMAX)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("ERD/ERS (%)", fontsize=9)
        ax.set_title(
            f"{band_name.capitalize()} ({r['fmin']}–{r['fmax']} Hz)  |  "
            f"{int(ch_in.sum())} cluster ch",
            fontsize=10,
        )
        ax.legend(fontsize=7, framealpha=0.8, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"wholescalp_clustermean_{contrast_label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Run plotting for every contrast
# ─────────────────────────────────────────────────────────────────────────────
for contrast_label, cond_a, cond_b, contrast_title in ec.CONTRASTS:
    if (contrast_label, "alpha") not in results:
        continue
    plot_cluster_topomap(contrast_label, contrast_title)
    plot_cluster_mean_timecourse(contrast_label, cond_a, cond_b, contrast_title)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WHOLE-SCALP SPATIO-TEMPORAL CLUSTER PERMUTATION TEST — RESULTS SUMMARY")
print("=" * 70)
print(f"\nChannels = {len(ch_names)} (all EEG)  |  "
      f"N_PERMUTATIONS = {ec.N_PERMUTATIONS}  |  "
      f"cluster-forming α = {ec.ALPHA_CLUSTER} (two-tailed)  |  "
      f"cluster significance p < {ec.P_ACCEPT}")
print("\nNo violin / a-priori-window magnitude figure (decision B=wait, 2026-06-24).")
print("ROI divergence: parent paper used the 4-channel ROI; this is whole-scalp.\n")
print(ec.MULTIPLE_COMPARISONS)
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
                  f"| {len(chans)} ch: {chans}  | p = {p:.4f}")
    else:
        print(f"  ✗ No significant clusters (p < {ec.P_ACCEPT})")

print("\nDone.")
