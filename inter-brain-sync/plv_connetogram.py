"""
plv_connectogram.py — HyPyP two-head connectogram for the 2×2 inter-brain PLV.

Supervisor #2 ("plot the connections from head A and head B — HyPyP has a way of
visualizing that") and #3 ("point out the clusters"). The 64×64 matrix figures
(tmap/diffmap) are NOT a topography; the connectogram is the readable spatial view:
two head outlines facing each other with Bézier links between coupled electrode
pairs.

STANDALONE RENDERER — decoupled from the compute. It reads the per-band matrices
that `plv_condition_contrast_v4_2x2.py` saves per analysis subfolder
(`diff_<band>.npy`, `tmap_<band>.npy`, `sigmask_<band>.npy`) and draws three
variants per analysis × band. It does NOT recompute PLV, so the heavy pipeline and
the (fiddly, version-sensitive) HyPyP viz stack stay separated.

VENV: runs in the HyPyP venv (hypyp==0.6.0, NumPy ≥2.2 — see
`hypyp-crossval/requirements-hypyp.txt`), NOT the main MNE 1.6.1 pipeline venv.
The .npy matrices are plain arrays, readable from either; only `hypyp.viz` forces
the HyPyP venv here. Run v4 first (main venv) to produce the .npy, then this.

THE THREE VARIANTS (per analysis × band):
  1. full        — all links with |ΔPLV| above a low percentile. Honest "is there
                   broad structure" view; for a diffuse cluster this is a hairball
                   (that's the truthful picture, cf. v3's 329-pair cluster).
  2. top         — only the strongest |ΔPLV| links (high percentile). For
                   legibility ONLY — do NOT let it imply more focality than the
                   diffuse cluster actually has (supervisor #3 caveat).
  3. cluster     — links restricted to the significant-cluster pairs (sigmask),
                   coloured by ΔPLV. This is "point out the clusters". GUARDED:
                   skipped when the analysis returned no significant cluster
                   (sigmask all-False) — no empty connectogram is drawn.

HyPyP viz API (verified against hypyp 0.6.0 source):
  viz_2D_topomap_inter(epo1, epo2, C, threshold=0.95, steps=10, lab=False, …)
    • C : (n_ch_head_A, n_ch_head_B). Our matrices are rows=head-A, cols=head-B.
    • threshold : ABSOLUTE cutoff — draws a link where C ≥ threshold (red, Reds)
      or C ≤ −threshold (blue, Blues_r); link weight scales with |C|.
    • creates its own figure, calls plt.show(), returns the Axes. We grab
      ax.get_figure() and savefig (Agg backend → show() is a no-op).
    • reads epo.ch_names, epo.info['bads'], epo.info['chs'][i]['loc'][:3] — so a
      single montage suffices; we pass the same Epochs as both heads.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: viz_2D_topomap_inter calls plt.show()
import matplotlib.pyplot as plt
import mne

from hypyp.viz import viz_2D_topomap_inter

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  EDIT PATHS FOR YOUR MACHINE
# ═════════════════════════════════════════════════════════════════════════════
# Where v4 wrote its per-analysis subfolders (must match v4's OUTPUT_DIR).
V4_OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv_2x2_v4"
)
# Any one preprocessed epoch file — used only for the 64-channel montage geometry.
MONTAGE_EPOCH_FILE = sorted(glob.glob(os.path.join(
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData", "*_FG_preprocessed-epo.fif")))[0]

CONNECTOGRAM_DIR = os.path.join(V4_OUTPUT_DIR, "connectograms")
os.makedirs(CONNECTOGRAM_DIR, exist_ok=True)

BANDS = ["alpha", "beta"]

# Percentile cutoffs (on |ΔPLV| over all off-diagonal pairs) for the two
# descriptive variants. Tune to taste once the real ΔPLV scale is known.
FULL_PCT = 90.0     # variant 1: broad structure (hairball-honest)
TOP_PCT  = 99.0     # variant 2: strongest links only (legibility)

# ═════════════════════════════════════════════════════════════════════════════
# Montage (one Epochs object serves as both heads — single shared montage)
# ═════════════════════════════════════════════════════════════════════════════
print(f"Loading montage from {os.path.basename(MONTAGE_EPOCH_FILE)} …")
epo = mne.read_epochs(MONTAGE_EPOCH_FILE, preload=True, verbose=False)
epo.pick("eeg")
n_ch = len(epo.ch_names)
print(f"  {n_ch} EEG channels.\n")


def _save(ax, fname):
    """viz_2D_topomap_inter returns an Axes and calls plt.show(); persist its fig."""
    fig = ax.get_figure()
    path = os.path.join(CONNECTOGRAM_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


def _off_diag_abs(M):
    """|values| on the off-diagonal only (the head-A==head-B self-pairs are not
    inter-brain links and would dominate the percentile)."""
    mask = ~np.eye(M.shape[0], dtype=bool)
    return np.abs(M[mask])


def connectograms_for(analysis, band):
    """Draw the three connectogram variants for one analysis × band, if the
    matrices exist."""
    subdir = os.path.join(V4_OUTPUT_DIR, analysis)
    diff_f = os.path.join(subdir, f"diff_{band}.npy")
    sig_f  = os.path.join(subdir, f"sigmask_{band}.npy")
    if not os.path.isfile(diff_f):
        return
    mean_diff = np.load(diff_f).mean(axis=0)          # (n_ch, n_ch) mean ΔPLV
    if mean_diff.shape != (n_ch, n_ch):
        raise ValueError(
            f"{analysis}/{band}: matrix {mean_diff.shape} ≠ montage ({n_ch}). "
            "Channel set mismatch between v4 and the montage file.")

    absvals = _off_diag_abs(mean_diff)
    tag = f"{analysis}_{band}"

    # Variant 1 — full / broad structure.
    thr_full = float(np.percentile(absvals, FULL_PCT))
    print(f"  {tag}: full connectogram (|ΔPLV| ≥ {thr_full:.4f}, {FULL_PCT:.0f}th pct)")
    ax = viz_2D_topomap_inter(epo, epo, mean_diff, threshold=thr_full, lab=False)
    _save(ax, f"connectogram_{tag}_full.png")

    # Variant 2 — top links only (legibility).
    thr_top = float(np.percentile(absvals, TOP_PCT))
    print(f"  {tag}: top connectogram (|ΔPLV| ≥ {thr_top:.4f}, {TOP_PCT:.0f}th pct)")
    ax = viz_2D_topomap_inter(epo, epo, mean_diff, threshold=thr_top, lab=False)
    _save(ax, f"connectogram_{tag}_top.png")

    # Variant 3 — significant-cluster pairs only. GUARDED.
    if os.path.isfile(sig_f):
        sigmask = np.load(sig_f).astype(bool)
        if sigmask.any():
            C = np.where(sigmask, mean_diff, np.nan)      # non-cluster → NaN (skipped)
            sig_abs = np.abs(mean_diff[sigmask])
            thr_clu = float(sig_abs.min()) * 0.999        # ensure all sig pairs pass
            print(f"  {tag}: cluster connectogram ({int(sigmask.sum())} sig pairs)")
            ax = viz_2D_topomap_inter(epo, epo, C, threshold=thr_clu, lab=False)
            _save(ax, f"connectogram_{tag}_cluster.png")
        else:
            print(f"  {tag}: no significant cluster — cluster connectogram skipped "
                  f"(null reported in text, no empty figure).")


# ═════════════════════════════════════════════════════════════════════════════
# Iterate every analysis subfolder v4 produced
# ═════════════════════════════════════════════════════════════════════════════
analyses = sorted(
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(os.path.join(V4_OUTPUT_DIR, "*", "diff_alpha.npy"))
)
if not analyses:
    raise SystemExit(
        f"No analysis subfolders with diff_alpha.npy under {V4_OUTPUT_DIR}. "
        "Run plv_condition_contrast_v4_2x2.py first.")

print(f"Rendering connectograms for analyses: {analyses}\n")
for analysis in analyses:
    print(f"ANALYSIS: {analysis}")
    for band in BANDS:
        connectograms_for(analysis, band)
    print()

print("Done. Connectograms under:", CONNECTOGRAM_DIR)
print("  per analysis × band: *_full.png, *_top.png, and *_cluster.png where a")
print("  significant cluster exists (guarded — no empty connectogram otherwise).")
