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


def connectograms_for(analysis, band, cell_a, cell_b):
    """Draw solo/trio/contrast connectograms for one analysis × band.
    cell_a and cell_b are the two condition names whose grand-mean PLV matrices
    were saved as plv_<cell_a>_<band>.npy and plv_<cell_b>_<band>.npy."""
    subdir = os.path.join(V4_OUTPUT_DIR, analysis)
    diff_f = os.path.join(subdir, f"diff_{band}.npy")
    sig_f  = os.path.join(subdir, f"sigmask_{band}.npy")
    solo_f = os.path.join(subdir, f"plv_{cell_b}_{band}.npy")   # cell_b = T1P / T1Pn / etc.
    trio_f = os.path.join(subdir, f"plv_{cell_a}_{band}.npy")   # cell_a = T3P / T3Pn / etc.
    if not os.path.isfile(diff_f):
        return
    mean_diff = np.load(diff_f).mean(axis=0)          # (n_ch, n_ch) mean ΔPLV
    mean_solo = np.load(solo_f)    # (n_ch, n_ch) grand-mean PLV for Solo condition
    mean_trio = np.load(trio_f)    # (n_ch, n_ch) grand-mean PLV for Trio condition
    if mean_diff.shape != (n_ch, n_ch):
        raise ValueError(
            f"{analysis}/{band}: matrix {mean_diff.shape} ≠ montage ({n_ch}). "
            "Channel set mismatch between v4 and the montage file.")

    absvals = _off_diag_abs(mean_diff)
    tag = f"{analysis}_{band}"

    # Figure 1 — raw PLV for Solo (sequential: blue = low PLV, red = high PLV).
    # Threshold is the FULL_PCT percentile of the Solo matrix itself (not the
    # contrast), so the cutoff reflects the actual PLV scale of this condition.
    thr_solo = float(np.percentile(_off_diag_abs(mean_solo), FULL_PCT))
    print(f"  {tag}: solo PLV connectogram (PLV ≥ {thr_solo:.4f}, {FULL_PCT:.0f}th pct)")
    ax = viz_2D_topomap_inter(epo, epo, mean_solo, threshold=thr_solo, lab=False)
    _save(ax, f"connectogram_{tag}_solo.png")

    # Figure 2 — raw PLV for Trio (same sequential colour logic as Solo).
    thr_trio = float(np.percentile(_off_diag_abs(mean_trio), FULL_PCT))
    print(f"  {tag}: trio PLV connectogram (PLV ≥ {thr_trio:.4f}, {FULL_PCT:.0f}th pct)")
    ax = viz_2D_topomap_inter(epo, epo, mean_trio, threshold=thr_trio, lab=False)
    _save(ax, f"connectogram_{tag}_trio.png")

    # Figure 3 — contrast Trio − Solo (diverging: blue = Trio < Solo, red = Trio > Solo).
    # Threshold is based on |ΔPLV| exactly as the old Variant 1 was, so the
    # same FULL_PCT percentile governs which contrast links are drawn.
    thr_contrast = float(np.percentile(absvals, FULL_PCT))
    print(f"  {tag}: contrast connectogram (|ΔPLV| ≥ {thr_contrast:.4f}, {FULL_PCT:.0f}th pct)")
    ax = viz_2D_topomap_inter(epo, epo, mean_diff, threshold=thr_contrast, lab=False)
    _save(ax, f"connectogram_{tag}_contrast.png")


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
# Build a lookup so the connectogram renderer knows which two cells each
# analysis contrasts, without duplicating the ANALYSES definition.
CELL_LOOKUP = {
    "grpsize_in_fb":     ("T3P",  "T1P"),
    "grpsize_in_nofb":   ("T3Pn", "T1Pn"),
    "feedback_in_solo":  ("T1P",  "T1Pn"),
    "feedback_in_triad": ("T3P",  "T3Pn"),
    "interaction":       ("T3P",  "T1P"),   # first two cells; descriptive only
}

for analysis in analyses:
    print(f"ANALYSIS: {analysis}")
    cell_a, cell_b = CELL_LOOKUP.get(analysis, ("T3P", "T1P"))
    for band in BANDS:
        connectograms_for(analysis, band, cell_a, cell_b)

print("Done. Connectograms under:", CONNECTOGRAM_DIR)
print("  per analysis × band: *_solo.png (raw PLV Solo), *_trio.png (raw PLV Trio),")
print("  *_contrast.png (Trio − Solo diverging). No cluster-guard needed.")
