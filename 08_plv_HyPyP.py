"""
hypyp_adjacency_crossval.py — validate HyPyP's inter-brain neighbourhood == our A⊗A
═══════════════════════════════════════════════════════════════════════════════
SCOPE: ADJACENCY ONLY. This is the one part of HyPyP that exactly reproduces a
piece of `plv_matrix_pipeline.py`. It cross-checks that HyPyP's
`metaconn_matrix_2brains` (the Dumas a/b/c inter-brain neighbourhood) equals our
hand-derived `A ⊗ A` pair adjacency on the real 64-channel montage. If they
match, our Kronecker derivation (plv_matrix_pipeline.py §6) is independently
validated against the published reference implementation.

WHY ONLY ADJACENCY (the other two HyPyP layers are NOT used — see
`../hypyp-decision.md` §"Source verification", read from HyPyP 0.6.0 on 2026-06-15):

  • MEASURE — `compute_sync(mode='plv')` computes OVER-TIME PLV (average over time
    samples within an epoch, then over epochs; `hypyp/sync/plv.py`:
    `con = abs(dphi)/n_samp`). Our `matrix_plv` computes ACROSS-TRIALS PLV
    (average over trials at each time, then over time). These are DIFFERENT
    estimators and cannot be reconciled by feeding compute_sync differently, so
    HyPyP cannot reproduce our PLV numbers. Whether to run the over-time
    estimator as a complementary secondary analysis is deferred to the supervisor
    (the across-trial vs across-time question the matrix proposal §"Open" raises).

  • INFERENCE — `statscluster(test='rel ttest')` routes through MNE's
    `permutation_cluster_test` (independent-groups label shuffle), NOT
    `permutation_cluster_1samp_test` (within-subject sign-flip). That is the wrong
    null for our paired-by-triad design. Inference stays on our sign-flip.

  • BANDS — `metaconn_matrix_2brains` ALSO returns `metaconn_freq`, which CHAINS
    adjacent bands (band i ~ i±1). We use ONLY the base `metaconn` (single
    frequency) and keep our own block-diagonal band stacking. Do not use
    `metaconn_freq`.

WHAT THIS SCRIPT NEEDS: a single *-epo.fif file (for the montage only — adjacency
depends on channel positions, not on any signal). No friendship join, no trial
alignment, no PLV. Runs in the HyPyP venv (requirements-hypyp.txt).

THE MATH (why we expect an EXACT match). HyPyP connects two inter-brain pairs
e1=(i,j) and e2=(i2,j2) iff:
    (ch_con[i,i2] AND ch_con[j,j2]) OR (ch_con[i,i2] AND j==j2)
      OR (ch_con[j,j2] AND i==i2) OR (i==i2 AND j==j2)
That is precisely the expansion of  (C+I)[i,i2] AND (C+I)[j,j2]  — i.e. our
self-inclusive A⊗A — whether or not ch_con carries self-loops. So a mismatch can
only come from (a) a different single-head adjacency, (b) a pair-ordering
difference, or (c) a montage/channel-order surprise. This script isolates each.

REFERENCES
  Dumas et al. 2010 — a/b/c neighbourhood (p.5).  → ../01-literature-review/lit-dumas.md
  Ayrolles et al. 2020 — HyPyP.                    → ../../raw/literature/HyPyP_paper.pdf
  Our derivation + pipeline: ../plv_matrix_pipeline.py §6 ; design ../plv-proposal-matrix.md
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import numpy as np
from scipy import sparse
import mne

from hypyp.stats import con_matrix, metaconn_matrix_2brains

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — EDIT FOR YOUR MACHINE
# ═════════════════════════════════════════════════════════════════════════════
# Same DATA_DIR as plv_matrix_pipeline.py (Clara's machine paths). Only ONE epoch
# file is read, and only for its montage — point this at your copy.
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\figures\plv_matrix"
    r"\hypyp_crossval"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))
# freqs_mean only feeds HyPyP's freq-extended (chained) matrices, which we ignore;
# the BASE adjacency is frequency-independent. Two values just to exercise the API.
FREQS_MEAN = [10.0, 21.5]   # alpha-mu (8–12) & beta (13–30) midpoints


# ═════════════════════════════════════════════════════════════════════════════
# OUR side — copied VERBATIM from plv_matrix_pipeline.py §6 (kept identical on
# purpose: this script must compare against exactly what the pipeline builds).
# ═════════════════════════════════════════════════════════════════════════════
def build_single_head_adjacency(info) -> sparse.csr_matrix:
    A, names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    assert names == info["ch_names"], "adjacency channel order mismatch"
    A = (A + sparse.eye(A.shape[0])) > 0          # force self-inclusion
    return A.tocsr()


def build_pair_adjacency(A_chan: sparse.csr_matrix) -> sparse.csr_matrix:
    """A ⊗ A over flattened pairs p = i*n_ch + k (Dumas a/b/c)."""
    return (sparse.kron(A_chan, A_chan, format="csr") > 0).tocsr()


def _get_field(result, name, pos=0):
    """HyPyP returns namedtuples; fall back to positional if field name differs
    across 0.6.x point releases."""
    val = getattr(result, name, None)
    return result[pos] if val is None else val


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — montage from one epoch file
# ═════════════════════════════════════════════════════════════════════════════
if not EPOCH_FILES:
    raise SystemExit(f"No *_FG_preprocessed-epo.fif under {DATA_DIR}")

print(f"Reading montage from {os.path.basename(EPOCH_FILES[0])} …")
epochs = mne.read_epochs(EPOCH_FILES[0], preload=False, verbose=False)
info = epochs.info
n = len(info["ch_names"])
print(f"  {n} EEG channels\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — OUR adjacency (A_chan, then A⊗A)
# ═════════════════════════════════════════════════════════════════════════════
A_chan   = build_single_head_adjacency(info)                  # (n, n) self-incl.
pair_adj = build_pair_adjacency(A_chan)                        # (n², n²)
our_pair = (pair_adj.toarray() > 0)
print(f"Our A⊗A built: {n} ch → {pair_adj.shape[0]} pairs.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — HyPyP single-head adjacency (con_matrix) + single-head comparison
# ═════════════════════════════════════════════════════════════════════════════
con = con_matrix(epochs, FREQS_MEAN)
hp_ch_con = sparse.csr_matrix(_get_field(con, "ch_con"))       # (n, n)

our_raw, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
our_raw = sparse.csr_matrix(our_raw)

# Compare off-diagonal structure (self-loop convention may differ — irrelevant to
# the pair-level result, but we report it for transparency).
def _offdiag_bool(M, size):
    B = (np.asarray(M.todense()) > 0)
    np.fill_diagonal(B, False)
    return B

hp_self  = bool((hp_ch_con.diagonal() > 0).all())
our_self = bool((our_raw.diagonal() > 0).all())
offdiag_same = np.array_equal(_offdiag_bool(hp_ch_con, n), _offdiag_bool(our_raw, n))
print("\nSingle-head adjacency:")
print(f"  off-diagonal identical (HyPyP con_matrix vs our find_ch_adjacency): "
      f"{offdiag_same}")
print(f"  self-loops — HyPyP: {hp_self}, ours(raw): {our_self}  "
      f"(we force self-inclusion in build_single_head_adjacency either way)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — HyPyP inter-brain neighbourhood (base metaconn) in OUR pair order
#
# We construct `electrodes` ourselves in canonical i-major order p = i*n + k so
# metaconn's row/col order matches our A⊗A flattening exactly (rather than relying
# on indices_connectivity_interbrain's internal ordering, which would also need a
# 2-brain "hyper" epochs object). Brain-B indices are offset by n, as HyPyP
# expects (metaconn does ch_con[e12-n, e22-n] internally).
#
# NOTE: metaconn_matrix_2brains uses Python double loops over len(electrodes)² =
# n⁴ entries; for n=64 that's ~16.7M iterations — expect a few minutes. We pass a
# DENSE ch_con so the per-entry lookups are fast.
# ═════════════════════════════════════════════════════════════════════════════
ch_con_dense = (np.asarray(hp_ch_con.todense()) > 0)
electrodes = [(i, j + n) for i in range(n) for j in range(n)]   # i-major: p=i*n+j

print(f"\nBuilding HyPyP metaconn over {len(electrodes)} pairs "
      f"(~{len(electrodes)**2/1e6:.1f}M iterations; this can take a few minutes) …")
meta = metaconn_matrix_2brains(electrodes, ch_con_dense, FREQS_MEAN)
hp_pair = (np.asarray(_get_field(meta, "metaconn")) > 0)        # base, NOT _freq

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — the headline comparison: HyPyP metaconn  ==  our A⊗A ?
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PAIR-LEVEL COMPARISON — HyPyP metaconn (base)  vs  our A⊗A")
print("=" * 70)
if hp_pair.shape != our_pair.shape:
    print(f"  SHAPE MISMATCH: HyPyP {hp_pair.shape} vs ours {our_pair.shape}  … FAIL")
    exact = False
else:
    exact = bool(np.array_equal(hp_pair, our_pair))
    diff = hp_pair != our_pair
    n_diff = int(diff.sum())
    print(f"  shape {hp_pair.shape}  |  edges HyPyP={int(hp_pair.sum())}, "
          f"ours={int(our_pair.sum())}  |  differing entries={n_diff}")
    print(f"  EXACT MATCH … {'PASS' if exact else 'FAIL'}")
    if not exact:
        # Localise the first few disagreements as (i,k)~(i2,k2) channel names.
        rr, cc = np.where(diff)
        ch = info["ch_names"]
        print("  first disagreements (pair1 ~ pair2 : HyPyP/ours):")
        for r, c in list(zip(rr, cc))[:10]:
            i, k = r // n, r % n
            i2, k2 = c // n, c % n
            print(f"    ({ch[i]}~{ch[k]}) ~ ({ch[i2]}~{ch[k2]}) : "
                  f"{int(hp_pair[r, c])}/{int(our_pair[r, c])}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — toy 3-electrode hand-check on the REAL HyPyP function
#   self-inclusive line graph 0–1–2 ; neigh(0)={0,1}. Pair (0,0) should connect to
#   {(0,0),(0,1),(1,0),(1,1)} → degree 4 (matches plv_matrix_pipeline.py Step 9).
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nTOY VERIFICATION (3-electrode line)\n" + "=" * 70)
m = 3
toy_ch = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)   # self-incl line
toy_elec = [(i, j + m) for i in range(m) for j in range(m)]
toy_meta = (np.asarray(_get_field(
    metaconn_matrix_2brains(toy_elec, toy_ch, FREQS_MEAN), "metaconn")) > 0)
toy_kron = (sparse.kron(sparse.csr_matrix(toy_ch),
                        sparse.csr_matrix(toy_ch)) > 0).toarray()
toy_match = np.array_equal(toy_meta, toy_kron)
deg00 = int(toy_meta[0].sum())   # pair index 0 == (0,0)
print(f"  HyPyP toy metaconn == our toy kron … {'PASS' if toy_match else 'FAIL'}")
print(f"  pair(0,0) degree = {deg00} (expect 4) … {'PASS' if deg00 == 4 else 'FAIL'}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — save a compact, auditable report
# ═════════════════════════════════════════════════════════════════════════════
summary = os.path.join(OUTPUT_DIR, "adjacency_crossval_summary.txt")
with open(summary, "w", encoding="utf-8") as fh:
    fh.write("HyPyP adjacency cross-validation\n")
    fh.write(f"montage file : {os.path.basename(EPOCH_FILES[0])}\n")
    fh.write(f"n_channels   : {n}\n")
    fh.write(f"single-head off-diagonal identical : {offdiag_same}\n")
    fh.write(f"single-head self-loops  HyPyP/ours-raw : {hp_self}/{our_self}\n")
    fh.write(f"pair-level EXACT MATCH (metaconn == A⊗A) : {exact}\n")
    fh.write(f"  edges HyPyP/ours : {int(hp_pair.sum())}/{int(our_pair.sum())}\n")
    fh.write(f"toy metaconn == toy kron : {toy_match}\n")
    fh.write(f"toy pair(0,0) degree (expect 4) : {deg00}\n")
# Single-head matrices are small (n×n) — keep them for provenance.
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "adjacency_crossval_singlehead.npz"),
    hypyp_ch_con=ch_con_dense, our_A_chan=(A_chan.toarray() > 0),
    ch_names=np.array(info["ch_names"]),
)

print("\n" + "=" * 70)
print("RESULT:", "metaconn ≡ A⊗A — Kronecker derivation independently validated."
      if exact else "MISMATCH — investigate (see disagreements above).")
print("Wrote:", summary)
print("=" * 70)
