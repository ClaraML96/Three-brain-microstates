import mne
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import pickle

# ============================================================
# AUTOREJECT - POST-ICA ARTIFACT INTERPOLATION
# ============================================================
# Follows Li et al. (2025) preprint pipeline:
#   - Autoreject 0.4.3 interpolates bad channels within epochs
#   - Does NOT drop epochs automatically
#   - Flagged epochs are highlighted for manual review
# Install: pip install autoreject
# ============================================================

# ============================================================
# CONFIGURATION — update paths/ID to match your ICA script
# ============================================================

ICA_OUTPUT_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\ica_cleaned"
OUTPUT_DIR     = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed_final"

PARTICIPANT_ID = "301"

ICA_CLEANED_FILE = os.path.join(ICA_OUTPUT_DIR, f"{PARTICIPANT_ID}_ica_cleaned-epo.fif")
AR_OUTPUT_FILE   = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_final_clean-epo.fif")
AR_OBJECT_FILE   = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}-autoreject.pkl")

# ============================================================
# STEP 1: Load ICA-cleaned epochs
# ============================================================

print("="*70)
print("AUTOREJECT POST-ICA INTERPOLATION")
print("="*70)
print(f"\nSTEP 1: Loading ICA-cleaned epochs")
print("-"*70)

if not os.path.exists(ICA_CLEANED_FILE):
    raise FileNotFoundError(f"ICA-cleaned epochs not found: {ICA_CLEANED_FILE}")

epochs = mne.read_epochs(ICA_CLEANED_FILE, preload=True, verbose=False)

print(f"✓ Loaded: {ICA_CLEANED_FILE}")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names)}")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")

# ============================================================
# STEP 2: Fit Autoreject
# ============================================================
# Learns per-channel thresholds from the data.
# n_interpolate = candidate numbers of channels to interpolate
# per epoch before considering the epoch fully bad.

print(f"\nSTEP 2: Fitting Autoreject")
print("-"*70)
print("Learning per-channel rejection thresholds...")
print("This may take several minutes...")

try:
    from autoreject import AutoReject
except ImportError:
    raise ImportError("Run: pip install autoreject")

ar = AutoReject(
    n_interpolate=[1, 2, 4],  # Channels to try interpolating per epoch
    random_state=97,           # Matches ICA random state
    n_jobs=-1,                 # Use all CPU cores
    verbose='tqdm'
)

ar.fit(epochs)
print("✓ Autoreject fitting complete")

# ============================================================
# STEP 3: Apply — interpolate bad channels, flag bad epochs
# ============================================================
# Per the preprint: epochs are NOT dropped automatically here.
# The reject_log only guides the manual inspection below.

print(f"\nSTEP 3: Applying interpolation")
print("-"*70)

epochs_ar, reject_log = ar.transform(epochs, return_log=True)

n_flagged = reject_log.bad_epochs.sum()
print(f"✓ Autoreject applied")
print(f"  Epochs flagged as bad: {n_flagged} / {len(epochs)}")
print(f"  No epochs dropped yet — manual inspection follows")

# ============================================================
# STEP 4: Visual inspection with flagged epochs highlighted
# ============================================================
# The preprint notes that Autoreject is prone to false-positives
# at this late stage (signal already clean), so manual
# confirmation is required before dropping anything.

print(f"\nSTEP 4: Manual visual inspection")
print("-"*70)
print("RED epochs = flagged by Autoreject")
print("Mark/unmark epochs manually, then close the window.")
print("Per preprint: be conservative — only drop clear artefacts.\n")

epochs_ar.plot(
    n_epochs=10,
    n_channels=32,
    scalings='auto',
    title=f"Post-Autoreject inspection — {PARTICIPANT_ID}",
    show_scrollbars=True,
    block=True,
    bad_color='red'
)

# ============================================================
# STEP 5: Drop manually selected bad epochs
# ============================================================

print(f"\nSTEP 5: Dropping manually selected epochs")
print("-"*70)

n_before = len(epochs_ar)
epochs_ar.drop_bad()
n_after  = len(epochs_ar)
n_dropped = n_before - n_after

print(f"✓ Epochs before: {n_before}")
print(f"  Epochs dropped: {n_dropped}")
print(f"  Epochs remaining: {n_after}")
print(f"  Preprint average: 3.45 dropped (SD=5.12)")

if n_dropped > 15:
    print(f"  ⚠ Higher than expected — consider reviewing earlier steps")

# ============================================================
# STEP 6: Save outputs
# ============================================================

print(f"\nSTEP 6: Saving outputs")
print("-"*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

epochs_ar.save(AR_OUTPUT_FILE, overwrite=True)
size_mb = os.path.getsize(AR_OUTPUT_FILE) / (1024 * 1024)
print(f"✓ Final clean epochs: {AR_OUTPUT_FILE} ({size_mb:.2f} MB)")

with open(AR_OBJECT_FILE, 'wb') as f:
    pickle.dump(ar, f)
print(f"✓ Autoreject object:  {AR_OBJECT_FILE}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("AUTOREJECT PIPELINE COMPLETE")
print("="*70)
print(f"Input:           {ICA_CLEANED_FILE}")
print(f"Output:          {AR_OUTPUT_FILE}")
print(f"Epochs dropped:  {n_dropped}")
print(f"Final count:     {n_after}")
print(f"Final channels:  {len(epochs_ar.ch_names)}")
print("="*70 + "\n")