import os
import numpy as np
import mne
from autoreject import AutoReject

# ============================================================================
# AUTOREJECT AND FINAL EPOCH CLEANING
# ============================================================================
# This script applies AutoReject to ICA-cleaned epochs to catch any residual
# artefacts that ICA did not remove — e.g. movement spikes, isolated channel
# glitches within single epochs.
#
# AutoReject is employed to catch any remaining
# artefacts by interpolating specific bad channels within each epoch,
# followed by a guided final manual visual inspection with highlighted
# bad epochs determined by the algorithm.
#
# Pipeline position:
#   01_preprocessing.py  → filter, resample, interpolate, avg ref, epoch
#   02_ica.py            → ICA artefact removal
#   03_autoreject.py     → this script (final residual cleaning)
#
# Prerequisites:
# - ICA-cleaned epochs from 02_ica.py
# - Data is already filtered, referenced, and ICA-cleaned
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"
OUTPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed_final"

PARTICIPANT_ID = 304
PARTICIPANT = 3
RANDOM_STATE = 97

INPUT_FILE = os.path.join(INPUT_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_ica_cleaned-epo.fif")
FINAL_EPOCHS_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_final_clean-epo.fif")
REJECT_LOG_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_reject_log.npz")

# ============================================================================
# STEP 1/6: LOAD EPOCHS
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1/6: LOAD EPOCHS")
print(f"{'='*70}")
print(f"Looking for input file: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"ICA-cleaned epochs file not found: {INPUT_FILE}\n"
        f"Expected file name format: {{PARTICIPANT_ID}}_p{{PARTICIPANT}}_ica_cleaned-epo.fif\n"
        f"Run 02_ica.py first."
    )

epochs = mne.read_epochs(INPUT_FILE, preload=True, verbose=False)

epochs_before_ar = len(epochs)
print(f"✓ Loaded ICA-cleaned epochs")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names)}")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
print(f"  Reference: common average (carried through from preprocessing)")

# ============================================================================
# STEP 2/6: FIT AND APPLY AUTOREJECT
# ============================================================================
# AutoReject interpolates bad channels within individual epochs rather than
# dropping whole epochs. This preserves trial count while correcting
# localised artefacts that survive ICA (e.g. movement, transient channel
# dropouts).
#
# The fit-then-transform approach is used so we can inspect the reject_log
# before deciding whether to drop any epochs manually.
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2/6: FIT AND APPLY AUTOREJECT")
print(f"{'='*70}")
print("Fitting AutoReject...")
print("This may take several minutes for large epoch sets.")

ar = AutoReject(random_state=RANDOM_STATE)
ar.fit(epochs)

epochs_ar, reject_log = ar.transform(epochs, return_log=True)

flagged_by_ar = int(np.sum(reject_log.bad_epochs))
print("✓ AutoReject fitting and transform complete")
print(f"  Epochs flagged as bad by AutoReject: {flagged_by_ar} / {epochs_before_ar}")
print("  Flagged epochs are highlighted for visual review in Step 3.")
print("  No epochs are dropped automatically at this stage.")

# ============================================================================
# STEP 3/6: MANUAL VISUAL INSPECTION
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3/6: MANUAL VISUAL INSPECTION")
print(f"{'='*70}")
print("Methodology note:")
print(
    "  Using AutoReject this late in the pipeline (after ICA) can produce "
    "false positives if the signal is already clean. The reject_log is used "
    "as a visual guide only — no automatic dropping is performed."
)
print("\nOpening reject_log visualization (bad epochs highlighted in red)...")
reject_log.plot_epochs(epochs)
print("Opening interactive epoch browser...")
epochs_ar.plot(block=True)

# ============================================================================
# STEP 4/6: MANUAL EPOCH REJECTION
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4/6: MANUAL EPOCH REJECTION")
print(f"{'='*70}")

manual_dropped = 0
user_input = input(
    "Enter epoch indices to drop (1-based, comma-separated), or press Enter to skip: "
).strip()

if user_input:
    tokens = [token.strip() for token in user_input.split(',') if token.strip()]
    try:
        indices_1based = sorted(set(int(token) for token in tokens))
    except ValueError as exc:
        raise ValueError(
            "Invalid input format. Please enter integers only, e.g., 1,5,12"
        ) from exc

    invalid = [idx for idx in indices_1based if idx < 1 or idx > len(epochs_ar)]
    if invalid:
        raise ValueError(
            f"Invalid epoch indices (1-based): {invalid}. "
            f"Valid range is 1 to {len(epochs_ar)}."
        )

    indices_0based = [idx - 1 for idx in indices_1based]
    epochs_ar.drop(indices_0based, reason='AUTOREJECT_MANUAL', verbose=False)
    manual_dropped = len(indices_0based)

    print(f"✓ Manually dropped epochs (1-based): {indices_1based}")
    print(f"  Total manually dropped: {manual_dropped}")
else:
    print("✓ No manual drops requested")

# ============================================================================
# STEP 5/6: SAVE OUTPUTS
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5/6: SAVE OUTPUTS")
print(f"{'='*70}")
print(f"Target output directory: {OUTPUT_DIR}")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("✓ Output directory created/verified")
except Exception as exc:
    print(f"⚠ Failed to create output directory: {exc}")
    raise

print("Saving final cleaned epochs...")
epochs_ar.save(FINAL_EPOCHS_FILE, overwrite=True)

print("Saving reject log...")
reject_log.save(REJECT_LOG_FILE, overwrite=True)

if not os.path.exists(FINAL_EPOCHS_FILE):
    raise FileNotFoundError(f"Final epochs file was not created: {FINAL_EPOCHS_FILE}")
if not os.path.exists(REJECT_LOG_FILE):
    raise FileNotFoundError(f"Reject log file was not created: {REJECT_LOG_FILE}")

final_epochs_size_mb = os.path.getsize(FINAL_EPOCHS_FILE) / (1024 * 1024)
reject_log_size_mb = os.path.getsize(REJECT_LOG_FILE) / (1024 * 1024)

print(f"✓ Final epochs saved: {FINAL_EPOCHS_FILE}")
print(f"  Size: {final_epochs_size_mb:.2f} MB")
print(f"✓ Reject log saved:   {REJECT_LOG_FILE}")
print(f"  Size: {reject_log_size_mb:.2f} MB")

# ============================================================================
# STEP 6/6: SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("STEP 6/6: SUMMARY")
print(f"{'='*70}")
print(f"Epochs before AutoReject:     {epochs_before_ar}")
print(f"Epochs flagged by AutoReject: {flagged_by_ar}")
print(f"Epochs manually dropped:      {manual_dropped}")
print(f"Final epochs remaining:       {len(epochs_ar)}")
print(f"\nPipeline complete.")
print(f"Final file: {FINAL_EPOCHS_FILE}")
print(f"{'='*70}\n")