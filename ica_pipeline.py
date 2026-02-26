import mne
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ============================================================================
# ICA ARTIFACT REMOVAL PIPELINE
# ============================================================================
# This script performs Independent Component Analysis (ICA) on preprocessed
# EEG epochs to identify and remove ocular, cardiac, and muscle artifacts.
# 
# Prerequisites:
# - Epochs must already be filtered, cleaned of bad channels, and have bad
#   epochs removed
# - This script does NOT redo basic preprocessing
# 
# ICA Component Rejection Guidelines:
# ------------------------------------
# OCULAR ARTIFACTS (eye blinks, saccades):
#   - Topography: Strong frontal focus (Fp1, Fp2, AF3, AF4)
#   - Time course: Sharp, transient peaks synchronized with blinks
#   - Frequency: Broadband but prominent in low frequencies (<4 Hz)
# 
# CARDIAC ARTIFACTS (heartbeat, pulse):
#   - Topography: Lateral or posterior focus, sometimes asymmetric
#   - Time course: Regular, rhythmic oscillations (~1 Hz, 60 bpm)
#   - Frequency: Sharp peak at heartbeat frequency and harmonics
# 
# MUSCLE ARTIFACTS (jaw clenching, neck tension):
#   - Topography: Temporal, frontal, or diffuse scalp distribution
#   - Time course: High-frequency bursts, irregular timing
#   - Frequency: Dominant power >20 Hz, often >30 Hz
# 
# BRAIN ACTIVITY (keep these components):
#   - Topography: Smooth, biologically plausible scalp distribution
#   - Time course: Oscillatory patterns in EEG frequency bands
#   - Frequency: Clear peaks in alpha (8-12 Hz), theta (4-8 Hz), etc.
# ============================================================================

# ============================================================
# CONFIGURATION
# ============================================================

# Input/output paths (absolute paths for Windows reliability)
DATA_DIR = r"C:\\Users\\clara\\Three-brain-microstates\\data\\preprocessed"
OUTPUT_DIR = r"C:\\Users\\clara\\Three-brain-microstates\\data\\ica_cleaned"

PARTICIPANT_ID = "301"                # Used for output filenames
EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p1_clean-epo.fif")

# ICA parameters
N_COMPONENTS = 32                     # Number of ICA components
RANDOM_STATE = 97                     # For reproducibility
METHOD = "picard"                     # ICA algorithm (fast, reliable)

# Output files
ICA_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}-ica.fif")
CLEANED_EPOCHS_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_ica_cleaned-epo.fif")

# ============================================================
# STEP 1: Load preprocessed epochs
# ============================================================
# Load epochs that have already been:
# - Filtered (e.g., 1-40 Hz)
# - Bad channels interpolated
# - Bad epochs rejected
# ============================================================

print("="*70)
print("ICA ARTIFACT REMOVAL PIPELINE")
print("="*70)
print(f"\nSTEP 1: Loading preprocessed epochs")
print("-"*70)

print(f"Looking for: {EPOCH_FILE}")

if not os.path.exists(EPOCH_FILE):
    print(f"⚠ File not found: {EPOCH_FILE}")
    print(f"\nSearching in data directory...")
    if os.path.exists(DATA_DIR):
        print(f"Files in {DATA_DIR}:")
        for f in os.listdir(DATA_DIR):
            print(f"  - {f}")
    raise FileNotFoundError(f"Epoch file not found: {EPOCH_FILE}")

epochs = mne.read_epochs(EPOCH_FILE, preload=True, verbose=False)

print(f"✓ Loaded: {EPOCH_FILE}")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names)} (excluding stim)")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
print(f"  Time window: {epochs.tmin:.2f} to {epochs.tmax:.2f} s")

# ============================================================
# STEP 2: Validate and apply common average reference
# ============================================================
# Average reference is critical for ICA:
# - Removes reference bias
# - Ensures all channels contribute equally
# - Required before ICA decomposition
# ============================================================

print(f"\nSTEP 2: Applying common average reference")
print("-"*70)

# Check current reference
current_ref = epochs.info['custom_ref_applied']
if current_ref == mne.io.constants.FIFF.FIFFV_MNE_CUSTOM_REF_ON:
    print("⚠ Warning: Data already has a custom reference applied")
    print("  Re-applying average reference...")

epochs.set_eeg_reference("average", projection=False, verbose=False)
print("✓ Common average reference applied")

# ============================================================
# STEP 3: Validate ICA parameters
# ============================================================
# Ensure n_components does not exceed available EEG channels
# ============================================================

print(f"\nSTEP 3: Validating ICA parameters")
print("-"*70)

eeg_channels = mne.pick_types(epochs.info, eeg=True, exclude='bads')
n_eeg = len(eeg_channels)

if N_COMPONENTS > n_eeg:
    print(f"⚠ Warning: n_components ({N_COMPONENTS}) > EEG channels ({n_eeg})")
    N_COMPONENTS = n_eeg
    print(f"  Adjusted to: {N_COMPONENTS} components")
else:
    print(f"✓ n_components: {N_COMPONENTS}")

print(f"  Method: {METHOD}")
print(f"  Random state: {RANDOM_STATE}")
print(f"  Max iterations: auto")

# ============================================================
# STEP 4: Fit ICA decomposition
# ============================================================
# Picard algorithm decomposes EEG into independent components:
# - Each component = spatially fixed source
# - Components are statistically independent
# - Artifacts typically isolated to single components
# ============================================================

print(f"\nSTEP 4: Fitting ICA decomposition")
print("-"*70)
print("This may take 1-3 minutes depending on data size...")

ica = mne.preprocessing.ICA(
    n_components=N_COMPONENTS,
    method=METHOD,
    random_state=RANDOM_STATE,
    max_iter="auto",
    fit_params=dict(ortho=False, extended=True)
)

ica.fit(epochs, verbose=False)

print("✓ ICA fitting complete")
print(f"  Components extracted: {ica.n_components_}")

# Calculate explained variance
explained_var = ica.get_explained_variance_ratio(epochs, ch_type='eeg')
print(f"  Explained variance: {explained_var['eeg']:.1%}")

# ============================================================
# STEP 5: Automatic artifact detection
# ============================================================
# Attempt automatic detection of EOG and ECG components:
# - find_bads_eog: correlates components with eye blinks
# - find_bads_ecg: correlates components with heartbeat
# These are suggestions only - always verify visually!
# ============================================================

print(f"\nSTEP 5: Automatic artifact detection")
print("-"*70)

# EOG detection
eog_inds = []
try:
    eog_inds, eog_scores = ica.find_bads_eog(epochs, verbose=False)
    if eog_inds:
        print(f"✓ EOG components detected: {eog_inds}")
        print(f"  Correlation scores: {[f'{eog_scores[i]:.2f}' for i in eog_inds]}")
    else:
        print("  No strong EOG components detected")
except Exception as e:
    print(f"⚠ EOG detection failed: {str(e)}")
    print("  (No EOG channels or insufficient data)")

# ECG detection
ecg_inds = []
try:
    ecg_inds, ecg_scores = ica.find_bads_ecg(epochs, verbose=False)
    if ecg_inds:
        print(f"✓ ECG components detected: {ecg_inds}")
        print(f"  Correlation scores: {[f'{ecg_scores[i]:.2f}' for i in ecg_inds]}")
    else:
        print("  No strong ECG components detected")
except Exception as e:
    print(f"⚠ ECG detection failed: {str(e)}")
    print("  (No ECG channels or insufficient data)")

# Combine suggestions
suggested = sorted(list(set(eog_inds + ecg_inds)))
if suggested:
    print(f"\n→ Suggested components to remove: {suggested}")
    print("  (Verify these visually before accepting!)")
else:
    print("\n  No components automatically detected")

# ============================================================
# STEP 6: Visual inspection of ICA components
# ============================================================
# Open interactive plots for manual inspection:
# 1. Topographic maps (spatial distribution)
# 2. Time courses and frequency spectra
# 
# Use the guidelines at the top of this script to identify:
# - Ocular artifacts (frontal, low frequency)
# - Cardiac artifacts (rhythmic, ~1 Hz)
# - Muscle artifacts (high frequency, >20 Hz)
# ============================================================

print(f"\nSTEP 6: Visual component inspection")
print("-"*70)
print("Opening interactive plots...")
print("\n" + "="*70)
print("COMPONENT INSPECTION GUIDE")
print("="*70)
print("1. TOPOGRAPHIC MAPS:")
print("   - Check spatial distribution")
print("   - Look for focal frontal (EOG) or lateral (ECG) patterns")
print("\n2. TIME COURSES:")
print("   - Eye blinks: sharp transients")
print("   - Heartbeat: regular ~1 Hz rhythm")
print("   - Muscle: irregular high-frequency bursts")
print("\n3. POWER SPECTRA:")
print("   - EOG: broadband, low frequency dominant")
print("   - ECG: sharp peak at heartbeat frequency")
print("   - Muscle: high power >20 Hz")
print("   - Brain: peaks in alpha/theta bands")
print("="*70)

# Plot all components as topographic maps
ica.plot_components(inst=epochs, picks=range(ica.n_components_))

# Plot time courses and power spectra
ica.plot_sources(epochs, show_scrollbars=False, block=True)

# ============================================================
# STEP 7: Manual component selection
# ============================================================
# Based on visual inspection, manually enter components to exclude
# ============================================================

print(f"\nSTEP 7: Manual component selection")
print("-"*70)
print("Based on your visual inspection, enter component numbers to exclude.")
print(f"Automatic suggestions: {suggested if suggested else 'None'}")
print("\nFormat: comma-separated numbers (e.g., 0,3,7,12)")
print("Press Enter without typing to skip removal\n")

user_input = input("Components to remove: ").strip()

if user_input:
    try:
        exclude_components = sorted([int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()])
        
        # Validate component numbers
        invalid = [c for c in exclude_components if c >= ica.n_components_]
        if invalid:
            raise ValueError(f"Invalid component numbers: {invalid} (max: {ica.n_components_-1})")
        
        ica.exclude = exclude_components
        print(f"\n✓ Components marked for removal: {exclude_components}")
        print(f"  Total: {len(exclude_components)} / {ica.n_components_} components")
        
    except ValueError as e:
        print(f"\n⚠ Error parsing input: {e}")
        print("No components will be removed.")
        ica.exclude = []
else:
    print("\n→ No components selected - skipping ICA cleaning")
    ica.exclude = []

# ============================================================
# STEP 8: Apply ICA cleaning
# ============================================================
# Remove selected components and reconstruct clean epochs
# ============================================================

print(f"\nSTEP 8: Applying ICA cleaning")
print("-"*70)

if ica.exclude:
    print(f"Removing {len(ica.exclude)} component(s)...")
    cleaned_epochs = ica.apply(epochs.copy(), verbose=False)
    print(f"✓ ICA applied - artifacts removed")
else:
    print("No components excluded - returning original epochs")
    cleaned_epochs = epochs.copy()

# ============================================================
# STEP 9: Save outputs
# ============================================================

print("\nSTEP 9: Saving outputs")
print("-"*70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# --------------------
# Save ICA object
# --------------------
try:
    ica.save(ICA_FILE, overwrite=True)
    if os.path.exists(ICA_FILE):
        file_size_mb = os.path.getsize(ICA_FILE) / (1024 * 1024)
        print(f"✓ ICA object saved: {ICA_FILE}")
        print(f"  Size: {file_size_mb:.2f} MB")
    else:
        raise FileNotFoundError(f"ICA file was not created: {ICA_FILE}")
except Exception as e:
    print(f"Failed to save ICA object: {e}")
    raise

# --------------------
# Save cleaned epochs
# --------------------
try:
    cleaned_epochs.save(CLEANED_EPOCHS_FILE, overwrite=True)
    if os.path.exists(CLEANED_EPOCHS_FILE):
        file_size_mb = os.path.getsize(CLEANED_EPOCHS_FILE) / (1024 * 1024)
        print(f"✓ Cleaned epochs saved: {CLEANED_EPOCHS_FILE}")
        print(f"  Size: {file_size_mb:.2f} MB")
    else:
        raise FileNotFoundError(f"Cleaned epochs file was not created: {CLEANED_EPOCHS_FILE}")
except Exception as e:
    print(f"❌ Failed to save cleaned epochs: {e}")
    raise


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("ICA PIPELINE COMPLETE")
print("="*70)
print(f"Input epochs:      {EPOCH_FILE}")
print(f"ICA decomposition: {ICA_FILE}")
print(f"Cleaned epochs:    {CLEANED_EPOCHS_FILE}")
print(f"Components excluded: {ica.exclude if ica.exclude else 'None'}")
print(f"Final epochs: {len(cleaned_epochs)}")
print(f"Final channels: {len(cleaned_epochs.ch_names)}")
print("="*70 + "\n")