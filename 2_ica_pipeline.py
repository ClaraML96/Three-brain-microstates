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
# - Epochs must already be:
#     • Filtered (1–40 Hz)
#     • Bad channels interpolated
#     • Re-referenced to common average        ← done in 01_preprocessing.py
#     • Bad epochs removed (predefined list)
# - Do NOT re-apply average reference here — it was applied in step 1.
# - Do NOT create a separate high-pass filtered copy for ICA fitting.
#   Fit and apply ICA on the same data object so the unmixing matrix
#   matches the data it is applied to.
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

DATA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed"
OUTPUT_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\ica_cleaned"

PARTICIPANT_ID = "302"
PARTICIPANT = 2

EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")

# ICA parameters
N_COMPONENTS = 32       # Number of ICA components
RANDOM_STATE = 97       # For reproducibility
METHOD = "picard"       # ICA algorithm (fast, reliable)

# Output files
ICA_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}-ica.fif")
CLEANED_EPOCHS_FILE = os.path.join(OUTPUT_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_ica_cleaned-epo.fif")

# ============================================================
# STEP 1: Load preprocessed epochs
# ============================================================
# Epochs loaded here already have:
#   - 1–40 Hz bandpass filter
#   - Bad channels interpolated
#   - Common average reference applied
#   - Bad epochs removed
# ============================================================

print("="*70)
print("ICA ARTIFACT REMOVAL PIPELINE")
print("="*70)
print("\nPIPELINE PHASES:")
print("  1) Fit ICA")
print("  2) Automatic artifact detection")
print("  3) Visual inspection and apply ICA")
print(f"\nPHASE 1 — FIT ICA")
print(f"Step 1.1: Loading preprocessed epochs")
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

# Set standard montage for topographic plotting
montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, on_missing="ignore")
print("✓ Standard 10-20 montage applied")

print(f"✓ Loaded: {EPOCH_FILE}")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names)}")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
print(f"  Time window: {epochs.tmin:.2f} to {epochs.tmax:.2f} s")
print(f"  Reference: already set to common average in preprocessing")

# Confirm average reference is already applied — do not re-apply
current_ref = epochs.info.get('custom_ref_applied', False)
if current_ref:
    print("✓ Custom reference confirmed — no action needed")
else:
    print("⚠ WARNING: Average reference not detected in file metadata.")
    print("  Check that 01_preprocessing.py applied it before saving.")
    print("  If you need to re-apply: epochs.set_eeg_reference('average')")

# ============================================================
# STEP 2: Estimate data rank and validate ICA parameters
# ============================================================
# Compute rank to determine maximum number of ICA components.
# Rank is reduced when channels are interpolated, because
# interpolated channels are linear combinations of their
# neighbours — they don't add independent information.
# n_components must not exceed the data rank.
# ============================================================

print(f"\nStep 1.2: Estimating data rank and validating ICA parameters")
print("-"*70)

rank = mne.compute_rank(epochs, tol=1e-6, tol_kind="relative")
rank_eeg = rank['eeg']
print(f"Estimated data rank: {rank_eeg}")

eeg_channels = mne.pick_types(epochs.info, eeg=True, exclude='bads')
n_eeg = len(eeg_channels)

if N_COMPONENTS > rank_eeg:
    print(f"⚠ Warning: n_components ({N_COMPONENTS}) > rank ({rank_eeg})")
    N_COMPONENTS = rank_eeg
    print(f"  Adjusted to: {N_COMPONENTS} components (data rank)")
elif N_COMPONENTS > n_eeg:
    print(f"⚠ Warning: n_components ({N_COMPONENTS}) > EEG channels ({n_eeg})")
    N_COMPONENTS = n_eeg
    print(f"  Adjusted to: {N_COMPONENTS} components")
else:
    print(f"✓ n_components: {N_COMPONENTS}")

print(f"  Method: {METHOD}")
print(f"  Random state: {RANDOM_STATE}")

# ============================================================
# STEP 3: Fit ICA decomposition
# ============================================================
# ICA is fitted directly on the preprocessed epochs — the same
# data object that will be cleaned. This is critical:
#
#   WHY NOT A SEPARATE FILTERED COPY?
#   The unmixing matrix W learned during ICA.fit() is specific
#   to the statistical structure of the data it was trained on.
#   If you fit on a differently-filtered version (e.g. an extra
#   1 Hz high-pass copy) and apply to the original, the matrix
#   is mismatched: W assumes source distributions that don't
#   exist in the target data, causing brain signal to leak into
#   artifact components and vice versa.
#
#   The data is already filtered at 1–40 Hz, which is sufficient
#   for ICA. No additional filtering step is needed.
#
# Rejection threshold: epochs with peak-to-peak amplitude
# > 300 µV are excluded from the ICA fitting (not from the
# final data) to prevent gross artifacts from biasing the
# decomposition. These epochs are still present in the output.
# ============================================================

print(f"\nStep 1.3: Fitting ICA decomposition")
print("-"*70)
print("Fitting ICA on preprocessed epochs (1–40 Hz, average referenced)...")
print("Note: fitting and applying on the same data — no separate filtered copy.")
print("This may take 1-3 minutes depending on data size...")

ica = mne.preprocessing.ICA(
    n_components=N_COMPONENTS,
    method=METHOD,
    random_state=RANDOM_STATE,
    max_iter="auto"
)

ica.fit(
    epochs,
    decim=2,              # Speed up fitting; does not affect output quality
    reject=dict(eeg=300e-6),  # Exclude grossly artefacted epochs from fitting only
    verbose=False
)

print("✓ ICA fitting complete")
print(f"  Components extracted: {ica.n_components_}")
print(f"  Fitted and will be applied to: same preprocessed epochs")

explained_var = ica.get_explained_variance_ratio(epochs, ch_type='eeg')
print(f"  Explained variance: {explained_var['eeg']:.1%}")

# ============================================================
# STEP 4: Automatic artifact detection
# ============================================================
# No dedicated EOG/ECG channels are present in this dataset.
# Automatic detection uses EEG proxies:
#   - EOG: frontal channels Fp1/Fp2 as blink surrogates
#   - ECG: lateral channel T8 as heartbeat surrogate
#   - Muscle: find_bads_muscle on the epoched data
#
# Automatic muscle detection is used to GUIDE
# manual inspection — not as ground truth. Treat all automatic
# suggestions as candidates to verify visually.
# ============================================================

print(f"\nPHASE 2 — AUTOMATIC ARTIFACT DETECTION")
print("Step 2.1: Automatic artifact detection")
print("-"*70)
print("Note: no dedicated EOG/ECG channels — using EEG proxies.")
print("Automatic suggestions guide visual inspection only.")

eog_inds = []
try:
    eog_inds, eog_scores = ica.find_bads_eog(
        epochs,
        ch_name=['Fp1', 'Fp2'],
        verbose=False
    )
    if eog_inds:
        eog_inds = [int(i) for i in eog_inds]
        print(f"✓ EOG components detected: {eog_inds}")
        eog_scores = np.array(eog_scores)
        if eog_scores.ndim == 1:
            scores_str = [f'{float(eog_scores[i]):.2f}' for i in eog_inds]
        else:
            scores_str = [f'{float(np.max(np.abs(eog_scores[:, j]))):.2f}' for j in range(len(eog_inds))]
        print(f"  Correlation scores: {scores_str}")
    else:
        print("  No strong EOG components detected automatically")
except Exception as e:
    print(f"⚠ EOG detection failed: {str(e)}")

ecg_inds = []
try:
    ecg_inds, ecg_scores = ica.find_bads_ecg(
        epochs,
        ch_name='T8',
        method='correlation',
        verbose=False
    )
    # Only keep components with meaningful correlation (>0.3)
    ecg_inds = [i for i in ecg_inds if abs(float(ecg_scores[i])) > 0.3]
    if ecg_inds:
        print(f"✓ ECG components detected: {ecg_inds}")
        print(f"  Correlation scores: {[f'{float(ecg_scores[i]):.2f}' for i in ecg_inds]}")
    else:
        print("  No strong ECG components detected (all below 0.3 threshold)")
except Exception as e:
    print(f"⚠ ECG detection failed: {str(e)}")

muscle_inds = []
try:
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs, verbose=False)
    if muscle_inds:
        print(f"✓ Muscle components detected: {muscle_inds}")
        print(f"  Scores: {[f'{muscle_scores[i]:.2f}' for i in muscle_inds]}")
    else:
        print("  No strong muscle components detected automatically")
except Exception as e:
    print(f"⚠ Muscle detection failed: {str(e)}")

suggested = sorted(int(i) for i in set(eog_inds + ecg_inds + muscle_inds))
if suggested:
    print(f"\n→ Suggested components to inspect: {suggested}")
    print("  Verify all of these visually before excluding anything!")
else:
    print("\n  No components automatically suggested — inspect all manually")

# ============================================================
# STEP 5: Visual inspection of ICA components
# ============================================================
# Open interactive plots for manual inspection.
# Use the guidelines at the top of this script.
# ============================================================

print(f"\nPHASE 3 — VISUAL INSPECTION AND APPLY ICA")
print("Step 3.1: Visual component inspection")
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
print("\n4. BRAIN LEAKAGE WARNING:")
print("   - If a putative artifact component has a smooth,")
print("     biologically plausible topography AND clear alpha/")
print("     theta peaks in its spectrum → do NOT remove it.")
print("   - Removing brain components is worse than keeping")
print("     a small artifact.")
print("="*70)

# Plot all components as topographic maps
ica.plot_components(inst=epochs, picks=range(ica.n_components_))

# Plot time courses — use block=True to pause execution until closed
ica.plot_sources(epochs, show_scrollbars=False, block=True)

print("\nPlotting detailed properties of candidate components...")
print("Each plot shows: topomap, time course, power spectrum, and epoch image")
print("Close each window to proceed to the next component\n")

candidate_components = suggested if suggested else []

# ============================================================
# STEP 6: Manual component selection
# ============================================================

print(f"\nStep 3.2: Manual component selection")
print("-"*70)
print("Based on your visual inspection, enter component numbers to exclude.")
print(f"Automatic suggestions: {suggested if suggested else 'None'}")
print("\nFormat: comma-separated numbers (e.g., 0,3,7,12)")
print("Press Enter without typing to skip removal\n")

user_input = input("Components to remove: ").strip()

if user_input:
    try:
        exclude_components = sorted([int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()])

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
    print("\n→ No components selected — skipping ICA cleaning")
    ica.exclude = []

# ============================================================
# STEP 7: Apply ICA cleaning
# ============================================================
# ICA is applied to the same epochs object it was fitted on.
# This is consistent and avoids the mismatch problem described
# in Step 3 above.
# ============================================================

print(f"\nStep 3.3: Applying ICA cleaning")
print("-"*70)

if ica.exclude:
    print(f"Removing {len(ica.exclude)} component(s): {ica.exclude}")
    cleaned_epochs = ica.apply(epochs.copy(), verbose=False)
    print(f"✓ ICA applied — artifacts removed")
else:
    print("No components excluded — returning original epochs unchanged")
    cleaned_epochs = epochs.copy()

# ============================================================
# STEP 8: Save outputs
# ============================================================

print("\nSTEP 8: Saving outputs")
print("-"*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

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

try:
    cleaned_epochs.save(CLEANED_EPOCHS_FILE, overwrite=True)
    if os.path.exists(CLEANED_EPOCHS_FILE):
        file_size_mb = os.path.getsize(CLEANED_EPOCHS_FILE) / (1024 * 1024)
        print(f"✓ Cleaned epochs saved: {CLEANED_EPOCHS_FILE}")
        print(f"  Size: {file_size_mb:.2f} MB")
    else:
        raise FileNotFoundError(f"Cleaned epochs file was not created: {CLEANED_EPOCHS_FILE}")
except Exception as e:
    print(f"✗ Failed to save cleaned epochs: {e}")
    raise

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("ICA PIPELINE COMPLETE")
print("="*70)
print(f"Input epochs:        {EPOCH_FILE}")
print(f"ICA decomposition:   {ICA_FILE}")
print(f"Cleaned epochs:      {CLEANED_EPOCHS_FILE}")
print(f"Components excluded: {ica.exclude if ica.exclude else 'None'}")
print(f"Final epochs:        {len(cleaned_epochs)}")
print(f"Final channels:      {len(cleaned_epochs.ch_names)}")
print("="*70 + "\n")