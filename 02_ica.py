import mne
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed"
OUTPUT_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\ica_cleaned"

PARTICIPANT_ID = "303"
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
print(f"Looking for: {EPOCH_FILE}")

if not os.path.exists(EPOCH_FILE):
    print(f"File not found: {EPOCH_FILE}")
    if os.path.exists(DATA_DIR):
        print(f"Files in {DATA_DIR}:")
        for f in os.listdir(DATA_DIR):
            print(f"  - {f}")
    raise FileNotFoundError(f"Epoch file not found: {EPOCH_FILE}")

epochs = mne.read_epochs(EPOCH_FILE, preload=True, verbose=False)

# Set standard montage for topographic plotting
montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, on_missing="ignore")

# Confirm average reference is already applied — do not re-apply
current_ref = epochs.info.get('custom_ref_applied', False)
if current_ref:
    print("Custom reference confirmed — no action needed")
else:
    print("WARNING: Average reference not detected in file metadata.")
    
# ============================================================
# STEP 2: Estimate data rank and validate ICA parameters
# ============================================================

rank = mne.compute_rank(epochs, tol=1e-6, tol_kind="relative")
rank_eeg = rank['eeg']

eeg_channels = mne.pick_types(epochs.info, eeg=True, exclude='bads')
n_eeg = len(eeg_channels)

if N_COMPONENTS > rank_eeg:
    print(f"Warning: n_components ({N_COMPONENTS}) > rank ({rank_eeg})")
    N_COMPONENTS = rank_eeg
    print(f" Adjusted to: {N_COMPONENTS} components (data rank)")
elif N_COMPONENTS > n_eeg:
    print(f"Warning: n_components ({N_COMPONENTS}) > EEG channels ({n_eeg})")
    N_COMPONENTS = n_eeg
    print(f" Adjusted to: {N_COMPONENTS} components")
else:
    print(f"n_components: {N_COMPONENTS}")

# ============================================================
# STEP 3: Fit ICA decomposition
# ============================================================
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
explained_var = ica.get_explained_variance_ratio(epochs, ch_type='eeg')

# ============================================================
# STEP 4: Automatic artifact detection
# ============================================================
eog_inds = []
try:
    eog_inds, eog_scores = ica.find_bads_eog(
        epochs,
        ch_name=['Fp1', 'Fp2'],
        verbose=False
    )
    if eog_inds:
        eog_inds = [int(i) for i in eog_inds]
        print(f"EOG components detected: {eog_inds}")
        eog_scores = np.array(eog_scores)
        if eog_scores.ndim == 1:
            scores_str = [f'{float(eog_scores[i]):.2f}' for i in eog_inds]
        else:
            scores_str = [f'{float(np.max(np.abs(eog_scores[:, j]))):.2f}' for j in range(len(eog_inds))]
        print(f" Correlation scores: {scores_str}")
    else:
        print(" No strong EOG components detected automatically")
except Exception as e:
    print(f"EOG detection failed: {str(e)}")

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
        print(f"ECG components detected: {ecg_inds}")
        print(f" Correlation scores: {[f'{float(ecg_scores[i]):.2f}' for i in ecg_inds]}")
    else:
        print(" No strong ECG components detected (all below 0.3 threshold)")
except Exception as e:
    print(f"ECG detection failed: {str(e)}")

muscle_inds = []
try:
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs, verbose=False)
    if muscle_inds:
        print(f"Muscle components detected: {muscle_inds}")
        print(f" Scores: {[f'{muscle_scores[i]:.2f}' for i in muscle_inds]}")
    else:
        print(" No strong muscle components detected automatically")
except Exception as e:
    print(f"Muscle detection failed: {str(e)}")

suggested = sorted(int(i) for i in set(eog_inds + ecg_inds + muscle_inds))
if suggested:
    print(f"\nSuggested components to inspect: {suggested}")
    print("  Verify all of these visually before excluding anything!")
else:
    print("\n  No components automatically suggested — inspect all manually")

# ============================================================
# STEP 5: Visual inspection of ICA components
# ============================================================

# Plot all components as topographic maps
ica.plot_components(inst=epochs, picks=range(ica.n_components_))

# Plot time courses — use block=True to pause execution until closed
ica.plot_sources(epochs, show_scrollbars=False, block=True)

candidate_components = suggested if suggested else []

# ============================================================
# STEP 6: Manual component selection
# ============================================================
user_input = input("Components to remove: ").strip()

if user_input:
    try:
        exclude_components = sorted([int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()])

        invalid = [c for c in exclude_components if c >= ica.n_components_]
        if invalid:
            raise ValueError(f"Invalid component numbers: {invalid} (max: {ica.n_components_-1})")

        ica.exclude = exclude_components
        print(f"\nComponents marked for removal: {exclude_components}")
        print(f" Total: {len(exclude_components)} / {ica.n_components_} components")

    except ValueError as e:
        print(f"\nError parsing input: {e}")
        print("No components will be removed.")
        ica.exclude = []
else:
    print("\nNo components selected — skipping ICA cleaning")
    ica.exclude = []

# ============================================================
# STEP 7: Apply ICA cleaning
# ============================================================
if ica.exclude:
    print(f"Removing {len(ica.exclude)} component(s): {ica.exclude}")
    cleaned_epochs = ica.apply(epochs.copy(), verbose=False)
    print(f"ICA applied — artifacts removed")
else:
    print("No components excluded — returning original epochs unchanged")
    cleaned_epochs = epochs.copy()

# ============================================================
# STEP 8: Save outputs
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    ica.save(ICA_FILE, overwrite=True)
    if os.path.exists(ICA_FILE):
        file_size_mb = os.path.getsize(ICA_FILE) / (1024 * 1024)
        print(f"ICA object saved: {ICA_FILE}")
        print(f" Size: {file_size_mb:.2f} MB")
    else:
        raise FileNotFoundError(f"ICA file was not created: {ICA_FILE}")
except Exception as e:
    print(f"Failed to save ICA object: {e}")
    raise

try:
    cleaned_epochs.save(CLEANED_EPOCHS_FILE, overwrite=True)
    if os.path.exists(CLEANED_EPOCHS_FILE):
        file_size_mb = os.path.getsize(CLEANED_EPOCHS_FILE) / (1024 * 1024)
        print(f"Cleaned epochs saved: {CLEANED_EPOCHS_FILE}")
        print(f" Size: {file_size_mb:.2f} MB")
    else:
        raise FileNotFoundError(f"Cleaned epochs file was not created: {CLEANED_EPOCHS_FILE}")
except Exception as e:
    print(f"Failed to save cleaned epochs: {e}")
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