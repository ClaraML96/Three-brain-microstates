import os
import mne
import matplotlib
matplotlib.use("TkAgg")

# ============================================================
# ICA COMPONENT VISUALIZATION ONLY
# ============================================================
# This script only visualizes already-fitted ICA components.
# It does NOT fit ICA, apply cleaning, or save cleaned epochs.
#
# Prerequisite:
# - ICA file must already exist (created by ica_pipeline.py)
# ============================================================

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed"
ICA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\ica_cleaned"

PARTICIPANT_ID = "302"
PARTICIPANT = 2

EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")
ICA_FILE = os.path.join(ICA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}-ica.fif")

print("=" * 70)
print("ICA COMPONENT TOPOGRAPHY VISUALIZATION")
print("=" * 70)

# ------------------------------------------------------------
# STEP 1: Load epochs (for click-through detail view)
# ------------------------------------------------------------
print("\nSTEP 1: Loading epochs")
print("-" * 70)
print(f"Epoch file: {EPOCH_FILE}")

if not os.path.exists(EPOCH_FILE):
    raise FileNotFoundError(f"Epoch file not found: {EPOCH_FILE}")

epochs = mne.read_epochs(EPOCH_FILE, preload=True, verbose=False)
montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, on_missing="ignore")
epochs.set_eeg_reference("average", projection=False, verbose=False)

print(f"✓ Loaded epochs: {len(epochs)}")

# ------------------------------------------------------------
# STEP 2: Load ICA decomposition
# ------------------------------------------------------------
print("\nSTEP 2: Loading ICA decomposition")
print("-" * 70)
print(f"ICA file: {ICA_FILE}")

if not os.path.exists(ICA_FILE):
    raise FileNotFoundError(
        f"ICA file not found: {ICA_FILE}\n"
        f"Run ica_pipeline.py first to create it."
    )

ica = mne.preprocessing.read_ica(ICA_FILE, verbose=False)
print(f"✓ Loaded ICA with {ica.n_components_} components")

# ------------------------------------------------------------
# STEP 3: Plot ICA components only
# ------------------------------------------------------------
print("\nSTEP 3: Plotting ICA component topomaps")
print("-" * 70)
print("Click a component map to open detailed properties for that component.")
ica.plot_components(inst=epochs, picks=range(ica.n_components_))

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print("Topographies were plotted with click-through detailed component view.")
