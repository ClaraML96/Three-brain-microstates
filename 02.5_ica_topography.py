import os
import mne
import matplotlib
matplotlib.use("TkAgg")

# ============================================================
# ICA COMPONENT VISUALIZATION ONLY
# ============================================================

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed"
ICA_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\ica_cleaned"

PARTICIPANT_ID = "303"
PARTICIPANT = 2

EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")
ICA_FILE = os.path.join(ICA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}-ica.fif")

# ------------------------------------------------------------
# STEP 1: Load epochs (for click-through detail view)
# ------------------------------------------------------------
print(f"Epoch file: {EPOCH_FILE}")

if not os.path.exists(EPOCH_FILE):
    raise FileNotFoundError(f"Epoch file not found: {EPOCH_FILE}")

epochs = mne.read_epochs(EPOCH_FILE, preload=True, verbose=False)
montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, on_missing="ignore")
epochs.set_eeg_reference("average", projection=False, verbose=False)

print(f"Loaded epochs: {len(epochs)}")

# ------------------------------------------------------------
# STEP 2: Load ICA decomposition
# ------------------------------------------------------------
print(f"ICA file: {ICA_FILE}")

if not os.path.exists(ICA_FILE):
    raise FileNotFoundError(
        f"ICA file not found: {ICA_FILE}\n"
        f"Run ica_pipeline.py first to create it."
    )

ica = mne.preprocessing.read_ica(ICA_FILE, verbose=False)
print(f"Loaded ICA with {ica.n_components_} components")

# ------------------------------------------------------------
# STEP 3: Plot ICA components only
# ------------------------------------------------------------
ica.plot_components(inst=epochs, picks=range(ica.n_components_))