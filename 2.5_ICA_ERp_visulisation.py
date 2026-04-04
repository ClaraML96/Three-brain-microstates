import mne
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"
ICA_DIR  = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"

PARTICIPANT_ID = "304"
PARTICIPANT = 1

EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")
ICA_FILE   = os.path.join(ICA_DIR,  f"{PARTICIPANT_ID}_p{PARTICIPANT}-ica.fif")

# ============================================================
# STEP 1: Load epochs
# ============================================================

print("Loading epochs...")
epochs = mne.read_epochs(EPOCH_FILE, preload=True)

montage = mne.channels.make_standard_montage("standard_1020")
epochs.set_montage(montage, on_missing="ignore")

print(f"Epochs loaded: {len(epochs)}")

# ============================================================
# STEP 2: Load ICA solution
# ============================================================

print("Loading ICA...")
ica = mne.preprocessing.read_ica(ICA_FILE)

print(f"ICA components: {ica.n_components_}")

# ============================================================
# STEP 3: Compute ERP (Evoked)
# ============================================================

print("Computing ERP...")
evoked = epochs.average()

# ============================================================
# STEP 4: Visualize ICA + ERP simultaneously
# ============================================================

print("Opening visualization windows...")

# ICA sources (scrollable)
ica.plot_sources(epochs, show_scrollbars=True)

# ICA component maps
ica.plot_components(inst=epochs)

# ERP plot
# evoked.plot(spatial_colors=True)

# ERP topomap over time
# evoked.plot_topomap(times="peaks")

plt.show()