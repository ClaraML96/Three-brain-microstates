import mne
import matplotlib
import os
import numpy as np

# Ensure the interactive window pops up
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed\\"
PARTICIPANT_ID = 303
PARTICIPANT = 2

# Path to your already processed data
EPOCHS_FILE = os.path.join(DATA_PATH, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")

# ============================================================================
# LOAD AND VISUALIZE
# ============================================================================

if not os.path.exists(EPOCHS_FILE):
    print(f"Error: File not found at {EPOCHS_FILE}")
else:
    print(f"Loading processed epochs: {EPOCHS_FILE}")
    
    # Load the .fif file
    # preload=True is necessary for interactive plotting/dropping
    epochs = mne.read_epochs(EPOCHS_FILE, preload=True, verbose=True)

    print(epochs.get_data().shape)
    print(np.min(epochs.get_data()), np.max(epochs.get_data())) 

    print(f"Number of epochs: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print("="*70)

    epochs.plot(
        picks="eeg",
        n_channels=32,
        n_epochs=5,
        scalings=dict(eeg=50e-6),
        title="Manual Review of Preprocessed Data",
        block=True
    )

    # Optional: Print summary after closing the window
    final_count = len(epochs)
    print(f"\nVisualization closed.")
    print(f"Epochs remaining: {final_count}")