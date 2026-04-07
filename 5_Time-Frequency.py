import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"

PARTICIPANT_ID = "301"
PARTICIPANT = 1

EPOCH_FILE = os.path.join(DATA_DIR, f"{PARTICIPANT_ID}_p{PARTICIPANT}_clean-epo.fif")


# Load epochs
epochs = mne.read_epochs(EPOCH_FILE, preload=True)

# sampling frequency
sfreq = epochs.info["sfreq"]

# Define wavelet parameters

# frequencies of interest
foi = np.linspace(1, 30, 30, dtype=int)

# number of cycles per frequency
n_cycles = 3 + 0.5 * foi

# baseline window (seconds)
baseline_window = (-0.25, 0)

# Compute time-frequency representation
tfr = epochs.compute_tfr(
    method="morlet",
    freqs=foi,
    n_cycles=n_cycles,
    return_itc=False,
    average=False,   # keep single epochs first
)

# Average across epochs
tfr_avg = tfr.average()

# Baseline correction
tfr_avg.apply_baseline(baseline_window, mode="percent")

# convert to percent
tfr_avg.data *= 100

# Plot time-frequency power
tfr_avg.plot(
    picks="C3",        # example channel
    title="Time-Frequency Power (C3)",
    baseline=baseline_window
)

plt.show()