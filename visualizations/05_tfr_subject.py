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

# ------------------------------------------------------------
# LOAD EPOCHS
# ------------------------------------------------------------

# Loading epochs
epochs = mne.read_epochs(EPOCH_FILE, preload=True)

# sampling frequency
sfreq = epochs.info["sfreq"]

# Listing conditions
print("Conditions:", list(epochs.event_id.keys()))

# ------------------------------------------------------------
# MORLET WAVELET PARAMETERS
# ------------------------------------------------------------

# Li's mail snippet

# frequencies of interest
foi = np.linspace(1, 30, 30, dtype=int)

# number of cycles per frequency
n_cycles = 3 + 0.5 * foi

# baseline window (seconds)
baseline_window = (-0.25, 0)

# ------------------------------------------------------------
# COMPUTE Time Frequency Representation per condition
# ------------------------------------------------------------

tfr_results = {}

for condition in epochs.event_id:

    # select epochs for condition
    epochs_cond = epochs[condition]

    # compute time-frequency power per epoch
    tfr = epochs_cond.compute_tfr(
        method="morlet",
        freqs=foi,
        n_cycles=n_cycles,
        return_itc=False,
        average=False
    )

    # average across epochs
    tfr_avg = tfr.average()

    # baseline correction
    tfr_avg.apply_baseline(baseline_window, mode="percent")

    # convert to percent
    tfr_avg.data *= 100

    # store result
    tfr_results[condition] = tfr_avg

# ------------------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------------------

for condition, tfr in tfr_results.items():

    tfr.plot(
        picks=["O1", "O2", "Oz"],
        # picks="C3",   # change to O1/O2/Oz later
        title=f"TFR Power - {condition} (O1/O2/Oz)",
        # title=f"TFR Power - {condition} (C3)",
        baseline=None
    )

plt.show()