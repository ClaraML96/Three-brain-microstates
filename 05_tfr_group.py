import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

data_path = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"
save_path = os.path.join(data_path, "tfr")
os.makedirs(save_path, exist_ok=True)

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

# Morlet parameters
foi = np.linspace(1,30,30,dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25,0)

channels_of_interest = ["C3", "O1", "O2", "Oz"]

# ------------------------------------------------------------
# COMPUTE & SAVE
# ------------------------------------------------------------
for pid, part in participants:
    print(f"\n{pid} part {part}")

    epoch_file = os.path.join(data_path, f"{pid}_p{part}_clean-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Restrict to channels of interest early → saves memory & time
    epochs.pick(channels_of_interest)

    for condition in epochs.event_id:
        save_file = os.path.join(
            save_path,
            f"tfr_{pid}_p{part}_{condition}-tfr.h5"   # MNE's native TFR format
        )

        if os.path.exists(save_file):
            print(f"  {condition}: already exists, skipping.")
            continue

        print(f"  Computing TFR for condition: {condition}")
        epochs_cond = epochs[condition]

        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False,          
        )

        # Average across epochs, then baseline-correct → ERD%
        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(baseline_window, mode="percent")
        tfr_avg.data *= 100         # now in ERD% units

        tfr_avg.save(save_file, overwrite=True)
        print(f"  Saved → {save_file}")