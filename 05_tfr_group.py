import os
import numpy as np
import mne

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

# Frequencies
foi = np.linspace(1, 30, 30, dtype=int)

# Cycles
n_cycles = 3 + 0.5 * foi

# Baseline
baseline_window = (-0.25, 0)

channels_of_interest = ["C3", "O1", "O2", "Oz"]

# ------------------------------------------------------------
# COMPUTE & SAVE
# ------------------------------------------------------------
for pid, part in participants:

    print(f"\nProcessing {pid} part {part}")

    epoch_file = os.path.join(data_path, f"{pid}_p{part}_clean-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Reduce channels early
    epochs.pick(channels_of_interest)

    for condition in epochs.event_id:

        save_file = os.path.join(
            save_path,
            f"tfr_{pid}_p{part}_{condition}-tfr.h5"
        )

        if os.path.exists(save_file):
            print(f"  {condition}: already exists")
            continue

        print(f"  Computing: {condition}")

        epochs_cond = epochs[condition]

        # ----------------------------------------------------
        # STEP 1: compute TFR per epoch
        # ----------------------------------------------------
        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False
        )

        # ----------------------------------------------------
        # STEP 2: average across epochs
        # ----------------------------------------------------
        tfr_avg = tfr.average()

        # ----------------------------------------------------
        # STEP 3: baseline correction
        # ----------------------------------------------------
        tfr_avg.apply_baseline(baseline_window, mode="percent")

        # ----------------------------------------------------
        # STEP 4: convert to %
        # ----------------------------------------------------
        tfr_avg.data *= 100

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------
        tfr_avg.save(save_file, overwrite=True)

        print(f"  Saved: {save_file}")