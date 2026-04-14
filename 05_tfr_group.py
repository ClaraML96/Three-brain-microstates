import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

Path = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"

participants = [
    ("301",1),
    ("301",2),
    ("301",3),

    # ("302",1),
    # ("302",2),
    # ("302",3),
    
    # ("303",1),
    # ("303",2),
    # ("303",3),
    
    # ("304",1),
    # ("304",2),
    # ("304",3),
]

# Morlet parameters
foi = np.linspace(1,30,30,dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25,0)

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

group_tfr = {}

# ------------------------------------------------------------
# LOOP OVER SUBJECTS
# ------------------------------------------------------------

for pid, part in participants:

    print(f"\nProcessing {pid} participant {part}")

    epoch_file = os.path.join(
        Path,
        f"{pid}_p{part}_clean-epo.fif"
    )

    epochs = mne.read_epochs(epoch_file, preload=True)

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

        # Computing ERD
        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(baseline_window, mode="percent")
        tfr_avg.data *= 100

        # Store for group averaging
        if condition not in group_tfr:
            group_tfr[condition] = []

        group_tfr[condition].append(tfr_avg)

# ------------------------------------------------------------
# AVERAGE ACROSS SUBJECTS
# ------------------------------------------------------------

group_avg = {}

for condition, tfr_list in group_tfr.items():

    group_avg[condition] = mne.grand_average(tfr_list)

# ------------------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------------------

for condition, tfr in group_avg.items():

    tfr.plot(
        picks="C3", # change to O1/O2/Oz later
        title=f"Group TFR ({condition})"
    )

plt.show()