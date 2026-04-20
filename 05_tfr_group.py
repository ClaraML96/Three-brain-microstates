import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

channels_of_interest = ["C3", "O1", "O2", "Oz"]

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25, 0)

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
group_tfr = {}  # { condition: [tfr_avg_sub1, tfr_avg_sub2, ...] }

# ------------------------------------------------------------
# LOOP OVER SUBJECTS
# ------------------------------------------------------------
for pid, part in participants:
    print(f"\nProcessing participant {pid}, part {part}")

    epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_clean-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Pick only channels of interest early to save memory
    epochs.pick(channels_of_interest)

    for condition in epochs.event_id:
        print(f"  Computing TFR for condition: {condition}")
        epochs_cond = epochs[condition]

        # STEP 1: Compute TFR per epoch (average=False keeps individual epochs)
        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False       # Keep individual epochs
        )

        # STEP 2: Average across epochs
        tfr_avg = tfr.average()

        # STEP 3: Baseline correction (percent = (A - R) / R)
        tfr_avg.apply_baseline(baseline_window, mode="percent")

        # STEP 4: Multiply by 100 to get actual percentage (ERD/ERS %)
        tfr_avg.data *= 100

        # Store per subject
        if condition not in group_tfr:
            group_tfr[condition] = []
        group_tfr[condition].append(tfr_avg)

# ------------------------------------------------------------
# GRAND AVERAGE ACROSS SUBJECTS
# ------------------------------------------------------------
group_avg = {}
for condition, tfr_list in group_tfr.items():
    group_avg[condition] = mne.grand_average(tfr_list)
    print(f"Grand average computed for condition: {condition}")

# ------------------------------------------------------------
# PLOT - one figure per condition, one subplot per channel
# ------------------------------------------------------------
output_dir = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\tfr"
os.makedirs(output_dir, exist_ok=True)

for condition, tfr in group_avg.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, channel in zip(axes, channels_of_interest):
        tfr.plot(
            picks=[channel],
            axes=ax,
            show=False,
            colorbar=False
        )
        ax.set_title(channel) 

    fig.suptitle(f"Group TFR — {condition}", fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"tfr_group_{condition}.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)