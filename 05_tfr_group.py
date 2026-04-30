import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

channels_of_interest = ["C3", "O1", "O2", "Oz"]

condition_labels = {
    "Condition_0": "Solo — With Feedback",
    "Condition_1": "Solo — No Feedback",
    "Condition_2": "Duo P1+P2 — With Feedback",
    "Condition_3": "Duo P1+P2 — No Feedback",
    "Condition_4": "Duo P1+P3 — With Feedback",
    "Condition_5": "Duo P1+P3 — No Feedback",
    "Condition_6": "Duo P2+P3 — With Feedback",
    "Condition_7": "Duo P2+P3 — No Feedback",
    "Condition_8": "Trio — With Feedback",
    "Condition_9": "Trio — No Feedback",
}

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
# wavelet_length = (5/np.pi)*(n_cycles*sfreq)/foi-1 
baseline_window = (-0.25, 0)

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
group_tfr = {}  

# ------------------------------------------------------------
# LOOP OVER SUBJECTS
# ------------------------------------------------------------
for pid, part in participants:
    print(f"\nProcessing participant {pid}, part {part}")

    epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Pick only channels of interest 
    epochs.pick(channels_of_interest)

    for condition in epochs.event_id:
        print(f"  Computing TFR for condition: {condition}")
        epochs_cond = epochs[condition]

        # Compute TFR per epoch
        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False       
        )

        # Average across epochs
        tfr_avg = tfr.average()

        # Baseline correction (percent = (A - R) / R)
        tfr_avg.apply_baseline(baseline_window, mode="percent")

        # Multiply by 100 to get actual percentage (ERD/ERS %)
        tfr_avg.data *= 100

        # Store per subject
        if condition not in group_tfr:
            group_tfr[condition] = []
        group_tfr[condition].append(tfr_avg)

# ------------------------------------------------------------
# DIAGNOSTIC: find which participant drives the beta burst
# ------------------------------------------------------------
channel = "C3"
fmin, fmax = 13, 30  # beta band

if "Condition_5" in group_tfr and group_tfr["Condition_5"]:
    ch_idx_check = group_tfr["Condition_5"][0].ch_names.index(channel)

    print("\nBeta power max per subject per condition (C3):")
    for condition in ["Condition_3", "Condition_5", "Condition_7"]:  # No feedback duo conditions
        if condition not in group_tfr:
            continue
        print(f"\n  {condition_labels[condition]}:")
        for i, tfr in enumerate(group_tfr[condition]):
            f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
            beta_data = tfr.data[ch_idx_check, f_mask, :]
            max_val = beta_data.max()
            print(f"    Subject {i} ({participants[i]}): max beta = {max_val:.1f}%")
else:
    print("\nSkipping beta diagnostic: Condition_5 is missing from group_tfr.")

# ------------------------------------------------------------
# AVERAGE ACROSS SUBJECTS
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

    label = condition_labels.get(condition, condition)
    fig.suptitle(f"Group TFR — {label}", fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"tfr_group_{condition}.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)