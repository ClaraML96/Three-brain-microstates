import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"

output_dir = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\erd"
os.makedirs(output_dir, exist_ok=True)

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

channels_of_interest = ["C3", "O1", "O2", "Oz"]

freq_bands = {
    "theta": (4,  7),
    "alpha": (8, 12),
    "beta":  (13, 30),
}

band_colors = {
    "theta": "steelblue",
    "alpha": "darkorange",
    "beta":  "seagreen",
}

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
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

    epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_clean-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)
    epochs.pick(channels_of_interest)

    for condition in epochs.event_id:
        print(f"  Computing TFR for condition: {condition}")
        epochs_cond = epochs[condition]

        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False
        )

        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(baseline_window, mode="percent")
        tfr_avg.data *= 100

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
# HELPER: BAND-AVERAGED ERD TIME COURSE
# ------------------------------------------------------------
def band_erd(tfr, channel, fmin, fmax):
    """Returns ERD% time course averaged across [fmin, fmax] Hz for one channel."""
    ch_idx = tfr.ch_names.index(channel)
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    return tfr.data[ch_idx, f_mask, :].mean(axis=0)

# ------------------------------------------------------------
# PLOT: one figure per condition, 2x2 subplots (one per channel)
#        each subplot shows one line per frequency band
# ------------------------------------------------------------
times = group_avg[list(group_avg.keys())[0]].times

for condition, tfr in group_avg.items():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for ax, channel in zip(axes, channels_of_interest):
        for band_name, (fmin, fmax) in freq_bands.items():
            erd = band_erd(tfr, channel, fmin, fmax)
            ax.plot(
                times, erd,
                lw=1.8,
                color=band_colors[band_name],
                label=f"{band_name} ({fmin}–{fmax} Hz)"
            )

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(baseline_window[0], baseline_window[1], color="gray", alpha=0.15)
        ax.axvline(0, color="gray", lw=0.8, ls=":")  # stimulus onset
        ax.set_title(channel)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ERD/ERS (%)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"ERD/ERS — {condition}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"erd_{condition}.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)