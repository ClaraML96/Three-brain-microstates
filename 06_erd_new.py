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

# One color per condition — adjust if you have more/fewer conditions
condition_colors = [
    "steelblue", "darkorange", "seagreen", "firebrick",
    "mediumpurple", "saddlebrown", "deeppink", "gray",
    "olive", "teal"
]

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25, 0)

# ------------------------------------------------------------
# STORAGE — keep individual subject TFRs for CI computation
# ------------------------------------------------------------
group_tfr = {}  # { condition: [tfr_sub1, tfr_sub2, ...] }

# ------------------------------------------------------------
# LOOP OVER PARTICIPANTS
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
# HELPER: band-averaged ERD per subject → mean + SEM
# ------------------------------------------------------------
def band_erd(tfr, channel, fmin, fmax):
    """Returns ERD% time course averaged across [fmin, fmax] Hz for one channel."""
    ch_idx = tfr.ch_names.index(channel)
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    return tfr.data[ch_idx, f_mask, :].mean(axis=0)  # shape: (n_times,)

def group_mean_sem(tfr_list, channel, fmin, fmax):
    """Returns mean and SEM across subjects for a given channel and frequency band."""
    # Stack ERD time courses across subjects: shape (n_subjects, n_times)
    subject_erds = np.array([band_erd(tfr, channel, fmin, fmax) for tfr in tfr_list])
    mean = subject_erds.mean(axis=0)
    sem = subject_erds.std(axis=0) / np.sqrt(len(subject_erds))
    return mean, sem

# ------------------------------------------------------------
# Get time axis from first available TFR
# ------------------------------------------------------------
first_condition = list(group_tfr.keys())[0]
times = group_tfr[first_condition][0].times
conditions = list(group_tfr.keys())

# ------------------------------------------------------------
# PLOT: one figure per channel
#        one subplot per frequency band
#        one line per condition with shaded SEM
# ------------------------------------------------------------
for channel in channels_of_interest:
    n_bands = len(freq_bands)
    fig, axes = plt.subplots(1, n_bands, figsize=(6 * n_bands, 5), sharey=True)

    for ax, (band_name, (fmin, fmax)) in zip(axes, freq_bands.items()):
        for i, condition in enumerate(conditions):
            color = condition_colors[i % len(condition_colors)]
            mean, sem = group_mean_sem(group_tfr[condition], channel, fmin, fmax)

            # Plot mean line
            ax.plot(times, mean, lw=1.8, color=color, label=condition)
            # Plot shaded confidence interval (±1 SEM)
            ax.fill_between(times, mean - sem, mean + sem, alpha=0.2, color=color)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(baseline_window[0], baseline_window[1], color="gray", alpha=0.1)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(band_name.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.set_ylabel("ERD/ERS (%)")

    # Shared legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.8)

    fig.suptitle(f"ERD/ERS — {channel}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"erd_bands_{channel}.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)