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
    "alpha": (8, 12),
    "beta":  (13, 30),
}

# Mapping from raw condition names to readable labels
condition_labels = {
    "Condition_0": "Solo — No Feedback",
    "Condition_1": "Solo — With Feedback",
    "Condition_2": "Duo P1+P2 — No Feedback",
    "Condition_3": "Duo P1+P2 — With Feedback",
    "Condition_4": "Duo P1+P3 — No Feedback",
    "Condition_5": "Duo P1+P3 — With Feedback",
    "Condition_6": "Duo P2+P3 — No Feedback",
    "Condition_7": "Duo P2+P3 — With Feedback",
    "Condition_8": "Trio — No Feedback",
    "Condition_9": "Trio — With Feedback",
}

# Colors matching paper style: cool for no feedback, warm for with feedback
condition_colors = {
    "Condition_0": "steelblue",
    "Condition_1": "firebrick",
    "Condition_2": "cornflowerblue",
    "Condition_3": "darkorange",
    "Condition_4": "royalblue",
    "Condition_5": "sandybrown",
    "Condition_6": "dodgerblue",
    "Condition_7": "peru",
    "Condition_8": "seagreen",
    "Condition_9": "darkred",
}

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25, 0)

# Time window to plot (matching paper)
plot_tmin, plot_tmax = 0.0, 4.0

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
group_tfr = {}

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
# HELPERS
# ------------------------------------------------------------
def band_erd(tfr, channel, fmin, fmax):
    ch_idx = tfr.ch_names.index(channel)
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    return tfr.data[ch_idx, f_mask, :].mean(axis=0)

def group_mean_sem(tfr_list, channel, fmin, fmax):
    subject_erds = np.array([band_erd(tfr, channel, fmin, fmax) for tfr in tfr_list])
    mean = subject_erds.mean(axis=0)
    sem = subject_erds.std(axis=0) / np.sqrt(len(subject_erds))
    return mean, sem

# ------------------------------------------------------------
# Get time axis and mask to 0–4s for plotting
# ------------------------------------------------------------
first_condition = list(group_tfr.keys())[0]
times = group_tfr[first_condition][0].times
time_mask = (times >= plot_tmin) & (times <= plot_tmax)
times_plot = times[time_mask]

conditions = list(group_tfr.keys())

# ------------------------------------------------------------
# PLOT: one figure per channel
#        one subplot per frequency band (alpha, beta)
#        one line per condition with shaded SEM
# ------------------------------------------------------------
for channel in channels_of_interest:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (band_name, (fmin, fmax)) in zip(axes, freq_bands.items()):
        for condition in conditions:
            if condition not in condition_labels:
                continue

            color = condition_colors.get(condition, "gray")
            label = condition_labels[condition]
            mean, sem = group_mean_sem(group_tfr[condition], channel, fmin, fmax)

            # Crop to 0–4s for plotting only
            mean_plot = mean[time_mask]
            sem_plot = sem[time_mask]

            ax.plot(times_plot, mean_plot, lw=1.8, color=color, label=label)
            ax.fill_between(times_plot, mean_plot - sem_plot, mean_plot + sem_plot,
                            alpha=0.2, color=color)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(plot_tmin, plot_tmax)
        ax.set_title(band_name.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.set_ylabel("Power (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.8,
               bbox_to_anchor=(1.18, 1.0))

    fig.suptitle(f"ERD/ERS — {channel}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"erd_alphabeta_{channel}.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)