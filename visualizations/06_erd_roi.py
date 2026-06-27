import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import glob

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\PreprocessedEEGData"
EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))
print(f"Found {len(EPOCH_FILES)} epoch files")

output_dir = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\erd"
os.makedirs(output_dir, exist_ok=True)

freq_bands = {
    "alpha": (8, 12),
    "beta":  (13, 30),
}

condition_remap = {
    "T12P":  "Duo_With_Feedback",
    "T13P":  "Duo_With_Feedback",
    "T23P":  "Duo_With_Feedback",
    "T12Pn": "Duo_No_Feedback",
    "T13Pn": "Duo_No_Feedback",
    "T23Pn": "Duo_No_Feedback",
}

condition_labels = {
    "T1P":               "Solo — With Feedback",
    "T1Pn":              "Solo — No Feedback",
    "T3P":               "Trio — With Feedback",
    "T3Pn":              "Trio — No Feedback",
    "Duo_With_Feedback": "Duo — With Feedback",
    "Duo_No_Feedback":   "Duo — No Feedback",
}

condition_colors = {
    "T1P":               "firebrick",
    "T1Pn":              "steelblue",
    "T3P":               "darkred",
    "T3Pn":              "seagreen",
    "Duo_With_Feedback": "darkorange",
    "Duo_No_Feedback":   "cornflowerblue",
}

# ------------------------------------------------------------
# CHANNELS OF INTEREST
# ------------------------------------------------------------
rois = {
    "Sensorimotor": ["C3", "C4", "Cz", "CP3", "CP4"],  # adjust to what's in your data
    "Occipital":    ["O1", "O2", "Oz"],
}

# Morlet parameters
foi = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline_window = (-0.25, 0)

# Time window to plot
plot_tmin, plot_tmax = 0.0, 4.0

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
group_tfr = {}

# ------------------------------------------------------------
# LOAD PREPROCESSED FILES
# ------------------------------------------------------------
# Flatten all ROI channels into one list for loading
all_roi_channels = [ch for channels in rois.values() for ch in channels]

for epoch_file in EPOCH_FILES:
    if not os.path.isfile(epoch_file):
        raise FileNotFoundError(f"Could not find epochs file: {epoch_file}")

    print(f"\nProcessing file: {os.path.basename(epoch_file)}")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Pick only the channels of interest that exist in this recording
    available = [ch for ch in all_roi_channels if ch in epochs.ch_names]
    if not available:
        print(f"  WARNING: No channels of interest found — skipping file.")
        continue
    epochs.pick(available)

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

        storage_key = condition_remap.get(condition, condition)

        if storage_key not in group_tfr:
            group_tfr[storage_key] = []
        group_tfr[storage_key].append(tfr_avg)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def band_erd_roi(tfr, fmin, fmax, roi_channels):
    """
    Average power in a frequency band over a subset of channels.
    Channels missing from the recording are silently skipped.
    """
    available = [ch for ch in roi_channels if ch in tfr.ch_names]
    if not available:
        raise ValueError(
            f"None of the requested channels {roi_channels} "
            f"were found in the data. Available channels: {tfr.ch_names}"
        )
    ch_idx = [tfr.ch_names.index(ch) for ch in available]
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    return tfr.data[ch_idx][:, f_mask, :].mean(axis=(0, 1))


def group_mean_sem_roi(tfr_list, fmin, fmax, roi_channels):
    subject_erds = np.array(
        [band_erd_roi(tfr, fmin, fmax, roi_channels) for tfr in tfr_list]
    )
    mean = subject_erds.mean(axis=0)
    sem  = subject_erds.std(axis=0) / np.sqrt(len(subject_erds))
    return mean, sem

# ------------------------------------------------------------
# Get time axis and mask to 0–4 s for plotting
# ------------------------------------------------------------
first_condition = list(group_tfr.keys())[0]
times = group_tfr[first_condition][0].times
time_mask = (times >= plot_tmin) & (times <= plot_tmax)
times_plot = times[time_mask]

conditions = list(group_tfr.keys())

# ------------------------------------------------------------
# PLOT: one figure, two subplots (alpha, beta)
# All channels of interest averaged together
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for roi_name, roi_channels in rois.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (band_name, (fmin, fmax)) in zip(axes, freq_bands.items()):
        for condition in conditions:
            if condition not in condition_labels:
                continue

            color = condition_colors.get(condition, "gray")
            label = condition_labels[condition]

            mean, sem = group_mean_sem_roi(
                group_tfr[condition], fmin, fmax, roi_channels
            )

            mean_plot = mean[time_mask]
            sem_plot  = sem[time_mask]

            ax.plot(times_plot, mean_plot, lw=1.8, color=color, label=label)
            ax.fill_between(times_plot, mean_plot - sem_plot, mean_plot + sem_plot,
                            alpha=0.2, color=color)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(plot_tmin, plot_tmax)
        ax.set_title(band_name.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.set_ylabel("Power change (%)")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=8, framealpha=0.8,
               bbox_to_anchor=(1.18, 1.0))

    # Title now clearly states the ROI and which channels
    ch_str = ", ".join(roi_channels)
    fig.suptitle(f"ERD/ERS — {roi_name} ROI ({ch_str})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = f"erd_ers_allparticipants_{roi_name}.png"
    output_file = os.path.join(output_dir, fname)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)