import os
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"
)

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\tfr_joint_roi"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

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

duo_merge = {
    "Duo — With Feedback": ["Condition_2", "Condition_4", "Condition_6"],
    "Duo — No Feedback":   ["Condition_3", "Condition_5", "Condition_7"],
}

rois = {
    "Sensorimotor": ["C3", "Cz", "CP3"],
    "Occipital":    ["O1", "O2", "Oz"],
}

# Morlet parameters
foi      = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline = (-0.25, 0)

# Time-frequency points for scalp topographies in plot_joint
topo_timefreqs = [
    (0.5, 10.0),
    (1.0, 10.0),
    (2.0, 10.0),
    (3.0, 20.0),
    (4.0, 20.0),
]

vmin, vmax = -60, 60

# ------------------------------------------------------------
# HELPER — compute and save one ROI line plot (mean over ROI channels)
# ------------------------------------------------------------
def save_roi_plot(tfr_list, condition_label, roi_name, roi_channels, out_dir):
    """
    Grand-average tfr_list, pick only the ROI channels, average them,
    and plot a time-frequency power map for that ROI.
    """
    grand_avg = mne.grand_average(tfr_list)
    grand_avg_crop = grand_avg.copy().crop(tmin=0.0, tmax=4.0)

    # Pick only channels that exist in this dataset
    available_chs = grand_avg_crop.ch_names
    roi_chs_present = [ch for ch in roi_channels if ch in available_chs]

    if not roi_chs_present:
        print(f"    Warning: none of {roi_channels} found in data for {condition_label}. Skipping.")
        return

    missing = set(roi_channels) - set(roi_chs_present)
    if missing:
        print(f"    Note: channels {missing} not found; using {roi_chs_present}")

    # Average data across the ROI channels
    ch_indices = [grand_avg_crop.ch_names.index(ch) for ch in roi_chs_present]
    roi_data = grand_avg_crop.data[ch_indices].mean(axis=0)  # shape: (n_freqs, n_times)

    # Build a single-channel AverageTFR for the ROI mean
    # We reuse the info from the grand average but with one virtual channel
    roi_info = mne.create_info(
        ch_names=[roi_name],
        sfreq=grand_avg_crop.info["sfreq"],
        ch_types=["eeg"],
    )
    roi_tfr = mne.time_frequency.AverageTFRArray(
        info=roi_info,
        data=roi_data[np.newaxis, :, :],  # shape: (1, n_freqs, n_times)
        times=grand_avg_crop.times,
        freqs=grand_avg_crop.freqs,
        nave=grand_avg_crop.nave,
    )

    # Plot as a simple time-frequency image (no topomap — single virtual channel)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        roi_tfr.data[0],
        aspect="auto",
        origin="lower",
        extent=[roi_tfr.times[0], roi_tfr.times[-1],
                roi_tfr.freqs[0], roi_tfr.freqs[-1]],
        vmin=vmin, vmax=vmax,
        cmap="RdBu_r",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power change (%)", fontsize=10)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)

    ch_str = ", ".join(roi_chs_present)
    ax.set_title(
        f"TFR — {condition_label}\n"
        f"ROI: {roi_name}  [{ch_str}]  (N={len(tfr_list)})",
        fontsize=11,
    )

    # Add vertical line at stimulus onset
    ax.axvline(0, color="k", linewidth=1, linestyle="--", alpha=0.7)

    plt.tight_layout()

    safe_cond  = condition_label.replace(" ", "_").replace("/", "-").replace("—", "-")
    safe_roi   = roi_name.replace(" ", "_")
    fname      = f"{safe_cond}_{safe_roi}_roi.png"
    out_path   = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ------------------------------------------------------------
# STEP 1 — collect one averaged TFR per participant per condition
# ------------------------------------------------------------
group_tfr = {}   # group_tfr[condition_key] = list of AverageTFR

for pid, part in participants:
    print(f"\nLoading {pid} / participant {part}")
    epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)
    epochs.pick("eeg")

    for condition in epochs.event_id:
        print(f"  TFR: {condition}")
        epochs_cond = epochs[condition]

        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False,
        )

        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(baseline, mode="percent")
        tfr_avg.data *= 100   # express as % change

        group_tfr.setdefault(condition, []).append(tfr_avg)

# ------------------------------------------------------------
# STEP 2 — plot individual (non-duo) conditions, per ROI
# ------------------------------------------------------------
duo_keys = {key for keys in duo_merge.values() for key in keys}

print("\n--- Individual conditions (per ROI) ---")
for condition, tfr_list in group_tfr.items():
    if condition in duo_keys:
        continue

    label = condition_labels.get(condition, condition)
    print(f"\nCondition: {label}")

    for roi_name, roi_channels in rois.items():
        print(f"  ROI: {roi_name}")
        save_roi_plot(tfr_list, label, roi_name, roi_channels, OUTPUT_DIR)

# ------------------------------------------------------------
# STEP 3 — combined Duo plots, per ROI
# ------------------------------------------------------------
print("\n--- Combined Duo conditions (per ROI) ---")
for combined_label, source_conditions in duo_merge.items():
    print(f"\nCombined: {combined_label}")

    pooled = []
    for cond in source_conditions:
        if cond in group_tfr:
            pooled.extend(group_tfr[cond])
        else:
            print(f"  Warning: {cond} not found in data, skipping.")

    if not pooled:
        print(f"  No data found for {combined_label}, skipping.")
        continue

    print(f"  Pooling {len(pooled)} participant-condition TFRs from {source_conditions}")

    for roi_name, roi_channels in rois.items():
        print(f"  ROI: {roi_name}")
        save_roi_plot(pooled, combined_label, roi_name, roi_channels, OUTPUT_DIR)

print("\nDone. All ROI figures saved to:", OUTPUT_DIR)