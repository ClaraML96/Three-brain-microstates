import os
import gc
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
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\tfr_joint"
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

# ------------------------------------------------------------
# CHANNELS OF INTEREST
# ------------------------------------------------------------
rois = {
    "Sensorimotor": ["C3", "Cz", "CP3"],
    "Occipital":    ["O1", "O2", "Oz"],
}

foi      = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline = (-0.25, 0)

topo_timefreqs = [
    (0.5, 10.0),
    (1.0, 10.0),
    (2.0, 10.0),
    (3.0, 20.0),
    (4.0, 20.0),
]

vmin, vmax = -60, 60

# ------------------------------------------------------------
# HELPER — compute and save one plot_joint figure
# ------------------------------------------------------------
def save_joint_plot(tfr_list, label, out_dir):
    """Grand-average tfr_list and save a plot_joint figure."""
    if not tfr_list:
        print(f"  Skipping plot for {label}: No data collected.")
        return

    grand_avg = mne.grand_average(tfr_list)
    grand_avg_crop = grand_avg.copy().crop(tmin=0.0, tmax=4.0)

    fig = grand_avg_crop.plot_joint(
        tmin=0.0,
        tmax=4.0,
        fmin=foi[0],
        fmax=foi[-1],
        timefreqs=topo_timefreqs,
        topomap_args=dict(vlim=(vmin, vmax)),
        title=f"TFR — {label}  (N={len(tfr_list)})",
        show=False,
    )

    fname = label.replace(" ", "_").replace("/", "-").replace("—", "-") + "_joint_ROI.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ------------------------------------------------------------
# STEP 1 — collect one averaged TFR per participant per condition
# ------------------------------------------------------------
group_tfr = {}  # keyed by (condition, roi_name)

for pid, part in participants:
    try:
        print(f"\nLoading {pid} / participant {part}")
        epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif")
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
        epochs.pick("eeg")

        for roi_name, roi_channels in rois.items():
            available = [ch for ch in roi_channels if ch in epochs.ch_names]
            if not available:
                print(f"  WARNING: No {roi_name} channels found — skipping ROI.")
                continue
            
            missing = [ch for ch in roi_channels if ch not in epochs.ch_names]
            if missing:
                print(f"  Note: {roi_name} channels not found (skipped): {missing}")

            # Crucial part: Pick ONLY the channels matching this ROI 
            epochs_roi = epochs.copy().pick(available)

            for condition in epochs_roi.event_id:
                if condition not in condition_labels:
                    continue

                print(f"  Computing TFR for condition: {condition}, ROI: {roi_name}")
                epochs_cond = epochs_roi[condition]

                tfr = epochs_cond.compute_tfr(
                    method="morlet",
                    freqs=foi,
                    n_cycles=n_cycles,
                    return_itc=False,
                    average=False,
                    verbose=False
                )

                tfr_avg = tfr.average()
                tfr_avg.apply_baseline(baseline, mode="percent", verbose=False)
                tfr_avg.data *= 100
                tfr_avg.data = tfr_avg.data.astype(np.float32)

                group_tfr.setdefault((condition, roi_name), []).append(tfr_avg)

        del epochs
        gc.collect()

    except Exception as e:
        print(f"  Skipping {pid}_p{part} due to error: {e}")
        continue

# ----------------------------
# STEP 2 — individual plots
# ----------------------------
duo_keys = {key for keys in duo_merge.values() for key in keys}

print("\n--- Individual conditions ---")
for (condition, roi_name), tfr_list in group_tfr.items():
    if condition in duo_keys:
        continue

    base_label = condition_labels.get(condition, condition)
    label = f"{base_label} — {roi_name}"
    print(f"\nPlotting: {label}")
    save_joint_plot(tfr_list, label, OUTPUT_DIR)

# ----------------------------
# STEP 3 — combined Duo plots
# ----------------------------
print("\n--- Combined Duo conditions ---")
for combined_label, source_conditions in duo_merge.items():
    for roi_name in rois:
        display_label = f"{combined_label} — {roi_name}"
        print(f"\nPlotting combined: {display_label}")

        pooled = []
        for cond in source_conditions:
            key = (cond, roi_name)
            if key in group_tfr:
                pooled.extend(group_tfr[key])
            else:
                print(f"  Warning: {cond} / {roi_name} not found, skipping.")

        if not pooled:
            print(f"  No data found for {display_label}, skipping.")
            continue

        print(f"  Pooling {len(pooled)} participant-condition TFRs")
        save_joint_plot(pooled, display_label, OUTPUT_DIR)

print("\nDone. All plot_joint ROI figures saved to:", OUTPUT_DIR)