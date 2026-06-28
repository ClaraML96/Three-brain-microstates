import os
import glob
import gc
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\PreprocessedEEGData"

OUTPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\tfr_joint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))
print(f"Found {len(EPOCH_FILES)} preprocessed epoch files.")

condition_labels = {
    "T1P":   "Solo — With Feedback",
    "T1Pn":  "Solo — No Feedback",
    "T3P":   "Trio — With Feedback",
    "T3Pn":  "Trio — No Feedback",
    "T12P":  "Duo P1+P2 — With Feedback",
    "T12Pn": "Duo P1+P2 — No Feedback",
    "T13P":  "Duo P1+P3 — With Feedback",
    "T13Pn": "Duo P1+P3 — No Feedback",
    "T23P":  "Duo P2+P3 — With Feedback",
    "T23Pn": "Duo P2+P3 — No Feedback",
}

duo_merge = {
    "Duo — With Feedback": ["T12P", "T13P", "T23P"],
    "Duo — No Feedback":   ["T12Pn", "T13Pn", "T23Pn"],
}

# ------------------------------------------------------------
# CHANNELS OF INTEREST
# ------------------------------------------------------------
rois = {
    # "Sensorimotor": ["C3", "C4", "Cz", "CP3", "CP4"],
    "Sensorimotor": ["C3", "Cz", "CP3"],
    # "Occipital":    ["O1", "O2", "Oz"],
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

for epoch_file in EPOCH_FILES:
    try:
        print(f"\nProcessing file: {os.path.basename(epoch_file)}")
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

        for roi_name, roi_channels in rois.items():
            available = [ch for ch in roi_channels if ch in epochs.ch_names]
            if not available:
                print(f"  WARNING: No {roi_name} channels found — skipping ROI.")
                continue
            missing = [ch for ch in roi_channels if ch not in epochs.ch_names]
            if missing:
                print(f"  Note: {roi_name} channels not found (skipped): {missing}")

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
        print(f"  Skipping {os.path.basename(epoch_file)} due to error: {e}")
        continue

# ----------------------------
# STEP 2 — individual plots
# ----------------------------
duo_keys = {key for keys in duo_merge.values() for key in keys}

print("\n--- Individual conditions ---")
for (condition, roi_name), tfr_list in group_tfr.items():
    if condition in duo_keys:
        continue

    label = f"{condition_labels.get(condition, condition)} [{roi_name}]"
    print(f"\nPlotting: {label}")
    save_joint_plot(tfr_list, label, OUTPUT_DIR)

# ----------------------------
# STEP 3 — combined Duo plots
# ----------------------------
print("\n--- Combined Duo conditions ---")
for combined_label, source_conditions in duo_merge.items():
    for roi_name in rois:
        print(f"\nPlotting combined: {combined_label} [{roi_name}]")

        pooled = []
        for cond in source_conditions:
            key = (cond, roi_name)
            if key in group_tfr:
                pooled.extend(group_tfr[key])
            else:
                print(f"  Warning: {cond} / {roi_name} not found, skipping.")

        if not pooled:
            print(f"  No data found for {combined_label} [{roi_name}], skipping.")
            continue

        print(f"  Pooling {len(pooled)} participant-condition TFRs")
        save_joint_plot(pooled, f"{combined_label} [{roi_name}]", OUTPUT_DIR)