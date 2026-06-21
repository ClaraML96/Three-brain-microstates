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

# Which conditions to merge into a single "Duo" plot.
# Keys are the new combined label; values are the original condition keys to pool.
duo_merge = {
    "Duo — With Feedback": ["Condition_2", "Condition_4", "Condition_6"],
    "Duo — No Feedback":   ["Condition_3", "Condition_5", "Condition_7"],
}

# Morlet parameters
foi      = np.linspace(1, 30, 30, dtype=int)
n_cycles = 3 + 0.5 * foi
baseline = (-0.25, 0)

# Time-frequency points shown as scalp topographies in plot_joint.
# Each entry is a (time_s, freq_hz) tuple.
topo_timefreqs = [
    (0.5, 10.0),   # alpha at 0.5 s
    (1.0, 10.0),   # alpha at 1.0 s
    (2.0, 10.0),   # alpha at 2.0 s
    (3.0, 20.0),   # beta  at 3.0 s
    (4.0, 20.0),   # beta  at 4.0 s
]

# plot_joint colour limits (percent change from baseline)
vmin, vmax = -60, 60

# ------------------------------------------------------------
# HELPER — compute and save one plot_joint figure
# ------------------------------------------------------------
def save_joint_plot(tfr_list, label, out_dir):
    """Grand-average tfr_list and save a plot_joint figure."""
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

    fname = label.replace(" ", "_").replace("/", "-").replace("—", "-") + "_joint.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ------------------------------------------------------------
# STEP 1 — collect one averaged TFR per participant per condition
# ------------------------------------------------------------
# group_tfr[condition_key] = list of AverageTFR (one per participant)
group_tfr = {}

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
        tfr_avg.data *= 100      # express as % change

        group_tfr.setdefault(condition, []).append(tfr_avg)

# ------------------------------------------------------------
# STEP 2 — plot individual (non-duo) conditions
# ------------------------------------------------------------
# Collect all condition keys that belong to a duo merge group
duo_keys = {key for keys in duo_merge.values() for key in keys}

print("\n--- Individual conditions ---")
for condition, tfr_list in group_tfr.items():
    if condition in duo_keys:
        continue                          # handled separately below

    label = condition_labels.get(condition, condition)
    print(f"\nPlotting: {label}")
    save_joint_plot(tfr_list, label, OUTPUT_DIR)

# ------------------------------------------------------------
# STEP 3 — combined Duo plots
# ------------------------------------------------------------
print("\n--- Combined Duo conditions ---")
for combined_label, source_conditions in duo_merge.items():
    print(f"\nPlotting combined: {combined_label}")

    # Pool all participant-level AverageTFR objects from the relevant conditions.
    # Each entry in group_tfr[cond] is already baseline-corrected % change for
    # one participant, so pooling them and calling grand_average gives the mean
    # across all participants × duo-pair combinations.
    pooled = []
    for cond in source_conditions:
        if cond in group_tfr:
            pooled.extend(group_tfr[cond])
        else:
            print(f"  Warning: {cond} not found in data, skipping.")

    if not pooled:
        print(f"  No data found for {combined_label}, skipping.")
        continue

    print(f"  Pooling {len(pooled)} participant-condition TFRs "
          f"from {source_conditions}")
    save_joint_plot(pooled, combined_label, OUTPUT_DIR)

print("\nDone. All figures saved to:", OUTPUT_DIR)