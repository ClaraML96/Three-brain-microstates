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
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet"
    r"\Skrivebord\DTU\Human Centeret Artificial Intelligence"
    r"\Thesis\data\ica_cleaned"
)

OUTPUT_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet"
    r"\Skrivebord\DTU\Human Centeret Artificial Intelligence"
    r"\Thesis\figures\erd_joint"
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

# Morlet parameters
foi        = np.linspace(1, 30, 30, dtype=int)
n_cycles   = 3 + 0.5 * foi
baseline   = (-0.25, 0)

# Time-frequency points shown as scalp topographies in plot_joint.
# Each entry is a (time_s, freq_hz) tuple.
# The freq here is the centre of the band shown in the topomap.
# Adjust times and frequencies to match your regions of interest.
topo_timefreqs = [
    (0.5, 10.0),   # alpha at 0.5 s
    (1.0, 10.0),   # alpha at 1.0 s
    (2.0, 10.0),   # alpha at 2.0 s
    (3.0, 20.0),   # beta  at 3.0 s
    (4.0, 20.0),   # beta  at 4.0 s
]

# plot_joint colour limits (percent change from baseline)
# None = automatic; set manually if plots look washed out
vmin, vmax = -60, 60

# ------------------------------------------------------------
# STEP 1 — collect one averaged TFR per participant per condition
# ------------------------------------------------------------
# group_tfr[condition] = list of AverageTFR (one per participant)
group_tfr = {}

for pid, part in participants:
    print(f"\nLoading {pid} / participant {part}")
    epoch_file = os.path.join(DATA_DIR, f"{pid}_p{part}_ica_cleaned-epo.fif")
    epochs = mne.read_epochs(epoch_file, preload=True)

    # Keep only EEG channels (drop Status / stim)
    epochs.pick("eeg")

    for condition in epochs.event_id:
        print(f"  TFR: {condition}")
        epochs_cond = epochs[condition]

        tfr = epochs_cond.compute_tfr(
            method="morlet",
            freqs=foi,
            n_cycles=n_cycles,
            return_itc=False,
            average=False,       # keep single-trial to average across participants
        )

        tfr_avg = tfr.average()
        tfr_avg.apply_baseline(baseline, mode="percent")
        tfr_avg.data *= 100      # express as % change

        group_tfr.setdefault(condition, []).append(tfr_avg)

# ------------------------------------------------------------
# STEP 2 — grand-average across participants and plot_joint
# ------------------------------------------------------------
for condition, tfr_list in group_tfr.items():
    label = condition_labels.get(condition, condition)
    print(f"\nPlotting: {label}")

    # Grand average — MNE averages .data and keeps the Info
    grand_avg = mne.grand_average(tfr_list)

    # crop to motor period for display (baseline is still applied above)
    grand_avg_crop = grand_avg.copy().crop(tmin=0.0, tmax=4.0)

    fig = grand_avg_crop.plot_joint(
        tmin=0.0,
        tmax=4.0,
        fmin=foi[0],
        fmax=foi[-1],
        timefreqs=topo_timefreqs,  # (time, freq) tuples — required by MNE
        topomap_args=dict(vlim=(vmin, vmax)),
        title=f"ERD/ERS — {label}  (N={len(tfr_list)})",
        show=False,
    )

    fname = condition.replace(" ", "_").replace("/", "-") + "_joint.png"
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone. All figures saved to:", OUTPUT_DIR)