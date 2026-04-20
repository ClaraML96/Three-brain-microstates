import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
save_path = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed\tfr"
erd_path = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\erd"
tfr_path = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\tfr"

participants = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

# Auto-discovered from your filenames
conditions = [f"Condition_{i}" for i in range(10)]

freq_bands = {
    "theta": (4,  7),
    "alpha": (8, 12),
    "beta":  (13, 30),
}

channels_of_interest = ["C3", "O1", "O2", "Oz"]

# ------------------------------------------------------------
# LOAD & GROUP
# ------------------------------------------------------------
group_tfr = {}  # {condition: [tfr_sub1, tfr_sub2, ...]}

for pid, part in participants:
    for condition in conditions:
        fname = os.path.join(save_path, f"tfr_{pid}_p{part}_{condition}-tfr.h5")
        if not os.path.exists(fname):
            print(f"Missing: {fname}, skipping.")
            continue

        tfr_list = mne.time_frequency.read_tfrs(fname)
        tfr = tfr_list[0] if isinstance(tfr_list, list) else tfr_list
        group_tfr.setdefault(condition, []).append(tfr)

print("\nLoaded conditions:")
for cond, lst in group_tfr.items():
    print(f"  {cond}: {len(lst)} subjects")

# ------------------------------------------------------------
# GROUP AVERAGE
# ------------------------------------------------------------
group_avg = {
    condition: mne.grand_average(tfr_list)
    for condition, tfr_list in group_tfr.items()
}

# ------------------------------------------------------------
# HELPER: BAND-AVERAGED ERD TIME COURSE
# ------------------------------------------------------------
def band_erd(tfr, channel, fmin, fmax):
    """Returns ERD% time course averaged across [fmin, fmax] Hz for one channel."""
    ch_idx = tfr.ch_names.index(channel)
    f_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    return tfr.data[ch_idx, f_mask, :].mean(axis=0)  # shape: (n_times,)

# ------------------------------------------------------------
# PLOT: BAND ERD TIME COURSES — one figure per channel
# ------------------------------------------------------------
for channel in channels_of_interest:
    n_bands = len(freq_bands)
    n_conds = len(group_avg)

    fig, axes = plt.subplots(
        n_bands, n_conds,
        figsize=(4 * n_conds, 3 * n_bands),
        sharex=True, sharey="row"
    )

    # Always work with a 2D array
    if n_bands == 1:
        axes = axes[np.newaxis, :]
    if n_conds == 1:
        axes = axes[:, np.newaxis]

    for col, (condition, tfr) in enumerate(group_avg.items()):
        for row, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
            ax = axes[row, col]
            erd = band_erd(tfr, channel, fmin, fmax)

            ax.plot(tfr.times, erd, lw=1.8, color="steelblue")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.axvspan(-0.25, 0, color="gray", alpha=0.15, label="baseline")
            ax.axvline(0, color="gray", lw=0.8, ls=":")  # stimulus onset

            if row == 0:
                ax.set_title(condition, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{band_name}\n({fmin}–{fmax} Hz)\nERD (%)", fontsize=9)
            if row == n_bands - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle(f"Band ERD — {channel}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(erd_path, f"erd_band_{channel}.png"), dpi=150)
    print(f"Saved figure: erd_band_{channel}.png")

# ------------------------------------------------------------
# Plotting TFR power spectrum distribution
# ------------------------------------------------------------
for channel in channels_of_interest:
    for condition, tfr in group_avg.items():
        fig = tfr.plot(
            picks=channel,
            title=f"Group TFR — {channel} | {condition}",
            vlim=(-50, 50),      # ← replaces vmin/vmax in newer MNE
            cmap="RdBu_r",
            show=False,
        )
        out_name = f"tfr_heatmap_{channel}_{condition}.png"
        fig[0].savefig(os.path.join(tfr_path, out_name), dpi=150)
        plt.close(fig[0])
        print(f"Saved: {out_name}")
