import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\PreprocessedEEGData"
OUTPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\erp"

# DYNAMICALLY FIND ALL PARTICIPANT FILES (Replaces the hardcoded list)
EPOCH_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif")))
print(f"Found {len(EPOCH_FILES)} ICA-cleaned epoch files.")

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
    "T1P":             "Solo — With Feedback",
    "T1Pn":            "Solo — No Feedback",
    "T3P":             "Trio — With Feedback",
    "T3Pn":            "Trio — No Feedback",
    "Duo_With_Feedback": "Duo — With Feedback",
    "Duo_No_Feedback":   "Duo — No Feedback",
}

condition_colors = {
    "T1P":             "firebrick",
    "T1Pn":            "steelblue",
    "T3P":             "darkred",
    "T3Pn":            "seagreen",
    "Duo_With_Feedback": "darkorange",
    "Duo_No_Feedback":   "cornflowerblue",
}

# Mapping conditions back to your structure
CONDITION_MAP = {
    "with_feedback/solo":    "T1P",
    "with_feedback/trio":    "T3P",
    "without_feedback/solo": "T1Pn",
    "without_feedback/trio": "T3Pn",
}
EVENT_ID = CONDITION_MAP

OCCIPITAL_CHANNELS = ["O1", "O2", "Oz"]
MOTOR_CHANNEL      = "C3"

# Time window 
TMIN, TMAX   = -0.5, 5.5
BASELINE     = (-0.5, 0.0)   # seconds; set to None to skip
SMOOTH_SIGMA = 5              # Gaussian smoothing in samples (0 = off)
ERROR_TYPE   = "ci95"         # "se" or "ci95" -- 95% CI shown in grand average

# Minimum trials a participant must have in a condition to be included
MIN_TRIALS = 10

PALETTE = {
    "solo":             "#2271B5",
    "trio":             "#E65F2B",
    "with_feedback":    "#1B7837",
    "without_feedback": "#762A83",
}
ALPHA_FILL    = 0.35   # opacity of the CI band
CI_EDGE_ALPHA = 0.7    # opacity of dashed CI boundary lines
CI_LINEWIDTH  = 0.8    # width of dashed CI boundary lines

# ============================================================
# PUBLICATION STYLE
# ============================================================

STYLE = {
    "font.family":      "serif",
    "font.serif":       ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.9,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.frameon":   False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
}

# ============================================================
# DATA UTILITIES
# ============================================================

def load_cleaned_epochs(fname):
    """Load ICA-cleaned epochs and attach standard 10-20 montage directly from path."""
    epochs = mne.read_epochs(fname, preload=True, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="ignore")
    return epochs

def select_condition(epochs, condition_key, event_id):
    """Return epochs matching a condition key directly using string keys."""
    mne_key = event_id.get(condition_key)
    if mne_key is None:
        raise KeyError(f"'{condition_key}' not in CONDITION_MAP.")
    
    # Check if this specific condition exists in this participant's file
    if mne_key not in epochs.event_id:
        return epochs[epochs.events[:, 2] == -1]   # return guaranteed empty epochs
        
    return epochs[mne_key]

def extract_channel_data(epochs, channels, baseline):
    """Extract and baseline-correct channel data from epochs."""
    if isinstance(channels, str):
        channels = [channels]

    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available:
        raise ValueError(f"None of {channels} found. Available: {epochs.ch_names[:10]}")

    picks = mne.pick_channels(epochs.ch_names, include=available)
    data  = epochs.get_data(picks=picks)      # (n_trials, n_ch, n_times)
    data  = data.mean(axis=1)                 # -> (n_trials, n_times)

    if baseline is not None:
        tmin_bl, tmax_bl = baseline
        bl_mask = (epochs.times >= tmin_bl) & (epochs.times <= tmax_bl)
        data    = data - data[:, bl_mask].mean(axis=1, keepdims=True)

    return epochs.times, data

# ============================================================
# AVERAGE UTILITIES
# ============================================================

def compute_grand_average(participant_erps):
    """Compute grand average and cross-participant variability."""
    if len(participant_erps) == 0:
        return {"mean": np.array([]), "lower": np.array([]), "upper": np.array([]), "n_participants": 0}
        
    stack = np.array(participant_erps)          # (n_participants, n_times)
    n     = stack.shape[0]
    mean  = stack.mean(axis=0)
    se    = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci    = 1.96 * se if ERROR_TYPE == "ci95" else se

    if SMOOTH_SIGMA > 0:
        lower = mean - ci
        upper = mean + ci
        mean  = gaussian_filter1d(mean,   SMOOTH_SIGMA)
        lower = gaussian_filter1d(lower, SMOOTH_SIGMA)
        upper = gaussian_filter1d(upper, SMOOTH_SIGMA)
    else:
        lower = mean - ci
        upper = mean + ci

    return {"mean": mean, "lower": lower, "upper": upper, "n_participants": n}

# ============================================================
# PLOTTING UTILITIES
# ============================================================

def _annotation():
    band = ("SE (across participants)" if ERROR_TYPE == "se" else "95% CI (across participants)")
    return (f"Shaded region: +/- {band}   |   "
            f"Baseline: {BASELINE[0]} to {BASELINE[1]} s   |   "
            f"Smoothing sigma={SMOOTH_SIGMA} samples")

def _draw_grand_avg_axes(ax, times, stats_a, stats_b,
                         label_a, label_b, color_a, color_b,
                         title, show_xlabel=True, show_legend=True):
    """Draw two grand-average ERP lines with cross-participant error bands."""
    for stats, label, color in [
        (stats_a, label_a, color_a),
        (stats_b, label_b, color_b),
    ]:
        n    = stats["n_participants"]
        if n == 0:
            continue
        mean = stats["mean"] * 1e6
        lo   = stats["lower"] * 1e6
        hi   = stats["upper"] * 1e6

        ax.plot(times, mean, color=color, linewidth=2.0, label=f"{label}  (n={n})", zorder=3)
        ax.fill_between(times, lo, hi, color=color, alpha=ALPHA_FILL, zorder=2)
        ax.plot(times, lo, color=color, linewidth=CI_LINEWIDTH, linestyle="--", alpha=CI_EDGE_ALPHA, zorder=2)
        ax.plot(times, hi, color=color, linewidth=CI_LINEWIDTH, linestyle="--", alpha=CI_EDGE_ALPHA, zorder=2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey",  linewidth=0.5, linestyle="-",  alpha=0.3)
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", pad=8)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    if show_legend:
        ax.legend(loc="upper right")

def plot_grand_avg_figure(times, grand_avg, channel_label, comparison="solo_vs_trio"):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

        if comparison == "solo_vs_trio":
            fig.suptitle(f"Grand Average ERP -- {channel_label} for all participants |  Solo vs Trio", fontsize=14, fontweight="bold", y=0.98)
            panels = [
                (axes[0], "with_feedback",    "With Feedback"),
                (axes[1], "without_feedback", "Without Feedback"),
            ]
            for ax, fb_key, panel_title in panels:
                _draw_grand_avg_axes(
                    ax, times,
                    grand_avg[f"{fb_key}/solo"],
                    grand_avg[f"{fb_key}/trio"],
                    label_a="Solo", label_b="Trio",
                    color_a=PALETTE["solo"], color_b=PALETTE["trio"],
                    title=panel_title,
                    show_xlabel=True,
                    show_legend=(ax is axes[0]),
                )
            axes[1].legend(loc="upper right")
        return fig

def plot_grand_avg_combined(times, grand_avg_occ, grand_avg_motor):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
        fig.suptitle("Grand Average ERP Overview for all participants  |  Solo vs Trio", fontsize=14, fontweight="bold", y=0.98)

        row_data = [
            (grand_avg_occ,   f"Occipital ({', '.join(OCCIPITAL_CHANNELS)} avg)"),
            (grand_avg_motor, f"Motor ({MOTOR_CHANNEL})"),
        ]
        col_data = [
            ("with_feedback",    "With Feedback"),
            ("without_feedback", "Without Feedback"),
        ]

        for row, (grand_avg, row_label) in enumerate(row_data):
            for col, (fb_key, panel_title) in enumerate(col_data):
                ax = axes[row][col]
                _draw_grand_avg_axes(
                    ax, times,
                    grand_avg[f"{fb_key}/solo"],
                    grand_avg[f"{fb_key}/trio"],
                    label_a="Solo", label_b="Trio",
                    color_a=PALETTE["solo"], color_b=PALETTE["trio"],
                    title=panel_title if row == 0 else "",
                    show_xlabel=(row == 1),
                    show_legend=(row == 0 and col == 0),
                )
                if col == 0:
                    ax.set_ylabel(f"{row_label}\nAmplitude (uV)")

        fig.text(0.5, -0.02, _annotation(), ha="center", fontsize=8, color="grey", style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Containers for participant-level ERPs
    participant_erps_occ   = {k: [] for k in CONDITION_MAP}
    participant_erps_motor = {k: [] for k in CONDITION_MAP}
    times_ref = None

    # Loop dynamically over the files discovered via glob
    for epoch_file in EPOCH_FILES:
        try:
            print(f"Processing: {os.path.basename(epoch_file)}")
            epochs = load_cleaned_epochs(epoch_file)
            
            if times_ref is None:
                times_ref = epochs.times
            
            for cond_key in CONDITION_MAP:
                cond_epochs = select_condition(epochs, cond_key, EVENT_ID)
                
                if len(cond_epochs) >= MIN_TRIALS:
                    _, occ_trials   = extract_channel_data(cond_epochs, OCCIPITAL_CHANNELS, BASELINE)
                    _, motor_trials = extract_channel_data(cond_epochs, MOTOR_CHANNEL, BASELINE)

                    # Average trials to create the single ERP waveform for this data file
                    participant_erps_occ[cond_key].append(occ_trials.mean(axis=0))
                    participant_erps_motor[cond_key].append(motor_trials.mean(axis=0))
        except Exception as e:
            print(f"Skipping file {epoch_file} due to error: {e}")
            continue

    if times_ref is None:
        print("No valid epoch data was loaded. Exiting.")
        return

    # Calculating Level-2 Grand Averages (Group) 
    grand_avg_occ   = {k: compute_grand_average(participant_erps_occ[k]) for k in CONDITION_MAP}
    grand_avg_motor = {k: compute_grand_average(participant_erps_motor[k]) for k in CONDITION_MAP}

    # Saving the Visuals
    fig_motor = plot_grand_avg_figure(times_ref, grand_avg_motor, f"Motor ({MOTOR_CHANNEL})", comparison="solo_vs_trio")
    fig_motor.savefig(os.path.join(OUTPUT_DIR, "grand_avg_motor_solo_vs_trio_allData.png"))
    plt.close(fig_motor)

    fig_occ = plot_grand_avg_figure(times_ref, grand_avg_occ, "Occipital (O1, O2, Oz avg)", comparison="solo_vs_trio")
    fig_occ.savefig(os.path.join(OUTPUT_DIR, "grand_avg_occipital_solo_vs_trio_allData.png"))
    plt.close(fig_occ)

    fig_comb = plot_grand_avg_combined(times_ref, grand_avg_occ, grand_avg_motor)
    fig_comb.savefig(os.path.join(OUTPUT_DIR, "grand_avg_combined_allData.png"))
    plt.close(fig_comb)
    print("All ERP processing completed successfully.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("ERP ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Baseline     : {BASELINE}")
    print(f"Error bands  : {ERROR_TYPE}  (grand average = across participants)")
    print(f"Smoothing    : {SMOOTH_SIGMA} samples")
    print(f"Min trials   : {MIN_TRIALS} per condition to enter grand average")
    print("=" * 60)
    run_pipeline()