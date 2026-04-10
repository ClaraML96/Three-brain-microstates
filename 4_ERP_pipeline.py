"""
ERP Analysis Pipeline — EEG Feedback × Group Conditions
========================================================
Two-level averaging structure:
  Level 1 — per participant: average across trials → one ERP waveform per
             participant per condition
  Level 2 — grand average:  average those waveforms across participants →
             one group-level ERP per condition, with SE / 95% CI across
             participants (not trials)

Conditions compared:
  - with_feedback/solo    (T1P)
  - with_feedback/trio    (T3P)
  - without_feedback/solo (T1Pn)
  - without_feedback/trio (T3Pn)

Channels of interest:
  - Occipital: O1, O2, Oz  (averaged into one signal)
  - Motor:     C3           (single channel)

Output per run
--------------
  figures/ grand_average/
      grand_avg_occipital_solo_vs_trio.png
      grand_avg_motor_solo_vs_trio.png
      grand_avg_occipital_feedback.png
      grand_avg_motor_feedback.png
      grand_avg_combined.png       <- 2x2: rows = channel, cols = feedback
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR   = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"
OUTPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\grand_average"

PARTICIPANTS = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

# Condition_id -> experimental cell (decoded from Force_df)
CONDITION_MAP = {
    "with_feedback/solo":    [1],   # T1P
    "with_feedback/trio":    [3],   # T3P
    "without_feedback/solo": [2],   # T1Pn
    "without_feedback/trio": [4],   # T3Pn
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
# in the grand average for that condition.
MIN_TRIALS = 10

# Set to False to skip per-participant plots and only produce grand average.
SAVE_PER_PARTICIPANT = False

PALETTE = {
    "solo":             "#2271B5",
    "trio":             "#E65F2B",
    "with_feedback":    "#1B7837",
    "without_feedback": "#762A83",
}
ALPHA_FILL    = 0.35   # opacity of the CI band (higher = more visible)
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

#  Cleaning the data

# Function returns an mne.Epochs that is cleaned (through ICA processing), ready for averaging steps.
def load_cleaned_epochs(data_dir, participant_id, session):
    """Load ICA-cleaned epochs and attach standard 10-20 montage."""
    fname = os.path.join(data_dir, f"{participant_id}_p{session}_ica_cleaned-epo.fif")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Epoch file not found: {fname}")
    epochs = mne.read_epochs(fname, preload=True, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="ignore")
    return epochs

# Function acts as a translator and a filter. 
def select_condition(epochs, condition_key, event_id):
    """Return epochs matching a condition key (pools multiple Condition_N ids)."""
    condition_ids  = event_id.get(condition_key)
    if condition_ids is None:
        raise KeyError(f"'{condition_key}' not in CONDITION_MAP.")
    mne_keys       = [f"Condition_{i}" for i in condition_ids]
    available_keys = [k for k in mne_keys if k in epochs.event_id]
    if not available_keys:
        return epochs[epochs.events[:, 2] == -1]   # guaranteed empty
    return epochs[available_keys]

# Function will "Data Slicer and Cleaner." 
def extract_channel_data(epochs, channels, baseline):
    """
    Extract and baseline-correct channel data from epochs.

    Parameters
    ----------
    epochs   : mne.Epochs
    channels : str or list[str]
    baseline : (tmin, tmax) or None

    Returns
    -------
    times : (n_times,)
    data  : (n_trials, n_times) -- averaged across channels if list given
    """
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

# Calculating standard error and confidence intervals across participants for the grand average plots.
def compute_grand_average(participant_erps):
    """
    Compute grand average and cross-participant variability.

    Parameters
    ----------
    participant_erps : list of (n_times,) arrays
        One trial-averaged ERP waveform per participant.

    Returns
    -------
    dict with keys: mean, lower, upper, n_participants
    """
    stack = np.array(participant_erps)          # (n_participants, n_times)
    n     = stack.shape[0]
    mean  = stack.mean(axis=0)
    se    = stack.std(axis=0, ddof=1) / np.sqrt(n)
    ci    = 1.96 * se if ERROR_TYPE == "ci95" else se

    if SMOOTH_SIGMA > 0:
        lower = mean - ci
        upper = mean + ci
        mean  = gaussian_filter1d(mean,  SMOOTH_SIGMA)
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
    band = ("SE (across participants)"
            if ERROR_TYPE == "se" else "95% CI (across participants)")
    return (f"Shaded region: +/- {band}   |   "
            f"Baseline: {BASELINE[0]} to {BASELINE[1]} s   |   "
            f"Smoothing sigma={SMOOTH_SIGMA} samples")

# Renders the ERP grand average plot
def _draw_grand_avg_axes(ax, times, stats_a, stats_b,
                         label_a, label_b, color_a, color_b,
                         title, show_xlabel=True, show_legend=True):
    """
    Draw two grand-average ERP lines with cross-participant error bands.
    stats_a / stats_b come from compute_grand_average().
    """
    for stats, label, color in [
        (stats_a, label_a, color_a),
        (stats_b, label_b, color_b),
    ]:
        n    = stats["n_participants"]
        mean = stats["mean"] * 1e6
        lo   = stats["lower"] * 1e6
        hi   = stats["upper"] * 1e6

        # Mean line
        ax.plot(times, mean, color=color, linewidth=2.0,
                label=f"{label}  (n={n})", zorder=3)

        # Filled CI band
        ax.fill_between(times, lo, hi,
                        color=color, alpha=ALPHA_FILL, zorder=2)

        # Dashed boundary lines for extra visibility
        ax.plot(times, lo, color=color, linewidth=CI_LINEWIDTH,
                linestyle="--", alpha=CI_EDGE_ALPHA, zorder=2)
        ax.plot(times, hi, color=color, linewidth=CI_LINEWIDTH,
                linestyle="--", alpha=CI_EDGE_ALPHA, zorder=2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey",  linewidth=0.5, linestyle="-",  alpha=0.3)
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", pad=8)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    if show_legend:
        ax.legend(loc="upper right")

# Aranging the panels
def plot_grand_avg_figure(times, grand_avg, channel_label, comparison="solo_vs_trio"):
    """
    Two-panel grand average figure.

    comparison="solo_vs_trio"
        Left panel  = with feedback,    solo vs trio
        Right panel = without feedback, solo vs trio

    comparison="feedback"
        Left panel  = solo,  with vs without feedback
        Right panel = trio,  with vs without feedback
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

        if comparison == "solo_vs_trio":
            fig.suptitle(f"Grand Average ERP -- {channel_label}  |  Solo vs Trio",
                         fontsize=14, fontweight="bold", y=0.98)
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

        elif comparison == "feedback":
            fig.suptitle(
                f"Grand Average ERP -- {channel_label}  |  With vs Without Feedback",
                fontsize=14, fontweight="bold", y=0.98)
            panels = [
                (axes[0], "solo", "Solo"),
                (axes[1], "trio", "Trio"),
            ]
            for ax, grp_key, panel_title in panels:
                _draw_grand_avg_axes(
                    ax, times,
                    grand_avg[f"with_feedback/{grp_key}"],
                    grand_avg[f"without_feedback/{grp_key}"],
                    label_a="With Feedback", label_b="Without Feedback",
                    color_a=PALETTE["with_feedback"],
                    color_b=PALETTE["without_feedback"],
                    title=panel_title,
                    show_xlabel=True,
                    show_legend=(ax is axes[0]),
                )
            axes[1].legend(loc="upper right")

        fig.text(0.5, -0.04, _annotation(),
                 ha="center", fontsize=8, color="grey", style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ERP overview
def plot_grand_avg_combined(times, grand_avg_occ, grand_avg_motor):
    """
    2x2 combined grand average figure.
      Rows    : occipital | motor
      Columns : with_feedback | without_feedback
    Solo vs trio shown in each panel.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
        fig.suptitle("Grand Average ERP Overview  |  Solo vs Trio",
                     fontsize=14, fontweight="bold", y=0.98)

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

        fig.text(0.5, -0.02, _annotation(),
                 ha="center", fontsize=8, color="grey", style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ============================================================
# PER-PARTICIPANT PLOT
# ============================================================

# def _draw_trial_axes(ax, times, stats_solo, stats_trio, title,
#                      show_xlabel=True, show_legend=True):
#     """Draw per-participant ERP axes (SE across trials)."""
#     for label, stats, color in [
#         ("Solo", stats_solo, PALETTE["solo"]),
#         ("Trio", stats_trio, PALETTE["trio"]),
#     ]:
#         ax.plot(times, stats["mean"] * 1e6,
#                 color=color, linewidth=1.6,
#                 label=f"{label}  (n={stats['n_trials']} trials)")
#         ax.fill_between(times,
#                         stats["lower"] * 1e6,
#                         stats["upper"] * 1e6,
#                         color=color, alpha=ALPHA_FILL)
#     ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
#     ax.axhline(0, color="grey",  linewidth=0.5, linestyle="-",  alpha=0.3)
#     ax.invert_yaxis()
#     ax.set_title(title, fontweight="bold", pad=8)
#     if show_xlabel:
#         ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude (uV)")
#     if show_legend:
#         ax.legend(loc="upper right")


# def compute_trial_stats(data):
#     """Mean +/- SE across trials (for per-participant plots)."""
#     n    = data.shape[0]
#     mean = np.nanmean(data, axis=0)
#     se   = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
#     ci   = 1.96 * se if ERROR_TYPE == "ci95" else se
#     if SMOOTH_SIGMA > 0:
#         mean  = gaussian_filter1d(mean,       SMOOTH_SIGMA)
#         lower = gaussian_filter1d(mean - ci,  SMOOTH_SIGMA)
#         upper = gaussian_filter1d(mean + ci,  SMOOTH_SIGMA)
#     else:
#         lower, upper = mean - ci, mean + ci
#     return {"mean": mean, "lower": lower, "upper": upper, "n_trials": n}


# def plot_participant_combined(times, occ_data, motor_data, participant_id, session):
#     """2x2 per-participant overview (SE across trials, not participants)."""
#     with plt.rc_context(STYLE):
#         fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
#         fig.suptitle(
#             f"ERP -- Participant {participant_id}  Session {session}  "
#             f"[variability = SE across trials]",
#             fontsize=13, fontweight="bold", y=0.98)

#         row_items = [
#             (occ_data,   f"Occipital ({', '.join(OCCIPITAL_CHANNELS)} avg)"),
#             (motor_data, f"Motor ({MOTOR_CHANNEL})"),
#         ]
#         for row, (data_dict, row_label) in enumerate(row_items):
#             for col, (fb_key, panel_title) in enumerate(
#                 [("with_feedback",    "With Feedback"),
#                  ("without_feedback", "Without Feedback")]
#             ):
#                 ax = axes[row][col]
#                 s_solo = compute_trial_stats(data_dict[f"{fb_key}/solo"])
#                 s_trio = compute_trial_stats(data_dict[f"{fb_key}/trio"])
#                 _draw_trial_axes(
#                     ax, times, s_solo, s_trio,
#                     title=panel_title if row == 0 else "",
#                     show_xlabel=(row == 1),
#                     show_legend=(row == 0 and col == 0),
#                 )
#                 if col == 0:
#                     ax.set_ylabel(f"{row_label}\nAmplitude (uV)")

#         fig.text(0.5, -0.02,
#                  f"Shaded region: +/- SE (across trials)   |   "
#                  f"Baseline: {BASELINE[0]} to {BASELINE[1]} s   |   "
#                  f"Smoothing sigma={SMOOTH_SIGMA} samples",
#                  ha="center", fontsize=8, color="grey", style="italic")
#         fig.tight_layout(rect=[0, 0, 1, 0.93])
#     return fig


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Containers for participant-level ERPs
    participant_erps_occ   = {k: [] for k in CONDITION_MAP}
    participant_erps_motor = {k: [] for k in CONDITION_MAP}
    times_ref = None

    # Averaging across trials
    for pid, session in PARTICIPANTS:
        try:
            epochs = load_cleaned_epochs(DATA_DIR, pid, session)
            if times_ref is None:
                times_ref = epochs.times
            
            for cond_key in CONDITION_MAP:
                cond_epochs = select_condition(epochs, cond_key, EVENT_ID)
                
                if len(cond_epochs) >= MIN_TRIALS:
                    # Extract channel data (Averages O1, O2, Oz | Picks C3)
                    _, occ_trials   = extract_channel_data(cond_epochs, OCCIPITAL_CHANNELS, BASELINE)
                    _, motor_trials = extract_channel_data(cond_epochs, MOTOR_CHANNEL, BASELINE)

                    # Average trials to create the single ERP waveform for this person
                    participant_erps_occ[cond_key].append(occ_trials.mean(axis=0))  # Averaging across participant
                    participant_erps_motor[cond_key].append(motor_trials.mean(axis=0))
        except FileNotFoundError:
            continue

    # Calculating Level-2 Grand Averages (Group) 
    grand_avg_occ   = {k: compute_grand_average(participant_erps_occ[k]) for k in CONDITION_MAP}
    grand_avg_motor = {k: compute_grand_average(participant_erps_motor[k]) for k in CONDITION_MAP}

    # Saving the Visuals
    # 1. THE MOTOR (C3) ERP VISUAL (Solo vs Trio)
    fig_motor = plot_grand_avg_figure(times_ref, grand_avg_motor, f"Motor ({MOTOR_CHANNEL})", comparison="solo_vs_trio")
    fig_motor.savefig(os.path.join(OUTPUT_DIR, "grand_avg_motor_solo_vs_trio.png"))
    plt.close(fig_motor)

    # 2. THE OCCIPITAL ERP VISUAL (Solo vs Trio)
    fig_occ = plot_grand_avg_figure(times_ref, grand_avg_occ, "Occipital (O1, O2, Oz avg)", comparison="solo_vs_trio")
    fig_occ.savefig(os.path.join(OUTPUT_DIR, "grand_avg_occipital_solo_vs_trio.png"))
    plt.close(fig_occ)

    # 3. THE COMBINED 2x2 OVERVIEW
    fig_comb = plot_grand_avg_combined(times_ref, grand_avg_occ, grand_avg_motor)
    fig_comb.savefig(os.path.join(OUTPUT_DIR, "grand_avg_combined.png"))
    plt.close(fig_comb)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("ERP ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Participants : {len(PARTICIPANTS)}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Baseline     : {BASELINE}")
    print(f"Error bands  : {ERROR_TYPE}  (grand average = across participants)")
    print(f"Smoothing    : {SMOOTH_SIGMA} samples")
    print(f"Min trials   : {MIN_TRIALS} per condition to enter grand average")
    print("=" * 60)
    run_pipeline()