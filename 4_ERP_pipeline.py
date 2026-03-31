"""
ERP Analysis Pipeline — EEG Feedback × Group Conditions
========================================================
Generates publication-ready ERP plots per participant, comparing:
  - Feedback: with_feedback vs without_feedback
  - Group:    solo vs trio

Channels of interest:
  - Occipital (average of O1, O2, Oz)
  - Motor (C3, analyzed separately)

Usage
-----
  python erp_analysis.py

Configure the paths and participant list in the CONFIGURATION section.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")               # non-interactive backend; change to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import mne
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"
OUTPUT_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\figures\erp"

# Define all participants to process. Each entry is (participant_id, session).
# Add as many as needed.
PARTICIPANTS = [
    ("301", 1), ("301", 2), ("301", 3),
    ("302", 1), ("302", 2), ("302", 3),
    ("303", 1), ("303", 2), ("303", 3),
    ("304", 1), ("304", 2), ("304", 3),
]

CONDITION_MAP = {
    "with_feedback/solo":    [1],   # Condition_1 → T1P  (30 epochs ✓)
    "with_feedback/trio":    [3],   # Condition_3 → T3P  (30 epochs ✓)
    "without_feedback/solo": [2],   # Condition_2 → T1Pn (29 epochs ✓)
    "without_feedback/trio": [4],   # Condition_4 → T3Pn (29 epochs ✓)
}

# Leave this alias as-is — it is passed into process_participant()
EVENT_ID = CONDITION_MAP

# Channels of interest
OCCIPITAL_CHANNELS = ["O1", "O2", "Oz"]
MOTOR_CHANNEL      = "C3"

# Epoch time window (should match your data)
TMIN, TMAX = -0.5, 5.5

# Baseline correction window (seconds). Set to None to skip.
BASELINE = (-0.5, 0.0)

# Smoothing sigma (in samples). Set to 0 to disable.
SMOOTH_SIGMA = 5

# Error band: "se" (standard error) or "ci95" (95% confidence interval)
ERROR_TYPE = "se"

# Visual style
PALETTE = {
    "solo": "#2271B5",   # blue
    "trio": "#E65F2B",   # orange
}
ALPHA_FILL = 0.18

# ============================================================
# DATA UTILITIES
# ============================================================

def load_cleaned_epochs(data_dir: str, participant_id: str, session: int) -> mne.Epochs:
    """Load ICA-cleaned epochs for one participant/session."""
    fname = os.path.join(data_dir, f"{participant_id}_p{session}_ica_cleaned-epo.fif")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Epoch file not found: {fname}")
    epochs = mne.read_epochs(fname, preload=True, verbose=False)
    # Apply standard montage for channel positions
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="ignore")
    print(f"  Loaded {len(epochs)} epochs — {participant_id} session {session}")
    return epochs


def select_condition(epochs: mne.Epochs, condition_key: str, event_id: dict) -> mne.Epochs:
    """
    Return the subset of epochs matching a condition key.

    Parameters
    ----------
    epochs       : mne.Epochs — full epoch object
    condition_key: str        — e.g. "with_feedback/solo"
    event_id     : dict       — maps condition keys to a list of Condition_N
                               integers, e.g. {"with_feedback/solo": [1, 2]}

    Strategy
    --------
    The saved epoch file uses names like "Condition_1", "Condition_2", etc.
    We look up the list of integers for this key, build the corresponding
    "Condition_N" string keys, and use MNE's string selector to pool epochs.
    """
    condition_ids = event_id.get(condition_key)
    if condition_ids is None:
        raise KeyError(f"Condition '{condition_key}' not found in CONDITION_MAP.")

    # Build MNE-compatible string keys: [1, 2] -> ["Condition_1", "Condition_2"]
    mne_keys = [f"Condition_{i}" for i in condition_ids]

    # Keep only keys that exist in this file (some may be absent after bad-epoch removal)
    available_keys = [k for k in mne_keys if k in epochs.event_id]

    if not available_keys:
        # Return an empty slice; the caller's len() == 0 guard will catch this
        return epochs[epochs.events[:, 2] == -1]

    return epochs[available_keys]

def get_occipital_erp(epochs: mne.Epochs, channels: list[str], baseline: tuple | None) -> tuple:
    """
    Average occipital channels and return (times, erp_matrix).

    Parameters
    ----------
    epochs   : mne.Epochs
    channels : list of channel names to average
    baseline : (tmin, tmax) for baseline correction, or None

    Returns
    -------
    times : ndarray (n_times,)
    data  : ndarray (n_trials, n_times)  — averaged across channels
    """
    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available:
        raise ValueError(f"None of {channels} found in epochs. Available: {epochs.ch_names}")

    picks = mne.pick_channels(epochs.ch_names, include=available)
    data  = epochs.get_data(picks=picks)          # (n_trials, n_channels, n_times)
    data  = data.mean(axis=1)                     # average across channels → (n_trials, n_times)

    if baseline is not None:
        tmin_bl, tmax_bl = baseline
        bl_mask = (epochs.times >= tmin_bl) & (epochs.times <= tmax_bl)
        data = data - data[:, bl_mask].mean(axis=1, keepdims=True)

    return epochs.times, data


def get_single_channel_erp(epochs: mne.Epochs, channel: str, baseline: tuple | None) -> tuple:
    """
    Extract a single channel's data, returning (times, erp_matrix).

    Returns
    -------
    times : ndarray (n_times,)
    data  : ndarray (n_trials, n_times)
    """
    if channel not in epochs.ch_names:
        raise ValueError(f"Channel '{channel}' not found. Available: {epochs.ch_names}")

    picks = mne.pick_channels(epochs.ch_names, include=[channel])
    data  = epochs.get_data(picks=picks).squeeze(axis=1)  # (n_trials, n_times)

    if baseline is not None:
        tmin_bl, tmax_bl = baseline
        bl_mask = (epochs.times >= tmin_bl) & (epochs.times <= tmax_bl)
        data = data - data[:, bl_mask].mean(axis=1, keepdims=True)

    return epochs.times, data


def compute_erp_stats(data: np.ndarray, error_type: str = "se", smooth_sigma: float = 0) -> dict:
    """
    Compute mean ± error band from trial matrix.

    Parameters
    ----------
    data         : (n_trials, n_times)
    error_type   : "se" → standard error; "ci95" → 95% CI (1.96 × SE)
    smooth_sigma : Gaussian smoothing sigma in samples (0 = off)

    Returns
    -------
    dict with keys: mean, lower, upper, n_trials
    """
    n = data.shape[0]
    # Use nanmean/nanstd so NaN-filled placeholders (missing conditions) don't crash
    mean = np.nanmean(data, axis=0)
    se   = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
    ci   = 1.96 * se if error_type == "ci95" else se

    if smooth_sigma > 0:
        mean  = gaussian_filter1d(mean,       smooth_sigma)
        lower = gaussian_filter1d(mean - ci,  smooth_sigma)
        upper = gaussian_filter1d(mean + ci,  smooth_sigma)
    else:
        lower = mean - ci
        upper = mean + ci

    return {"mean": mean, "lower": lower, "upper": upper, "n_trials": n}


# ============================================================
# PLOTTING UTILITIES
# ============================================================

# Shared publication style
STYLE = {
    "font.family":        "serif",
    "font.serif":         ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}


def _draw_erp_axes(ax, times, stats_solo, stats_trio, title, palette, alpha_fill,
                   show_xlabel=True, show_legend=True):
    """
    Draw two ERP lines (solo + trio) with error bands onto an existing Axes.

    Parameters are pre-computed stats dicts from compute_erp_stats().
    """
    for label, stats in [("Solo", stats_solo), ("Trio", stats_trio)]:
        color = palette[label.lower()]
        ax.plot(times, stats["mean"] * 1e6,
                color=color, linewidth=1.6, label=f"{label}  (n={stats['n_trials']})")
        ax.fill_between(times,
                        stats["lower"] * 1e6,
                        stats["upper"] * 1e6,
                        color=color, alpha=alpha_fill)

    # Stimulus onset line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey",  linewidth=0.5, linestyle="-",  alpha=0.3)

    # Invert y-axis (EEG convention: negative up)
    ax.invert_yaxis()

    ax.set_title(title, fontweight="bold", pad=8)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")

    if show_legend:
        ax.legend(loc="upper right")


def plot_erp_figure(times, condition_data: dict, channel_label: str,
                    participant_id: str, palette: dict, alpha_fill: float,
                    error_type: str, smooth_sigma: float) -> plt.Figure:
    """
    Build a publication-ready figure with two panels:
      Left  — WITH feedback (solo vs trio)
      Right — WITHOUT feedback (solo vs trio)

    Parameters
    ----------
    condition_data : dict with keys
        "with_feedback/solo", "with_feedback/trio",
        "without_feedback/solo", "without_feedback/trio"
        each mapping to a (n_trials, n_times) array

    Returns
    -------
    matplotlib Figure
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        fig.suptitle(
            f"ERP — {channel_label}  ·  Participant {participant_id}",
            fontsize=14, fontweight="bold", y=1.02
        )

        for ax, feedback_key, panel_title in [
            (axes[0], "with_feedback",    "With Feedback"),
            (axes[1], "without_feedback", "Without Feedback"),
        ]:
            stats_solo = compute_erp_stats(
                condition_data[f"{feedback_key}/solo"], error_type, smooth_sigma)
            stats_trio = compute_erp_stats(
                condition_data[f"{feedback_key}/trio"], error_type, smooth_sigma)

            _draw_erp_axes(
                ax, times,
                stats_solo, stats_trio,
                title=panel_title,
                palette=palette,
                alpha_fill=alpha_fill,
                show_xlabel=True,
                show_legend=(ax is axes[0]),
            )

        # Shared legend on right panel as well
        axes[1].legend(loc="upper right")

        # Error-band annotation
        band_label = "± SE" if error_type == "se" else "± 95% CI"
        fig.text(0.5, -0.04,
                 f"Shaded region: {band_label}   ·   Baseline: {BASELINE[0]}–{BASELINE[1]} s   ·   Smoothing σ={smooth_sigma} samples",
                 ha="center", fontsize=8, color="grey", style="italic")

        fig.tight_layout()
    return fig


def plot_combined_figure(times, occ_data: dict, motor_data: dict,
                         participant_id: str, palette: dict,
                         alpha_fill: float, error_type: str,
                         smooth_sigma: float) -> plt.Figure:
    """
    Optional combined figure: 2 rows × 2 columns.
      Row 1 — Occipital (O1/O2/Oz average)
      Row 2 — Motor (C3)
    Columns: with_feedback | without_feedback
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
        fig.suptitle(
            f"ERP Overview — Participant {participant_id}",
            fontsize=14, fontweight="bold", y=1.02
        )

        for row, (data_dict, row_label) in enumerate(
            [(occ_data, "Occipital (O1/O2/Oz avg)"),
             (motor_data, f"Motor ({MOTOR_CHANNEL})")]):

            for col, (feedback_key, panel_title) in enumerate(
                [("with_feedback",    "With Feedback"),
                 ("without_feedback", "Without Feedback")]):

                ax = axes[row][col]
                stats_solo = compute_erp_stats(
                    data_dict[f"{feedback_key}/solo"], error_type, smooth_sigma)
                stats_trio = compute_erp_stats(
                    data_dict[f"{feedback_key}/trio"], error_type, smooth_sigma)

                _draw_erp_axes(
                    ax, times,
                    stats_solo, stats_trio,
                    title=panel_title if row == 0 else "",
                    palette=palette,
                    alpha_fill=alpha_fill,
                    show_xlabel=(row == 1),
                    show_legend=(col == 0 and row == 0),
                )

                if col == 0:
                    ax.set_ylabel(f"{row_label}\nAmplitude (µV)")

        fig.tight_layout()
    return fig


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_participant(participant_id: str, session: int,
                        data_dir: str, output_dir: str,
                        event_id: dict, occ_channels: list,
                        motor_channel: str, baseline: tuple | None,
                        error_type: str, smooth_sigma: float,
                        palette: dict, alpha_fill: float):
    """
    Full ERP pipeline for one participant:
      1) Load ICA-cleaned epochs
      2) Select condition subsets
      3) Extract channel data + baseline correction
      4) Generate and save ERP figures
    """
    print(f"\n{'='*60}")
    print(f"  Participant {participant_id}  ·  Session {session}")
    print(f"{'='*60}")

    # --- Load ---
    epochs = load_cleaned_epochs(data_dir, participant_id, session)

    # Always print actual event IDs found in the file so the user can
    # verify / update the EVENT_ID dict at the top of the script.
    print(f"\n  Event IDs found in this file:")
    for name, eid in sorted(epochs.event_id.items(), key=lambda x: x[1]):
        count = np.sum(epochs.events[:, 2] == eid)
        print(f"    {eid:>6}  →  '{name}'  ({count} epochs)")

    # times is derived from the full epoch object — always available,
    # regardless of whether individual conditions have trials.
    times = epochs.times

    # --- Condition data containers ---
    occ_data   = {}   # keyed by "feedback_type/group"
    motor_data = {}

    for condition_key in event_id.keys():
        cond_epochs = select_condition(epochs, condition_key, event_id)

        if len(cond_epochs) == 0:
            print(f"  ⚠  No epochs for '{condition_key}' — "
                  f"check EVENT_ID dict matches the event IDs printed above")
            # Fill with NaN so flat zero lines don't mislead in plots
            n_times = len(times)
            occ_data[condition_key]   = np.full((1, n_times), np.nan)
            motor_data[condition_key] = np.full((1, n_times), np.nan)
            continue

        _, occ   = get_occipital_erp(cond_epochs, occ_channels, baseline)
        _, motor = get_single_channel_erp(cond_epochs, motor_channel, baseline)

        occ_data[condition_key]   = occ
        motor_data[condition_key] = motor
        print(f"  {condition_key}: {len(cond_epochs)} trials")

    # --- Output directory ---
    os.makedirs(output_dir, exist_ok=True)
    subj_dir = os.path.join(output_dir, f"{participant_id}_p{session}")
    os.makedirs(subj_dir, exist_ok=True)

    # --- Plot 1: Occipital ERP per feedback condition ---
    fig_occ = plot_erp_figure(
        times, occ_data,
        channel_label=f"Occipital ({', '.join(occ_channels)} avg)",
        participant_id=participant_id,
        palette=palette, alpha_fill=alpha_fill,
        error_type=error_type, smooth_sigma=smooth_sigma,
    )
    occ_path = os.path.join(subj_dir, f"{participant_id}_p{session}_erp_occipital.png")
    fig_occ.savefig(occ_path)
    plt.close(fig_occ)
    print(f"  ✓ Saved: {occ_path}")

    # --- Plot 2: Motor ERP per feedback condition ---
    fig_motor = plot_erp_figure(
        times, motor_data,
        channel_label=f"Motor ({motor_channel})",
        participant_id=participant_id,
        palette=palette, alpha_fill=alpha_fill,
        error_type=error_type, smooth_sigma=smooth_sigma,
    )
    motor_path = os.path.join(subj_dir, f"{participant_id}_p{session}_erp_motor.png")
    fig_motor.savefig(motor_path)
    plt.close(fig_motor)
    print(f"  ✓ Saved: {motor_path}")

    # --- Plot 3: Combined 2×2 overview ---
    fig_combined = plot_combined_figure(
        times, occ_data, motor_data,
        participant_id=participant_id,
        palette=palette, alpha_fill=alpha_fill,
        error_type=error_type, smooth_sigma=smooth_sigma,
    )
    combined_path = os.path.join(subj_dir, f"{participant_id}_p{session}_erp_combined.png")
    fig_combined.savefig(combined_path)
    plt.close(fig_combined)
    print(f"  ✓ Saved: {combined_path}")

    return times, occ_data, motor_data


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("ERP ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Participants:  {len(PARTICIPANTS)}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Baseline:      {BASELINE}")
    print(f"Error bands:   {ERROR_TYPE}")
    print(f"Smoothing σ:   {SMOOTH_SIGMA} samples")
    print("=" * 60)

    for pid, session in PARTICIPANTS:
        try:
            process_participant(
                participant_id  = pid,
                session         = session,
                data_dir        = DATA_DIR,
                output_dir      = OUTPUT_DIR,
                event_id        = EVENT_ID,
                occ_channels    = OCCIPITAL_CHANNELS,
                motor_channel   = MOTOR_CHANNEL,
                baseline        = BASELINE,
                error_type      = ERROR_TYPE,
                smooth_sigma    = SMOOTH_SIGMA,
                palette         = PALETTE,
                alpha_fill      = ALPHA_FILL,
            )
        except FileNotFoundError as e:
            print(f"\n  ✗  Skipped {pid} p{session}: {e}")
        except Exception as e:
            print(f"\n  ✗  Error for {pid} p{session}: {e}")
            raise

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)