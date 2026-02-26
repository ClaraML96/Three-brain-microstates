import numpy as np
import os
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def detect_bad_channels(raw, threshold_mad=5.0):
    """
    Detect bad EEG channels using robust MAD-based z-scores.
    
    Identifies channels with abnormally high variance, which typically indicate
    poor electrode contact or drift. Uses median absolute deviation (MAD) for
    robustness to outliers.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    threshold_mad : float
        MAD z-score threshold for bad channel detection (default: 5.0)
    
    Returns
    -------
    raw : mne.io.Raw
        Raw data with bad channels marked in raw.info['bads']
    """
    eeg = raw.copy().pick('eeg')
    data = eeg.get_data()
    
    # Variance-based detection
    channel_vars = np.var(data, axis=1)
    median_var = np.median(channel_vars)
    mad = np.median(np.abs(channel_vars - median_var))
    
    # Robust z-score (handle MAD == 0)
    z_scores = np.zeros_like(channel_vars) if mad == 0 else (channel_vars - median_var) / (mad * 1.4826)
    
    # Find bad channels
    bad_channels = [ch for ch, z in zip(eeg.ch_names, z_scores) if z > threshold_mad]
    raw.info['bads'] = bad_channels
    
    # Diagnostic output
    print(f"Bad Channels (MAD-based, z > {threshold_mad}):")
    print(f"  Checked: {len(eeg.ch_names)} channels")
    print(f"  Bad: {len(bad_channels)} channels")
    if bad_channels:
        print(f"  Channels: {', '.join(bad_channels)}")
    
    return raw


def detect_bad_epochs_ptp(epochs, threshold_uv=300):
    """
    Detect and reject bad epochs based on peak-to-peak amplitude.
    
    Identifies epochs with extreme amplitude values in any channel, which
    typically indicate muscle artifacts, electrode artifacts, or signal loss.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    threshold_uv : float
        Peak-to-peak amplitude threshold in microvolts (default: 300)
    
    Returns
    -------
    epochs : mne.Epochs
        Epochs with bad epochs removed
    """
    initial_count = len(epochs)
    threshold_v = threshold_uv * 1e-6
    
    # Automatic amplitude rejection
    epochs.drop_bad(reject=dict(eeg=threshold_v), verbose=False)
    
    final_count = len(epochs)
    dropped = initial_count - final_count
    
    # Diagnostic output
    print(f"Bad Epochs (peak-to-peak > {threshold_uv} µV):")
    print(f"  Initial: {initial_count} epochs")
    print(f"  Dropped: {dropped} epochs")
    print(f"  Remaining: {final_count} epochs")
    
    return epochs


def manual_epoch_inspection(epochs):
    """
    Visually inspect and manually mark bad epochs for rejection.
    
    Opens an interactive plot allowing manual rejection of epochs with
    artifacts not caught by automatic methods (e.g., movement artifacts,
    physiological noise).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    
    Returns
    -------
    epochs : mne.Epochs
        Epochs with manually marked bad epochs removed
    """
    print("\n" + "="*70)
    print("MANUAL EPOCH INSPECTION")
    print("="*70)
    print("Click on epochs to mark as bad | Arrow keys to navigate | Close to finish")
    print("="*70)
    
    initial_count = len(epochs)
    
    # Interactive inspection (32 channels, 5 epochs, 50 µV scaling)
    epochs.plot(
        n_channels=32,
        n_epochs=5,
        scalings=dict(eeg=50e-6),
        block=True
    )
    
    # Remove marked bad epochs
    epochs.drop_bad(verbose=False)
    
    final_count = len(epochs)
    rejected = initial_count - final_count
    
    print(f"\nManual Inspection:")
    print(f"  Initial: {initial_count} epochs")
    print(f"  Rejected: {rejected} epochs")
    print(f"  Remaining: {final_count} epochs")
    
    return epochs


def extract_real_trial_events(events, sfreq):
    """
    Extract and collapse real trial events from raw event array.
    
    This function performs robust filtering to isolate experimental trials:
    1. Excludes all events before ExpStart trigger (112)
    2. Excludes practice trials (triggers 1-11)
    3. Keeps only real experimental trials (triggers 10-59)
    4. Collapses force levels by extracting second digit (e.g., 23 → 3)
    5. Validates exactly 300 trials found (10 conditions × 30 force levels)
    6. Checks for duplicates and balance
    
    Parameters
    ----------
    events : np.ndarray
        MNE events array (n_events × 3)
    sfreq : float
        Sampling frequency in Hz
    
    Returns
    -------
    collapsed_events : np.ndarray
        Filtered and collapsed event array
    event_id : dict
        Mapping from condition names to event codes
    
    Raises
    ------
    RuntimeError
        If ExpStart trigger not found or wrong number of trials detected
    """
    print("\n" + "="*70)
    print("EVENT FILTERING AND VALIDATION")
    print("="*70)
    
    print(f"Total events found: {len(events)}")
    
    # -----------------------------------------------------------------------
    # Step 1: Find ExpStart trigger and filter events after experiment begins
    # -----------------------------------------------------------------------
    exp_start_samples = events[events[:, 2] == 112][:, 0]
    
    if len(exp_start_samples) == 0:
        raise RuntimeError(
            "ExpStart trigger (code 112) not found in event array. "
            "Cannot determine experiment start."
        )
    
    exp_start_sample = exp_start_samples[0]
    events_after_exp = events[events[:, 0] > exp_start_sample]
    
    print(f"✓ ExpStart found at sample {exp_start_sample}")
    print(f"  Events after ExpStart: {len(events_after_exp)}")
    
    # -----------------------------------------------------------------------
    # Step 2: Explicitly exclude practice trials (codes 1-11)
    # -----------------------------------------------------------------------
    practice_trials = events_after_exp[
        (events_after_exp[:, 2] >= 1) & 
        (events_after_exp[:, 2] <= 11)
    ]
    
    if len(practice_trials) > 0:
        print(f"  Practice trials detected: {len(practice_trials)} (codes 1-11)")
        print(f"  → Excluding practice trials")
    
    # -----------------------------------------------------------------------
    # Step 3: Keep only real experimental trials (codes 10-59)
    # -----------------------------------------------------------------------
    # Real trials: 10 conditions × 30 repetitions = 300 trials
    # Trigger codes: X0-X9 where X ∈ {1,2,3,4,5} represents force level
    # and second digit represents condition
    real_trials = events_after_exp[
        (events_after_exp[:, 2] >= 10) & 
        (events_after_exp[:, 2] <= 59)
    ]
    
    print(f"✓ Real trial triggers found: {len(real_trials)}")
    
    # -----------------------------------------------------------------------
    # Step 4: Validate exactly 300 trials (CRITICAL ASSERTION)
    # -----------------------------------------------------------------------
    expected_trials = 300
    if len(real_trials) != expected_trials:
        raise RuntimeError(
            f"Expected exactly {expected_trials} real trial triggers, "
            f"but found {len(real_trials)}. "
            f"Data integrity compromised - check raw data and trigger codes."
        )
    
    print(f"✓ Trial count validated: {len(real_trials)} trials (expected: {expected_trials})")
    
    # -----------------------------------------------------------------------
    # Step 5: Collapse force levels (extract second digit only)
    # -----------------------------------------------------------------------
    # Example: triggers 10,20,30,40,50 all become condition 0
    #          triggers 13,23,33,43,53 all become condition 3
    collapsed_events = real_trials.copy()
    
    for i, event in enumerate(collapsed_events):
        original_code = event[2]
        condition_code = original_code % 10  # Extract second digit
        collapsed_events[i, 2] = condition_code
    
    unique_conditions = np.unique(collapsed_events[:, 2])
    print(f"✓ Collapsed to {len(unique_conditions)} conditions: {sorted(unique_conditions)}")
    
    # -----------------------------------------------------------------------
    # Step 6: Check for duplicate sample indices (data integrity)
    # -----------------------------------------------------------------------
    sample_indices = collapsed_events[:, 0]
    unique_samples = np.unique(sample_indices)
    
    if len(unique_samples) != len(sample_indices):
        n_duplicates = len(sample_indices) - len(unique_samples)
        raise RuntimeError(
            f"Found {n_duplicates} duplicate event sample indices. "
            f"This indicates corrupted trigger data."
        )
    
    print(f"✓ No duplicate sample indices detected")
    
    # -----------------------------------------------------------------------
    # Step 7: Validate condition balance (10 epochs per condition)
    # -----------------------------------------------------------------------
    expected_per_condition = 30
    condition_counts = {}
    
    for cond in unique_conditions:
        count = np.sum(collapsed_events[:, 2] == cond)
        condition_counts[int(cond)] = count
    
    print(f"\nCondition balance check:")
    all_balanced = True
    for cond in sorted(unique_conditions):
        count = condition_counts[int(cond)]
        status = "✓" if count == expected_per_condition else "⚠"
        print(f"  {status} Condition {int(cond)}: {count} epochs (expected: {expected_per_condition})")
        if count != expected_per_condition:
            all_balanced = False
    
    if not all_balanced:
        print(f"\n⚠ WARNING: Condition imbalance detected!")
        print(f"  Not all conditions have exactly {expected_per_condition} epochs.")
        print(f"  This may affect statistical analysis.")
    else:
        print(f"\n✓ All conditions balanced ({expected_per_condition} epochs each)")
    
    # -----------------------------------------------------------------------
    # Step 8: Calculate minimum inter-event interval
    # -----------------------------------------------------------------------
    inter_event_samples = np.diff(collapsed_events[:, 0])
    min_iei_samples = np.min(inter_event_samples)
    min_iei_seconds = min_iei_samples / sfreq
    
    print(f"\nInter-event interval diagnostics:")
    print(f"  Minimum IEI: {min_iei_seconds:.3f} s ({min_iei_samples} samples)")
    print(f"  Mean IEI: {np.mean(inter_event_samples) / sfreq:.3f} s")
    print(f"  Epoch duration: {EPOCH_TMAX - EPOCH_TMIN:.1f} s")
    
    if min_iei_seconds < (EPOCH_TMAX - EPOCH_TMIN):
        print(f"  ⚠ WARNING: Minimum IEI < epoch duration")
        print(f"    Epoch overlap is mathematically expected")
    else:
        print(f"  ✓ No epoch overlap expected")
    
    # -----------------------------------------------------------------------
    # Step 9: Create event_id dictionary
    # -----------------------------------------------------------------------
    event_id = {f"Condition_{int(cond)}": int(cond) 
                for cond in unique_conditions}
    
    print(f"\n✓ Event filtering complete: {len(collapsed_events)} valid trials")
    print("="*70)
    
    return collapsed_events, event_id


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

# Configuration
DATA_PATH = "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4"
FILE_NAME = '301.bdf'
PARTICIPANT = 3

# Processing parameters
FILTER_LOW = 1.0  # Hz
FILTER_HIGH = 40.0  # Hz
RESAMPLE_FREQ = 512  # Hz
BAD_CHANNEL_THRESHOLD = 3.0  # MAD z-score
BAD_EPOCH_THRESHOLD = 300  # µV
EPOCH_TMIN = -0.5  # seconds
EPOCH_TMAX = 5.5  # seconds

# -------
# Step 1: Load and extract participant data
# -------
file_path = f"{DATA_PATH}\\{FILE_NAME}"
print(f"\n{'='*70}")
print(f"LOADING: {FILE_NAME} | Participant {PARTICIPANT}")
print(f"{'='*70}")

raw = mne.io.read_raw_bdf(file_path, preload=False)
participant_channels = [ch for ch in raw.ch_names if ch.startswith(f'{PARTICIPANT}-')]
stimulus_channels = [ch for ch in raw.ch_names if 'Status' in ch or 'STI' in ch]
raw_p = raw.copy().pick(participant_channels + stimulus_channels)

# -------
# Step 2: Prepare data (load, filter, resample)
# -------
print(f"\nData preparation:")
raw_p.load_data()
print(f"  Loaded: {len(raw_p.ch_names)} channels")

print(f"  Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz (Hamming)")
raw_p.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, fir_design='firwin', verbose=False)

print(f"  Resampling: {RESAMPLE_FREQ} Hz")
raw_p.resample(sfreq=RESAMPLE_FREQ, npad='auto')

# -------
# Step 3: Detect bad channels, rename, set montage, and interpolate
# -------
print(f"\n")
raw_p = detect_bad_channels(raw_p, threshold_mad=BAD_CHANNEL_THRESHOLD)

# Rename channels to remove participant prefix
channel_mapping = {ch: ch.replace(f'{PARTICIPANT}-', '') for ch in participant_channels}
raw_p.rename_channels(channel_mapping)

# Assign standard 10-20 montage for interpolation
if raw_p.info['bads']:
    print(f"  Setting standard 10-20 montage...")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_p.set_montage(montage)
    print(f"  Interpolating {len(raw_p.info['bads'])} bad channel(s)...")
    raw_p.interpolate_bads(reset_bads=True)

# -------
# Step 4: Find and filter trial events (robust extraction)
# -------
events = mne.find_events(raw_p, stim_channel='Status', shortest_event=1, verbose=False)

# Extract and validate real trial events using robust filtering
collapsed_events, event_id = extract_real_trial_events(events, raw_p.info['sfreq'])

# -------
# Step 5: Create epochs
# -------
print(f"\n{'='*70}")
print("EPOCH CREATION")
print(f"{'='*70}")

epochs = mne.Epochs(
    raw_p,
    collapsed_events,
    event_id=event_id,
    tmin=EPOCH_TMIN,
    tmax=EPOCH_TMAX,
    baseline=None,
    preload=True,
    verbose=False
)

print(f"✓ Epochs created: {len(epochs)}")
print(f"  Time window: {EPOCH_TMIN} to {EPOCH_TMAX} s")
print(f"  Baseline: None (will be applied later if needed)")
print(f"{'='*70}")

# -------
# Step 6: Manual inspection
# -------
epochs = manual_epoch_inspection(epochs)

# -------
# Step 7: Save cleaned epochs for ICA
# -------
print(f"\n{'='*70}")
print("SAVING CLEANED EPOCHS")
print(f"{'='*70}")

# Diagnostic: Current working directory
print(f"Current working directory: {os.getcwd()}")

# Define save directory (absolute path for Windows reliability)
SAVE_DIR = r"C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\data\\preprocessed"
print(f"Target save directory: {SAVE_DIR}")

# Create directory if it doesn't exist
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✓ Directory created/verified")
except Exception as e:
    print(f"⚠ Error creating directory: {e}")
    raise

# Verify directory exists
if os.path.isdir(SAVE_DIR):
    print(f"✓ Directory exists and is accessible")
else:
    raise FileNotFoundError(f"Failed to create directory: {SAVE_DIR}")

# Define output file path
EPOCHS_FILE = os.path.join(SAVE_DIR, "301_p3_clean-epo.fif")
print(f"Output file path: {EPOCHS_FILE}")

# Save epochs with error handling
try:
    print(f"Saving epochs...")
    epochs.save(EPOCHS_FILE, overwrite=True, verbose=True)
    print(f"✓ epochs.save() completed without errors")
except Exception as e:
    print(f"⚠ Error during save: {e}")
    raise

# Verify file was created
if os.path.exists(EPOCHS_FILE):
    file_size_mb = os.path.getsize(EPOCHS_FILE) / (1024 * 1024)
    print(f"✓ File created successfully")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Location: {EPOCHS_FILE}")
else:
    raise FileNotFoundError(f"File was not created: {EPOCHS_FILE}")

# List all files in the save directory
print(f"\nFiles in {SAVE_DIR}:")
try:
    files = os.listdir(SAVE_DIR)
    if files:
        for f in files:
            full_path = os.path.join(SAVE_DIR, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    else:
        print("  (directory is empty)")
except Exception as e:
    print(f"  Could not list directory: {e}")

print(f"{'='*70}")

# -------
# Summary
# -------
print(f"\n{'='*70}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Final Dataset:")
print(f"  Epochs: {len(epochs)}")
print(f"  Channels: {len(epochs.ch_names) - 1} EEG + 1 stimulus")
print(f"  Duration per epoch: {epochs.tmax - epochs.tmin:.1f} s")
print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
print(f"  Filter: {epochs.info['highpass']}–{epochs.info['lowpass']} Hz")
print(f"{'='*70}\n")

# 1 bad channel removed by visual inspection