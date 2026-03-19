import numpy as np
import os
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def extract_real_trial_events(events, sfreq, epoch_tmin, epoch_tmax):
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
    # Step 7: Validate condition balance (30 epochs per condition)
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
    print(f"  Epoch duration: {epoch_tmax - epoch_tmin:.1f} s")

    if min_iei_seconds < (epoch_tmax - epoch_tmin):
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
DATA_PATH = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\RawEEGData_1-4"
FILE_NAME = '301.bdf'
PARTICIPANT = 1

# Processing parameters
FILTER_LOW = 1.0    # Hz highpass filter — removes slow drifts
FILTER_HIGH = 40.0  # Hz lowpass filter — removes high-freq noise/line noise
RESAMPLE_FREQ = 512  # Hz — intentionally kept at 512 (paper uses 500 Hz)
EPOCH_TMIN = -0.5   # seconds
EPOCH_TMAX = 5.5    # seconds

# Bad channels and epochs lookup tables
BAD_CHANNELS_LOOKUP = {
    (301, 1): [], (301, 2): ['P8', 'T8', 'PO3'], (301, 3): [],
    (302, 1): [], (302, 2): [], (302, 3): [],
    (303, 1): ['T7', 'TP7'], (303, 2): [], (303, 3): ['FT7', 'FC5', 'T7'],
    (304, 1): ['T7'], (304, 2): [], (304, 3): [],
}

BAD_EPOCHS_LOOKUP = {
    (301, 1): [76], (301, 2): [0,1], (301, 3): [],
    (302, 1): [80,134,180,265,266], (302, 2): [65,66,91,239], (302, 3): [80,94],
    (303, 1): [260], (303, 2): [126,209,227,250,266,267,268,275,285,290], (303, 3): [9,119,257,272],
    (304, 1): [8,12], (304, 2): [50,93,175,265,288], (304, 3): [193,232,234,236,238,242,243,244,268,269,284,289,292],
}

# -------
# Step 1: Load data
# -------
file_path = f"{DATA_PATH}\\{FILE_NAME}"
print(f"\n{'='*70}")
print(f"STEP 1/7: LOAD DATA")
print(f"File: {FILE_NAME} | Participant {PARTICIPANT}")
print(f"{'='*70}")

raw = mne.io.read_raw_bdf(file_path, preload=False)
participant_channels = [ch for ch in raw.ch_names if ch.startswith(f'{PARTICIPANT}-')]
stimulus_channels = [ch for ch in raw.ch_names if 'Status' in ch or 'STI' in ch]
raw_p = raw.copy().pick(participant_channels + stimulus_channels)

# -------
# Step 2: Filter and resample
# -------
print(f"\n{'='*70}")
print("STEP 2/7: FILTER AND RESAMPLE")
print(f"{'='*70}")
raw_p.load_data()
print(f"  Loaded: {len(raw_p.ch_names)} channels")

print(f"  Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz (Hamming)")
raw_p.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, fir_design='firwin', verbose=False)

print(f"  Resampling: {RESAMPLE_FREQ} Hz")
raw_p.resample(sfreq=RESAMPLE_FREQ, npad='auto')

# -------
# Step 3: Bad channel interpolation
# -------
print(f"\n{'='*70}")
print("STEP 3/7: BAD CHANNEL INTERPOLATION")
print(f"{'='*70}")

# Rename channels to remove participant prefix
channel_mapping = {ch: ch.replace(f'{PARTICIPANT}-', '') for ch in participant_channels}
raw_p.rename_channels(channel_mapping)

# Look up predefined bad channels
trial_id = int(FILE_NAME.replace('.bdf', ''))
bad_channels = BAD_CHANNELS_LOOKUP.get((trial_id, PARTICIPANT), [])
raw_p.info['bads'] = bad_channels

print(f"Predefined Bad Channels:")
print(f"  Trial ID: {trial_id}, Participant: {PARTICIPANT}")
print(f"  Bad channels: {len(bad_channels)}")
if bad_channels:
    print(f"  Channels: {', '.join(bad_channels)}")

# Set standard 10-20 montage (required for spherical spline interpolation)
print(f"  Setting standard 10-20 montage...")
montage = mne.channels.make_standard_montage("standard_1020")
raw_p.set_montage(montage)

# Interpolate only when bad channels are present
if raw_p.info['bads']:
    print(f"  Interpolating {len(raw_p.info['bads'])} bad channel(s)...")
    raw_p.interpolate_bads(reset_bads=True)
else:
    print(f"  No bad channels to interpolate")

# -------
# Step 4: Apply common average reference
# -------
# IMPORTANT: Average reference is applied here, after bad channel interpolation,
# following the pipeline order described in the paper. This ensures ICA receives
# properly referenced data. Do NOT re-apply average reference in the ICA script.
# -------
print(f"\n{'='*70}")
print("STEP 4/7: APPLY COMMON AVERAGE REFERENCE")
print(f"{'='*70}")
raw_p.set_eeg_reference("average", projection=False, verbose=False)
print("✓ Common average reference applied")
print("  Note: applied after bad channel interpolation, before epoching")
print("  This reference will carry through to ICA — do not re-apply there")

# -------
# Step 5: Extract and validate trial events
# -------
print(f"\n{'='*70}")
print("STEP 5/7: EXTRACT AND VALIDATE TRIAL EVENTS")
print(f"{'='*70}")
events = mne.find_events(raw_p, stim_channel='Status', shortest_event=1, verbose=False)

collapsed_events, event_id = extract_real_trial_events(
    events,
    raw_p.info['sfreq'],
    EPOCH_TMIN,
    EPOCH_TMAX
)

# -------
# Step 6: Create epochs
# -------
print(f"\n{'='*70}")
print("STEP 6/7: CREATE EPOCHS")
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
# Step 7: Drop bad epochs
# -------
print(f"\n{'='*70}")
print("STEP 7/7: DROP BAD EPOCHS")
print(f"{'='*70}")

initial_count = len(epochs)
bad_epoch_indices_1based = BAD_EPOCHS_LOOKUP.get((trial_id, PARTICIPANT), [])
bad_epoch_indices_0based = [idx - 1 for idx in bad_epoch_indices_1based]

print(f"Trial ID: {trial_id}, Participant: {PARTICIPANT}")
print(f"Predefined bad epochs (1-based): {bad_epoch_indices_1based}")

if bad_epoch_indices_0based:
    epochs.drop(bad_epoch_indices_0based, reason='PREDEFINED_BAD', verbose=False)
    final_count = len(epochs)
    dropped = initial_count - final_count
    print(f"\nEpochs dropped: {dropped}")
    print(f"Epochs remaining: {final_count}")
else:
    print(f"\nNo predefined bad epochs to drop")
    print(f"Epochs remaining: {initial_count}")

print(f"{'='*70}")

# -------
# Save cleaned epochs for ICA
# -------
print(f"\n{'='*70}")
print("SAVING CLEANED EPOCHS")
print(f"{'='*70}")

print(f"Current working directory: {os.getcwd()}")

SAVE_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\preprocessed"
print(f"Target save directory: {SAVE_DIR}")

try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✓ Directory created/verified")
except Exception as e:
    print(f"⚠ Error creating directory: {e}")
    raise

if os.path.isdir(SAVE_DIR):
    print(f"✓ Directory exists and is accessible")
else:
    raise FileNotFoundError(f"Failed to create directory: {SAVE_DIR}")

EPOCHS_FILE = os.path.join(SAVE_DIR, f"{trial_id}_p{PARTICIPANT}_clean-epo.fif")
print(f"Output file path: {EPOCHS_FILE}")

try:
    print(f"Saving epochs...")
    epochs.save(EPOCHS_FILE, overwrite=True, verbose=True)
    print(f"✓ epochs.save() completed without errors")
except Exception as e:
    print(f"⚠ Error during save: {e}")
    raise

if os.path.exists(EPOCHS_FILE):
    file_size_mb = os.path.getsize(EPOCHS_FILE) / (1024 * 1024)
    print(f"✓ File created successfully")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Location: {EPOCHS_FILE}")
else:
    raise FileNotFoundError(f"File was not created: {EPOCHS_FILE}")

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
print(f"  Reference: common average (applied before epoching)")
print(f"{'='*70}\n")