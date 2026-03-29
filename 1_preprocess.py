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
FILE_NAME = '302.bdf'
PARTICIPANT = 1

# Processing parameters
FILTER_LOW = 1.0    # Hz highpass filter — removes slow drifts
FILTER_HIGH = 40.0  # Hz lowpass filter — removes high-freq noise/line noise
RESAMPLE_FREQ = 512  # Hz — intentionally kept at 512 (paper uses 500 Hz)
EPOCH_TMIN = -0.5   # seconds
EPOCH_TMAX = 5.5    # seconds

# Bad channels and epochs lookup tables
BAD_CHANNELS_LOOKUP = {
    (301, 1): [], (301, 2): ['P8', 'PO3'], (301, 3): [],
    (302, 1): ['T8', 'T7', 'P1'], (302, 2): [], (302, 3): [],
    (303, 1): ['T7'], (303, 2): ['AF8', 'TP8'], (303, 3): [],
    (304, 1): ['T7', 'T8', 'TP8', 'TP7'], (304, 2): ['F8', 'FT8', 'F7'], (304, 3): ['FT7', 'PO3'],
}

BAD_EPOCHS_LOOKUP = {
    (301, 1): [31, 76, 168, 175, 177, 197, 199, 209, 217, 225, 236, 239, 258, 272], 
    (301, 2): [1, 5, 6, 7], 
    (301, 3): [11, 85, 114, 115, 116, 118, 124, 129, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 164, 165, 166, 171, 172, 187, 189, 196, 199, 207, 208, 211, 213, 214, 215, 219, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 247, 251, 252, 289, 293],
    
    (302, 1): [81, 133, 135, 170, 181, 266, 267], 
    (302, 2):  [16, 17, 19, 21, 22, 31, 32, 34, 41, 42, 43, 44, 45, 55, 66, 67, 68, 71, 75, 76, 78, 82, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 109, 115, 116, 118, 124, 126, 127, 128, 129, 132, 133, 134, 137, 139, 143, 144, 148, 161, 165, 166, 201, 202, 221, 222, 223, 225, 227, 228, 232, 241, 247, 248, 249, 250, 251, 252, 255, 257, 266, 267, 269, 270, 271, 275, 276, 277, 278, 281, 282, 284, 285, 289, 291, 292, 293, 294, 296, 297], 
    (302, 3): [81, 95, 183, 231, 232, 272],
    
    (303, 1): [10, 11, 18, 24, 34, 37, 55, 79, 82, 86, 146, 174, 181, 183, 212, 241, 242, 243, 244, 251, 252, 261, 290, 291, 296], 
    (303, 2): [8, 10, 20, 21, 24, 30, 31, 57, 63, 69, 70, 71, 80, 81, 83, 87, 90, 91, 96, 97, 102, 103, 106, 116, 127, 130, 138, 147, 148, 151, 155, 160, 161, 171, 178, 179, 184, 186, 187, 188, 190, 193, 196, 210, 219, 226, 227, 228, 229, 230, 231, 232, 233, 236, 251, 266, 267, 268, 269, 270, 271, 276, 278, 279, 283, 284, 286, 290, 291, 299], 
    (303, 3): [9, 10, 11, 15, 16, 87, 97, 101, 112, 113, 119, 120, 123, 145, 232, 245, 247, 250, 251, 257, 258, 271, 272, 276, 277, 285, 291, 293, 294, 295, 296],
    
    (304, 1): [8, 11, 12, 13, 14, 17, 20, 22, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 42, 43, 46, 47, 50, 51, 52, 53, 54, 55, 56, 59, 60, 61, 62, 64, 65, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 94, 100, 104, 110, 113, 114, 118, 119, 120, 121, 123, 124, 128, 129, 130, 132, 133, 134, 138, 142, 151, 152, 153, 154, 155, 156, 157, 159, 167, 168, 169, 170, 172, 174, 178, 182, 183, 185, 186, 187, 189, 190, 195, 200, 201, 203, 207, 210, 211, 217, 218, 219, 220, 221, 222, 226, 227, 228, 230, 237, 240, 243, 244, 245, 246, 247, 248, 249, 250, 251, 265, 266, 268, 269, 270, 271, 272, 273, 274, 275, 277, 280, 281, 282, 283, 285, 286, 287, 290, 294, 295, 296], 
    (304, 2): [12, 36, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 60, 62, 63, 70, 72, 73, 75, 82, 86, 90, 91, 92, 93, 95, 98, 99, 100, 106, 125, 126, 140, 143, 146, 147, 148, 156, 157, 159, 160, 165, 166, 167, 170, 175, 177, 183, 185, 195, 197, 200, 201, 202, 221, 223, 234, 236, 243, 254, 255, 258, 259, 260, 261, 262, 265, 284, 285, 287, 288, 290, 295], 
    (304, 3): [15, 33, 35, 36, 38, 74, 85, 86, 87, 88, 95, 96, 105, 106, 107, 109, 118, 128, 150, 151, 162, 167, 169, 173, 176, 177, 178, 180, 181, 183, 184, 188, 191, 192, 193, 195, 198, 219, 223, 226, 227, 228, 229, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 268, 269, 274, 275, 276, 277, 279, 280, 281, 282, 284, 285, 286, 287, 290, 291, 292, 293, 297, 298],
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