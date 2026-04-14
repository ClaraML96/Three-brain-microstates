import numpy as np
import os
import struct
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

def find_exp_start_from_bdf(file_path):
    """
    Read only the Status channel from a BDF file to find:
    - ExpStart trigger (112) → crop start
    - Last real trial trigger (10-59) → crop end
    """
    with open(file_path, 'rb') as f:
        header_bytes = f.read(256)
        n_channels = int(header_bytes[252:256].decode('ascii').strip())
        chan_headers = f.read(256 * n_channels)

        labels = [
            chan_headers[i * 16:(i + 1) * 16].decode('ascii').strip()
            for i in range(n_channels)
        ]

        n_records = int(header_bytes[236:244].decode('ascii').strip())
        record_duration = float(header_bytes[244:252].decode('ascii').strip())

        spr_offset = 216 * n_channels
        samples_per_record = [
            int(chan_headers[spr_offset + i*8:spr_offset + (i+1)*8].decode('ascii').strip())
            for i in range(n_channels)
        ]

        sfreq = samples_per_record[0] / record_duration

        status_idx = next((i for i, label in enumerate(labels) if 'Status' in label), None)
        if status_idx is None:
            raise RuntimeError("No Status channel found in BDF file")

        bytes_per_sample = 3
        record_size = sum(s * bytes_per_sample for s in samples_per_record)
        status_offset_in_record = sum(
            samples_per_record[i] * bytes_per_sample for i in range(status_idx)
        )
        status_samples_per_record = samples_per_record[status_idx]
        status_bytes_per_record = status_samples_per_record * bytes_per_sample

        data_start = 256 * (n_channels + 1)

        sample_idx = 0
        exp_start_sample = None
        last_trial_sample = None
        prev_val = 0

        for rec in range(n_records):
            # Seek directly to Status channel within this record — no full record read
            f.seek(data_start + rec * record_size + status_offset_in_record)
            status_bytes = f.read(status_bytes_per_record)
            if len(status_bytes) < status_bytes_per_record:
                break

            for s in range(status_samples_per_record):
                b = status_bytes[s * 3:s * 3 + 3]
                val = struct.unpack('<I', b + b'\x00')[0] & 0xFFFF

                if val != prev_val:
                    if val == 112 and exp_start_sample is None:
                        exp_start_sample = sample_idx
                        print(f"✓ ExpStart (112) at sample {sample_idx} ({sample_idx/sfreq:.2f}s)")
                    if 10 <= val <= 59 and exp_start_sample is not None:
                        last_trial_sample = sample_idx

                prev_val = val
                sample_idx += 1

    if exp_start_sample is None:
        raise RuntimeError("ExpStart trigger (112) not found")

    exp_start_time = exp_start_sample / sfreq
    exp_end_time = (last_trial_sample / sfreq) if last_trial_sample else None
    print(f"✓ Last real trial at sample {last_trial_sample} ({exp_end_time:.2f}s)")
    return exp_start_time, exp_end_time, sfreq

def extract_bdf_to_fif(file_path, participant, exp_start_time, exp_end_time, output_path, target_sfreq=512):
    """
    Read only participant + Status channels from BDF, decimate to target_sfreq
    on the fly (record by record), and save as FIF — minimal RAM usage.
    """
    print(f"  Extracting channels directly from BDF (chunked, low-memory)...")

    with open(file_path, 'rb') as f:
        header_bytes = f.read(256)
        n_channels = int(header_bytes[252:256].decode('ascii').strip())
        chan_headers = f.read(256 * n_channels)

        labels = [chan_headers[i*16:(i+1)*16].decode('ascii').strip()
                  for i in range(n_channels)]
        n_records = int(header_bytes[236:244].decode('ascii').strip())
        record_duration = float(header_bytes[244:252].decode('ascii').strip())

        spr_offset = 216 * n_channels
        samples_per_record = [
            int(chan_headers[spr_offset + i*8:spr_offset + (i+1)*8].decode('ascii').strip())
            for i in range(n_channels)
        ]
        sfreq = samples_per_record[0] / record_duration

        # Decimation factor (e.g. 2048 → 512 = factor 4)
        decimate_factor = int(round(sfreq / target_sfreq))
        actual_sfreq = sfreq / decimate_factor
        print(f"  Native sfreq: {sfreq:.0f} Hz → decimating by {decimate_factor}x → {actual_sfreq:.0f} Hz")

        prefix = f'{participant}-'
        keep_idx = [i for i, l in enumerate(labels)
                    if l.startswith(prefix) or 'Status' in l]
        keep_labels = [labels[i].replace(prefix, '') for i in keep_idx]
        n_keep = len(keep_idx)
        print(f"  Keeping {n_keep} channels: {n_keep-1} EEG + 1 Status")

        bytes_per_sample = 3
        record_size = sum(s * bytes_per_sample for s in samples_per_record)

        start_record = int(exp_start_time / record_duration)
        end_record = min(int(np.ceil(exp_end_time / record_duration)) + 1, n_records)
        total_records = end_record - start_record

        spr_native = samples_per_record[keep_idx[0]]
        spr_decimated = spr_native // decimate_factor
        total_samples_out = total_records * spr_decimated

        print(f"  Records: {start_record}–{end_record} ({total_records} records)")
        print(f"  Output samples: {total_samples_out} (~{total_samples_out/actual_sfreq:.0f}s @ {actual_sfreq:.0f} Hz)")
        print(f"  Estimated RAM for output array: "
              f"{n_keep * total_samples_out * 8 / 1024**3:.2f} GiB")

        # Header gains/offsets
        phys_min = [float(chan_headers[104*n_channels + i*8:104*n_channels + (i+1)*8].decode('ascii').strip())
                    for i in range(n_channels)]
        phys_max = [float(chan_headers[112*n_channels + i*8:112*n_channels + (i+1)*8].decode('ascii').strip())
                    for i in range(n_channels)]
        dig_min  = [float(chan_headers[120*n_channels + i*8:120*n_channels + (i+1)*8].decode('ascii').strip())
                    for i in range(n_channels)]
        dig_max  = [float(chan_headers[128*n_channels + i*8:128*n_channels + (i+1)*8].decode('ascii').strip())
                    for i in range(n_channels)]
        gains  = [(phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i]) for i in range(n_channels)]
        offsets = [phys_min[i] - gains[i] * dig_min[i] for i in range(n_channels)]

        # Allocate decimated output only
        out_data = np.zeros((n_keep, total_samples_out), dtype=np.float32)
        
        f.seek(256 * (n_channels + 1) + start_record * record_size)

        for rec_i in range(total_records):
            record_data = f.read(record_size)
            if len(record_data) < record_size:
                break

            sample_out = rec_i * spr_decimated

            for out_i, ch_i in enumerate(keep_idx):
                ch_offset = sum(samples_per_record[j] * bytes_per_sample for j in range(ch_i))
                ch_spr = samples_per_record[ch_i]
                ch_bytes = record_data[ch_offset : ch_offset + ch_spr * bytes_per_sample]

                is_status = 'Status' in labels[ch_i]

                if is_status:
                    raw_ints = np.array([
                        struct.unpack('<I', ch_bytes[s*3:s*3+3] + b'\x00')[0] & 0xFFFF
                        for s in range(ch_spr)
                    ], dtype=np.float32)
                    out_data[out_i, sample_out:sample_out + spr_decimated] = raw_ints[::decimate_factor]
                else:
                    raw_ints = np.frombuffer(
                        b''.join(ch_bytes[s*3:s*3+3] +
                                 (b'\xff' if ch_bytes[s*3+2] & 0x80 else b'\x00')
                                 for s in range(ch_spr)),
                        dtype='<i4'
                    ).astype(np.float32)
                    scaled = (raw_ints * gains[ch_i] + offsets[ch_i]) * 1e-6
                    out_data[out_i, sample_out:sample_out + spr_decimated] = scaled[::decimate_factor]

            if rec_i % 500 == 0:
                print(f"    Record {rec_i}/{total_records} ({100*rec_i/total_records:.0f}%)")

    print("Data range check:")
    print("Min:", np.min(out_data))
    print("Max:", np.max(out_data))

    ch_types = ['eeg'] * (n_keep - 1) + ['stim']
    info = mne.create_info(ch_names=keep_labels, sfreq=actual_sfreq, ch_types=ch_types)
    raw_out = mne.io.RawArray(out_data, info, verbose=False)

    print(f"  Saving to FIF...")
    raw_out.save(output_path, overwrite=True, verbose=False)
    print(f"  ✓ Saved: {output_path}")
    return output_path

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

# Configuration
DATA_PATH = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\RawEEGData_1-4"
FILE_NAME = '302.bdf'
PARTICIPANT = 2

# Processing parameters
FILTER_LOW = 1.0    # Hz highpass filter — removes slow drifts
FILTER_HIGH = 40.0  # Hz lowpass filter — removes high-freq noise/line noise
RESAMPLE_FREQ = 512  # Hz — intentionally kept at 512 (paper uses 500 Hz)
EPOCH_TMIN = -0.5   # seconds
EPOCH_TMAX = 5.5    # seconds

# Bad channels and epochs lookup tables
BAD_CHANNELS_LOOKUP = {
    (301, 1): [], (301, 2): ['P8'], (301, 3): [],
    (302, 1): ['P1'], (302, 2): [], (302, 3): [],
    (303, 1): [], (303, 2): [], (303, 3): [],
    (304, 1): [], (304, 2): [], (304, 3): ['FT7'],
}

BAD_EPOCHS_LOOKUP = {
    (301, 1): [31, 84, 197, 236, 239, 258], 
    (301, 2): [0, 1, 16],
    (301, 3): [143, 145, 147, 148, 149, 150, 152, 154, 155, 156, 157, 231, 233, 234, 236, 237, 238, 239, 247],

    (302, 1): [80, 134, 180, 265, 266],
    (302, 2): [80, 82, 109],
    (302, 3): [80, 287],

    (303, 1): [10, 260, 295],
    (303, 2): [96, 126, 160, 225, 227, 228, 230, 231, 235, 250, 266, 267, 268, 275, 278, 285, 290],
    (303, 3): [9, 10, 112, 257, 271, 272, 291, 294, 295, 296],

    # (304, 1): [8, 11, 12, 13, 14, 17, 29, 30, 31, 34, 35, 36, 39, 42, 43, 47, 50, 51, 52, 53, 54, 55, 56, 59, 60, 64, 65, 68, 71, 73, 74, 75, 77, 78, 80, 81, 82, 83, 85, 86, 88, 89, 90, 91, 94, 100, 104, 110, 114, 118, 121, 123, 124, 129, 130, 132, 133, 134, 138, 151, 153, 154, 155, 156, 157, 159, 170, 172, 178, 183, 185, 186, 187, 189, 195, 200, 203, 207, 210, 217, 218, 219, 220, 221, 222, 227, 228, 240, 243, 246, 249, 251, 268, 270, 272, 273, 274, 277, 280, 281, 282, 283, 285, 286, 287, 290, 294, 295, 296], 
    (304, 1): [31, 52, 53, 54, 56, 60, 65, 75, 90, 94, 121, 155, 156, 159, 217, 219, 268, 270, 283],
    # (304, 2): [12, 36, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 60, 62, 63, 70, 72, 73, 75, 82, 86, 90, 91, 92, 93, 95, 98, 99, 100, 106, 125, 126, 140, 143, 146, 147, 148, 156, 157, 159, 160, 165, 166, 167, 170, 175, 177, 183, 185, 195, 197, 200, 201, 202, 221, 223, 234, 236, 243, 254, 255, 258, 259, 260, 261, 262, 265, 284, 285, 287, 288, 290, 295], 
    (304, 2): [41, 46, 50, 51, 75, 100, 126, 175, 288],
    # (304, 3): [15, 33, 35, 36, 38, 74, 85, 86, 87, 88, 95, 96, 105, 106, 107, 109, 118, 128, 150, 151, 162, 167, 169, 173, 176, 177, 178, 180, 181, 183, 184, 188, 191, 192, 193, 195, 198, 219, 223, 226, 227, 228, 229, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 268, 269, 274, 275, 276, 277, 279, 280, 281, 282, 284, 285, 286, 287, 290, 291, 292, 293, 297, 298
    (304, 3): [98, 235, 285],
}

trial_id = int(FILE_NAME.replace('.bdf', ''))

# -------
# Step 1: Load data
# -------
file_path = f"{DATA_PATH}\\{FILE_NAME}"
print(f"\n{'='*70}")
print(f"STEP 1/7: LOAD DATA")
print(f"File: {FILE_NAME} | Participant {PARTICIPANT}")
print(f"{'='*70}")

# Find crop boundaries from BDF directly (no RAM spike)
exp_start_time, last_trial_time, native_sfreq = find_exp_start_from_bdf(file_path)
exp_end_time = last_trial_time + EPOCH_TMAX + 1.0

print(f"  Cropping to: {exp_start_time:.2f}s → {exp_end_time:.2f}s  "
      f"({exp_end_time - exp_start_time:.0f}s total)")

# Extract only needed channels to a temp FIF (bypasses BDF full-width buffer)
import tempfile
temp_fif = os.path.join(tempfile.gettempdir(), f"temp_{trial_id}_p{PARTICIPANT}_raw.fif")
extract_bdf_to_fif(
    file_path,
    PARTICIPANT,
    exp_start_time,
    exp_end_time,
    temp_fif,
    target_sfreq=RESAMPLE_FREQ
)

# Load the small FIF — no memory issues
raw_p = mne.io.read_raw_fif(temp_fif, preload=False, verbose=False)
print(f"  ✓ Loaded from FIF: {len(raw_p.ch_names)} channels")
participant_channels = [
    ch for ch in raw_p.ch_names if ch.startswith(f'{PARTICIPANT}-')
]

# -------
# Step 2: Filter and resample
# -------
print(f"\n{'='*70}")
print("STEP 2/7: FILTER AND RESAMPLE")
print(f"{'='*70}")

raw_p.load_data()
print(f"  Loaded: {len(raw_p.ch_names)} channels at {raw_p.info['sfreq']:.0f} Hz")

print(f"  Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz (Hamming)")
raw_p.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, fir_design='firwin', verbose=False)

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
# trial_id = int(FILE_NAME.replace('.bdf', ''))
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