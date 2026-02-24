import numpy as np
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


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

# Configuration
DATA_PATH = "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4"
FILE_NAME = '301.bdf'
PARTICIPANT = 1

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
# Step 4: Re-reference to common average
# -------
print(f"\nRe-referencing:")
raw_p.set_eeg_reference('average', projection=False, verbose=False)
print(f"  Applied common average reference")

# -------
# Step 5: Find events
# -------
events = mne.find_events(raw_p, stim_channel='Status', shortest_event=1, verbose=False)
print(f"  Found {len(events)} trial onsets")

# -------
# Step 6: Create epochs
# -------
print(f"\nEpoch creation:")
print(f"  Window: {EPOCH_TMIN} to {EPOCH_TMAX} s")
epochs = mne.Epochs(
    raw_p, events,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=None,
    preload=True,
    verbose=False
)
print(f"  Created: {len(epochs)} epochs")

# -------
# Step 7: Manual inspection
# -------
epochs = manual_epoch_inspection(epochs)

# ============================================================================
# ICA ARTIFACT REMOVAL
# ============================================================================

# -------
# Step 8: Run Picard ICA with 32 components
# -------
print(f"\n{'='*70}")
print(f"ICA DECOMPOSITION")
print(f"{'='*70}")

print(f"Fitting ICA:")
print(f"  Method: Picard")
print(f"  Components: 32")
print(f"  Random state: 97")

ica = mne.preprocessing.ICA(
    n_components=32,
    method='picard',
    random_state=97,
    max_iter='auto'
)

# Fit ICA on epoched data
ica.fit(epochs, verbose=False)
print(f"  ICA fitted successfully")
print(f"  Explained variance: {ica.get_explained_variance_ratio(epochs)['eeg']:.2%}")

# -------
# Step 9: Visual identification of artifact components
# -------
print(f"\n{'='*70}")
print(f"COMPONENT IDENTIFICATION")
print(f"{'='*70}")
print(f"Identify artifact components:")
print(f"  - Ocular artifacts (eye blinks, saccades)")
print(f"  - Cardiac artifacts (heartbeat)")
print(f"  - Muscle artifacts (EMG)")
print(f"Close the plot windows when done.")
print(f"{'='*70}")

# Plot component topographies
ica.plot_components(inst=epochs, picks=range(32))

# Plot component time courses and spectra
ica.plot_sources(epochs, show_scrollbars=False, block=True)

# -------
# Step 10: Manual selection and removal of artifact components
# -------
print(f"\nEnter component numbers to exclude (comma-separated, e.g., 0,3,7):")
user_input = input("Components to remove: ").strip()

if user_input:
    # Parse user input
    exclude_components = [int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()]
    ica.exclude = exclude_components
    
    print(f"\nRemoving components:")
    print(f"  Selected: {exclude_components}")
    print(f"  Count: {len(exclude_components)} components")
    
    # Apply ICA to remove selected components
    epochs = ica.apply(epochs, verbose=False)
    print(f"  ICA applied - artifacts removed")
else:
    print(f"\nNo components selected - skipping ICA removal")

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

# 1 epoch removed by visual inspection