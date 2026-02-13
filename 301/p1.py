import mne
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt


def manual_epoch_inspection(epochs):
    """
    Manually inspect epochs to identify and remove bad epochs.
    
    Opens an interactive plot for visual inspection, allowing you to mark
    bad epochs by clicking on them. Then removes all marked bad epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to inspect
    
    Returns
    -------
    epochs : mne.Epochs
        The cleaned epochs with bad epochs removed
    """
    print("\n" + "="*60)
    print("Manual Epoch Inspection")
    print("="*60)
    print("Instructions:")
    print("  - Click on epochs to mark them as bad")
    print("  - Use arrow keys to navigate")
    print("  - Close the window when done")
    print("="*60)
    
    # Open interactive plot
    fig = epochs.plot(
        n_channels=32,      # Show 32 channels
        n_epochs=5,         # Show 5 epochs at a time
        scalings=dict(eeg=50e-6),  # Fixed scaling: 50 microvolts
        block=True          # Block until window closes
    )
    
    plt.show()
    
    # Remove bad epochs
    epochs.drop_bad()
    
    # Print results
    print(f"Remaining epochs: {len(epochs)}")
    
    return epochs


def amplitude_rejection(epochs, threshold_uv=150):
    """
    Reject epochs exceeding an amplitude threshold.
    
    Automatically removes epochs where any EEG channel exceeds the
    specified amplitude threshold.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to clean
    threshold_uv : float
        Amplitude threshold in microvolts (default: 150)
    
    Returns
    -------
    epochs : mne.Epochs
        The cleaned epochs with high-amplitude epochs removed
    """
    # Convert threshold from microvolts to volts
    threshold_v = threshold_uv * 1e-6
    
    # Count initial epochs
    initial_count = len(epochs)
    
    # Drop epochs exceeding threshold
    epochs.drop_bad(reject=dict(eeg=threshold_v), verbose=False)
    
    # Calculate dropped and remaining counts
    remaining_count = len(epochs)
    dropped_count = initial_count - remaining_count
    
    # Print summary
    print(f"Amplitude Rejection ({threshold_uv} µV):")
    print(f"  Initial:  {initial_count} epochs")
    print(f"  Dropped:  {dropped_count} epochs")
    print(f"  Remaining: {remaining_count} epochs")
    
    return epochs


# Define file path
base_path = "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4"

# Choose which file and participant to visualize
file_name = '301.bdf'  
participant_num = 1    

file_path = f"{base_path}\\{file_name}"

print(f"Loading {file_name} for Participant {participant_num}...")
print("="*60)

# Load the raw data (without preloading to save memory)
raw = mne.io.read_raw_bdf(file_path, preload=False)

# Get channels for the selected participant
participant_channels = [ch for ch in raw.ch_names if ch.startswith(f'{participant_num}-')]
stimulus_channels = [ch for ch in raw.ch_names if 'Status' in ch or 'STI' in ch]

# Pick only this participant's channels
raw_p = raw.copy().pick(participant_channels + stimulus_channels)

# Load data into memory
print("Loading data...")
raw_p.load_data()

# Apply bandpass filter: 1-40 Hz
print("Filtering 1-40 Hz...")
raw_p.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

# Downsample to 512 Hz
print("Downsampling to 512 Hz...")
raw_p.resample(sfreq=512, npad='auto')

# Rename channels to remove participant prefix
channel_mapping = {ch: ch.replace(f'{participant_num}-', '') for ch in participant_channels}
raw_p.rename_channels(channel_mapping)

# Find events (trial onsets)
print("Finding events...")
events = mne.find_events(raw_p, stim_channel='Status', shortest_event=1, verbose=False)
print(f"Found {len(events)} events")

# Create epochs: -0.5 to 5.5 seconds relative to trial onset
print("Creating epochs (-0.5 to 5.5 s)...")
epochs = mne.Epochs(raw_p, events, tmin=-0.5, tmax=5.5, baseline=None, preload=True, verbose=False)
print(f"Created {len(epochs)} epochs")

# Apply amplitude rejection
epochs = amplitude_rejection(epochs, threshold_uv=300)

print(f"\nData loaded successfully!")
print(f"Channels: {len(epochs.ch_names)-1} EEG + 1 stimulus")
print(f"Epochs: {len(epochs)}")
print(f"Epoch duration: {epochs.tmax - epochs.tmin:.1f} seconds")
print(f"Sample rate: {epochs.info['sfreq']} Hz")
print(f"Filter: {epochs.info['highpass']}-{epochs.info['lowpass']} Hz")

print("\n" + "="*60)
print("Opening interactive plot...")
print("="*60)
print("\nControls:")
print("  - Use arrow keys to navigate between epochs")
print("  - Use +/- to zoom in/out")
print("  - Click on channels to mark as bad")
print("  - Press 'h' for help")
print("="*60)

# Perform manual epoch inspection
epochs = manual_epoch_inspection(epochs)

print(f"Final epochs after manual marking: {len(epochs)}")

plt.show()

print("\nDone!")

plt.show()

print("\nPlot closed. Done!")