import mne
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

# Define file path
base_path = "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4"

# Choose which file and participant to visualize
file_name = '301.bdf'  
participant_num = 3     

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

# Rename channels to remove participant prefix
channel_mapping = {ch: ch.replace(f'{participant_num}-', '') for ch in participant_channels}
raw_p.rename_channels(channel_mapping)

print(f"\nData loaded successfully!")
print(f"Channels: {len(raw_p.ch_names)-1} EEG + 1 stimulus")
print(f"Duration: {raw_p.times[-1]:.2f} seconds ({raw_p.times[-1]/60:.2f} minutes)")
print(f"Sample rate: {raw_p.info['sfreq']} Hz")
print(f"Filter: {raw_p.info['highpass']}-{raw_p.info['lowpass']} Hz")

print("\n" + "="*60)
print("Opening interactive plot...")
print("="*60)
print("\nControls:")
print("  - Use arrow keys to navigate through time")
print("  - Use +/- to zoom in/out")
print("  - Click on channels to mark as bad")
print("  - Press 'h' for help")
print("="*60)

# Create the interactive plot
fig = raw_p.plot(
    n_channels=20,      # Show 20 channels at a time
    duration=10,        # Show 10 seconds of data
    scalings='auto',    # Auto-scale amplitudes
    start=0,            # Start at beginning
    block=True          # Keep window open
)

plt.show()

print("\nPlot closed. Done!")