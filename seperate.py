import mne
import numpy as np
from pathlib import Path

# Define file paths
base_path = "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4"
file_names = ['301.bdf', '302.bdf', '303.bdf', '304.bdf']

# Dictionary to store all raw data
all_participants = {1: [], 2: [], 3: []}

# Load and process all files
for file_name in file_names:
    file_path = f"{base_path}\\{file_name}"
    print(f"\n{'='*60}")
    print(f"Processing {file_name}...")
    print('='*60)
    
    # Load WITHOUT preloading to save memory
    raw = mne.io.read_raw_bdf(file_path, preload=False)
    
    print(f"Total channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Sample rate: {raw.info['sfreq']} Hz")
    
    # Get channel names for each participant
    participant_1_channels = [ch for ch in raw.ch_names if ch.startswith('1-')]
    participant_2_channels = [ch for ch in raw.ch_names if ch.startswith('2-')]
    participant_3_channels = [ch for ch in raw.ch_names if ch.startswith('3-')]
    
    # Get stimulus channel
    stimulus_channels = [ch for ch in raw.ch_names if 'Status' in ch or 'STI' in ch]
    
    print(f"\nParticipant 1 channels: {len(participant_1_channels)}")
    print(f"Participant 2 channels: {len(participant_2_channels)}")
    print(f"Participant 3 channels: {len(participant_3_channels)}")
    print(f"Stimulus channels: {stimulus_channels}")
    
    # Process each participant separately to save memory
    for participant_num, channels in [(1, participant_1_channels), 
                                       (2, participant_2_channels), 
                                       (3, participant_3_channels)]:
        
        print(f"\n  Processing Participant {participant_num}...")
        
        # Pick channels for this participant
        raw_p = raw.copy().pick(channels + stimulus_channels)
        
        # Now load into memory (only this participant's data)
        raw_p.load_data()
        
        # Apply bandpass filter: 1-40 Hz
        print(f"    Filtering 1-40 Hz...")
        raw_p.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
        
        # Rename channels to remove participant prefix
        channel_mapping = {ch: ch.replace(f'{participant_num}-', '') for ch in channels}
        raw_p.rename_channels(channel_mapping)
        
        # Store in dictionary
        all_participants[participant_num].append(raw_p)
        
        print(f"    Completed! ({len(raw_p.ch_names)-1} EEG channels)")
    
    # Clear the original raw object to free memory
    del raw

# Print summary
print("\n" + "="*60)
print("SUMMARY OF ALL FILES")
print("="*60)

for participant_num in [1, 2, 3]:
    print(f"\n--- Participant {participant_num} ---")
    for idx, raw_p in enumerate(all_participants[participant_num], 1):
        file_name = file_names[idx-1]
        print(f"\n  File {file_name}:")
        print(f"    Channels: {len(raw_p.ch_names)-1} EEG + 1 stimulus")
        print(f"    Duration: {raw_p.times[-1]:.2f} seconds ({raw_p.times[-1]/60:.2f} minutes)")
        print(f"    Sample rate: {raw_p.info['sfreq']} Hz")
        print(f"    Filter: {raw_p.info['highpass']}-{raw_p.info['lowpass']} Hz")

# Optional: Concatenate all sessions for each participant
print("\n" + "="*60)
print("CONCATENATING SESSIONS FOR EACH PARTICIPANT")
print("="*60)

concatenated_participants = {}

for participant_num in [1, 2, 3]:
    print(f"\nConcatenating data for Participant {participant_num}...")
    concatenated = mne.concatenate_raws(all_participants[participant_num])
    concatenated_participants[participant_num] = concatenated
    
    print(f"  Total duration: {concatenated.times[-1]:.2f} seconds ({concatenated.times[-1]/60:.2f} minutes)")
    print(f"  Total samples: {len(concatenated.times)}")
    print(f"  Channels: {len(concatenated.ch_names)-1} EEG + 1 stimulus")
    print(f"  Filter applied: {concatenated.info['highpass']}-{concatenated.info['lowpass']} Hz")

# Optional: Save concatenated data
save_data = False  # Set to True if you want to save
if save_data:
    output_path = f"{base_path}\\processed"
    Path(output_path).mkdir(exist_ok=True)
    
    for participant_num in [1, 2, 3]:
        output_file = f"{output_path}\\participant_{participant_num}_filtered_1-40Hz_raw.fif"
        concatenated_participants[participant_num].save(output_file, overwrite=True)
        print(f"Saved: {output_file}")

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
print("\nFiltered data (1-40 Hz) is now available:")
print("  - all_participants[participant_num][session_idx]: Individual sessions")
print("  - concatenated_participants[participant_num]: All sessions combined")