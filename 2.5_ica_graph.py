import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\data\ica_cleaned"

PARTICIPANT_ID = "303"
PARTICIPANT = 1
EPOCH_FILE = os.path.join(
    DATA_DIR,
    f"{PARTICIPANT_ID}_p{PARTICIPANT}_ica_cleaned-epo.fif"
)

# Channel of interest
CHANNEL = "C3"

# ============================================================
# STEP 1: Load cleaned epochs
# ============================================================

print("Loading ICA-cleaned epochs...")

epochs = mne.read_epochs(EPOCH_FILE, preload=True)

print(f"Epochs loaded: {len(epochs)}")
print(f"Channels: {len(epochs.ch_names)}")

# ============================================================
# STEP 2: Extract data for channel C3
# ============================================================

print(f"Extracting channel: {CHANNEL}")

data = epochs.copy().pick(CHANNEL).get_data()

# shape = (n_epochs, 1, n_times)
data = data[:, 0, :]   # remove channel dimension

times = epochs.times

# ============================================================
# STEP 3: Compute average across trials
# ============================================================

avg_signal = np.mean(data, axis=0)

# ============================================================
# PLOT 1: Average signal across trials (C3)
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(times, avg_signal)

plt.axvline(0, linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title(f"Average ERP across trials") 

plt.show()

# ============================================================
# PLOT 2: All trials + average
# ============================================================

# plt.figure(figsize=(8,4))

# for trial in data:
#     plt.plot(times, trial, alpha=0.1)

# plt.plot(times, avg_signal, linewidth=3)

# plt.axvline(0, linestyle="--")

# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (µV)")
# plt.title(f"All trials + average – Channel {CHANNEL}")

# plt.show()

# ============================================================
# PLOT 3: Standard ERP plot (all channels)
# ============================================================

print("Plotting ERP for all channels")

evoked = epochs.average()

evoked.plot()

# ============================================================
# PLOT 4: Topographic map
# ============================================================

print("Plotting ERP topography")

evoked.plot_topomap(times=np.linspace(0.05,0.4,6))

# ============================================================
# PLOT 5: Global Field Power (GFP)
# ============================================================

# print("Computing Global Field Power")

# evoked_data = evoked.data

# gfp = np.std(evoked_data, axis=0)

# plt.figure(figsize=(8,4))

# plt.plot(times, gfp)

# plt.axvline(0, linestyle="--")

# plt.xlabel("Time (s)")
# plt.ylabel("GFP")

# plt.title("Global Field Power")

# plt.show()

print("Visualization complete")