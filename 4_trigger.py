# import mne
# import numpy as np

# raw = mne.io.read_raw_bdf(
#     r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\RawEEGData_1-4\301.bdf",
#     preload=False
# )

# # Pick participant 1's channels + status
# p_chans = [ch for ch in raw.ch_names if ch.startswith('1-')]
# raw_p = raw.copy().pick(p_chans + ['Status'])

# events = mne.find_events(raw_p, stim_channel='Status', 
#                           shortest_event=1, verbose=False)

# exp_start = events[events[:, 2] == 112][0, 0]
# sfreq = raw.info['sfreq']
# post = events[events[:, 0] > exp_start]

# # Show ALL unique codes with counts — including non-trial ones
# print("ALL trigger codes after ExpStart:")
# print(f"{'Code':>6}  {'Count':>6}  {'First time (s)':>16}")
# print("-" * 35)
# unique, counts = np.unique(post[:, 2], return_counts=True)
# for code, count in zip(unique, counts):
#     t = (post[post[:, 2] == code][0, 0] - exp_start) / sfreq
#     print(f"{code:>6}  {count:>6}  {t:>14.1f}s")

# # Show first 60 trials in order so you can see the sequence/pattern
# real = post[(post[:, 2] >= 10) & (post[:, 2] <= 59)]
# print(f"\nFirst 60 real trial codes (chronological):")
# print(f"{'#':>4}  {'Raw code':>10}  {'→ Condition':>12}  {'Time (s)':>10}")
# print("-" * 42)
# for i, ev in enumerate(real[:60]):
#     t = (ev[0] - exp_start) / sfreq
#     cond = ev[2] % 10
#     print(f"{i+1:>4}  {ev[2]:>10}  {cond:>12}  {t:>10.1f}")


import pandas as pd

# Update path to wherever your folder is
force_df = pd.read_pickle(r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students\Force_df_v2.pkl")   # or .pkl / .fif etc.

print(force_df.dtypes)
print(force_df.head(20))
print(force_df.columns.tolist())

# If there's a condition column, show unique values
for col in force_df.columns:
    if 'cond' in col.lower() or 'task' in col.lower() or 'trial' in col.lower():
        print(f"\n{col}:", force_df[col].unique())