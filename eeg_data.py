import mne

raw = mne.io.read_raw_bdf(
    "C:\\Users\\clara\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\DTU\\Human Centeret Artificial Intelligence\\Thesis\\FG_Data_For_Students\\RawEEGData_1-4\\301.bdf",
    preload=True
)

print(raw.info)
