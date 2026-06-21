import glob, os, itertools
import numpy as np, pandas as pd, mne

DATA_DIR = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\PreprocessedEEGData"
)
OVERVIEW_PKL = (
    r"C:\Users\clara\OneDrive - Danmarks Tekniske Universitet\Skrivebord\DTU"
    r"\Human Centeret Artificial Intelligence\Thesis\FG_Data_For_Students"
    r"\FG_overview_df_v2.pkl"
)
CONTRAST, EXCLUDE = ("T3P", "T1P"), [] #[330]

fg = pd.read_pickle(OVERVIEW_PKL)
sel = {}   # subj_id -> {cond: selection}
for f in sorted(glob.glob(os.path.join(DATA_DIR, "*_FG_preprocessed-epo.fif"))):
    m = fg[fg["Exp_id"] == os.path.basename(f).split("_")[0]]
    if m.empty: continue
    ep = mne.read_epochs(f, preload=False, verbose=False)
    sel[int(m["Subject_id"].iloc[0])] = {c: ep[c].selection
                                         for c in CONTRAST if c in ep.event_id}

rows = []
for tid, grp in fg.groupby("Triad_id"):
    sids = grp["Subject_id"].astype(int).tolist()
    if tid in EXCLUDE:
        rows.append((tid, np.nan, "excluded (config)")); continue
    if not all(s in sel for s in sids):
        rows.append((tid, np.nan, "missing member file")); continue
    if not all(len(sel[s]) == 2 for s in sids):
        rows.append((tid, np.nan, "missing a condition")); continue
    per_pair = [min(len(np.intersect1d(sel[a][CONTRAST[0]], sel[b][CONTRAST[0]])),
                    len(np.intersect1d(sel[a][CONTRAST[1]], sel[b][CONTRAST[1]])))
                for a, b in itertools.combinations(sids, 2)]
    rows.append((tid, min(per_pair), f"min matched n = {min(per_pair)}"))

tbl = pd.DataFrame(rows, columns=["Triad_id", "min_matched_n", "note"])
for thr in (15, 20, 25, 30):
    tbl[f"in@{thr}"] = tbl["min_matched_n"] >= thr
print(tbl.to_string(index=False))
print("\nIncluded triads by N_MIN:")
for thr in (15, 20, 25, 30):
    print(f"  N_MIN={thr}:  {int(tbl[f'in@{thr}'].sum())}")