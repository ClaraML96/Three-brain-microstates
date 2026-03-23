# EEG Preprocessing Pipeline

## Overview

This repository contains a structured EEG preprocessing pipeline built with **MNE-Python** and **AutoReject**. It covers the full workflow from raw BioSemi `.bdf` recordings to fully cleaned, analysis-ready epochs.

The pipeline is designed for **multi-participant recordings stored within a single file**, where each participant’s channels are prefixed (e.g., `1-Fz`, `2-Fz`).

---

## Pipeline Structure

### 1. Initial Preprocessing (`preprocessing.py`)

Transforms raw EEG into cleaned epochs:

* Load raw `.bdf` data
* Band-pass filtering (1–40 Hz)
* Resampling (512 Hz)
* Bad channel interpolation (based on predefined lookup)
* Event extraction and validation
* Epoching (-0.5 to 5.5 s)
* Removal of predefined bad epochs

**Output:**

```
*_clean-epo.fif
```

---

### 2. ICA Cleaning (separate step)

Independent Component Analysis (ICA) is applied to remove artifacts such as:

* Eye blinks
* Muscle activity
* Line noise

**Output:**

```
*_ica_cleaned-epo.fif
```

---

### 3. Final Cleaning (`autoreject_pipeline.py`)

Final quality control using AutoReject and manual inspection:

* Adaptive artifact detection (AutoReject)
* Visualization of rejected segments
* Optional manual epoch rejection
* Saving rejection logs for reproducibility

**Output:**

```
*_final_clean-epo.fif
*_reject_log.npz
```
