# Neural oscillations underlying group coordination MSc Thesis

A repository dedicated to the analysis of EEG hyperscanning data tracking brain activity and inter-brain synchronization during a multi-person force-coordination task.

## Project Overview

This project investigates the neural mechanisms behind social coordination by moving beyond single-subject paradigms. Using EEG hyperscanning on a sample size of 92 participants, we assess how group size (Solo vs. Trio) and the presence/absence of visual feedback shape both localized neural dynamics and inter-subject synchronization.

## Methodology & Pipeline

* Experimental Design: Force-coordination task completed across multiple social architectures:
    * Solo vs. Trio conditions.
    * With Visual Feedback vs. No Visual Feedback conditions.

* Intra-Brain Analysis:
    * Event-Related Potentials (ERP)
    * Time-Frequency Representations (TFR)
    * Event-Related Desynchronization/Synchronization (ERD/ERS)

* Inter-Brain Analysis: Inter-brain synchronization quantified via Phase-Locking Value (PLV).

* Statistical Inference: Non-parametric cluster-based permutation testing across the scalp.


## The scripts

Active scripts by stage. Walkthroughs and notes are kept alongside the scripts.

### Thesis scripts

| Script | Folder | 
| --- | --- |
| [01_preprocess.py](preprocess/01_preprocess.py) | `preprocess` | 
| [02_ica.py](preprocess/02_ica.py) | `preprocess` | 
| [04_erp_92_participants.py](visualizations/04_erp_92_participants.py) | `visualizations` | 
| [05_tfr_joint_92_participants.py](visualizations/05_tfr_joint_92_participants.py) | `visualizations` | 
| [06_erd_92_participants.py](visualizations/06_erd_92_participants.py) | `visualizations` |
| [erd_cluster_stats.py](statistics/erd_cluster_stats.py) | `statistics` | 
| [07_permutation.py](statistics/07_permutation.py) | `statistics` |
| [07_permutation_joint.py](statistics/07_permutation_joint.py) | `statistics` | 
| [08_plv_condition_contrast.py](inter-brain-sync/08_plv_condition_contrast.py) | `inter-brain-sync` |
| [08_plv_boxplots.py](inter-brain-sync/08_plv_boxplots.py) | `inter-brain-sync` | 

### Exploratory / side-track scripts

| Script | Folder | Stage |
| --- | --- | --- |
| [05_tfr_group.py](visualizations/05_tfr_group.py) | `visualizations` | Alternative / exploratory TFR aggregation |
| [05_tfr_subject.py](visualizations/05_tfr_subject.py) | `visualizations` | Subject-level TFR inspection |
| [06_erd_92_participants.py](visualizations/06_erd_allChannels.py) | `visualizations` | ERD/ERS across regions of interst |
| [06_erd_ers.py](visualizations/06_erd_ers.py) | `visualizations` | ERD/ERS variant view |
| [06_erd_roi.py](visualizations/06_erd_roi.py) | `visualizations` | ROI-based ERD summary |
| [06_erd_exactComparison.py](visualizations/06_erd_exactComparison.py) | `visualizations` | Exact-condition comparison helper |
| [07_permutation_ROI.py](statistics/07_permutation_ROI.py) | `statistics` | ROI-focused permutation variant |
| [08_plv.py](inter-brain-sync/08_plv.py) | `inter-brain-sync` | Earlier PLV pipeline |
| [08_plv_abs.py](inter-brain-sync/08_plv_abs.py) | `inter-brain-sync` | Absolute-PLV variant |
| [08_plv_triadAnalysis.py](inter-brain-sync/08_plv_triadAnalysis.py) | `inter-brain-sync` | Triad-level PLV analysis |

## Two input sources — and the split that matters

The numbering suggests one linear pipeline. It is not. The scripts draw on two different data sources, and the `01 -> 02` chain feeds nothing else in the vault.

```text
SOURCE 1 — raw BioSemi recordings
	RAW BDF -> 01_preprocess.py -> 02_ica.py -> *_clean-epo.fif / *_ica_cleaned-epo.fif
											  └-> consumed by nothing else here

SOURCE 2 — preprocessed epochs used by the analysis scripts
	-> visualizations/
	-> statistics/
	-> inter-brain-sync/
```

Read this honestly: `01_preprocess.py` and `02_ica.py` are a self-contained preprocessing reimplementation demonstrating LO2 competence on the raw BDF, not the source of the epochs the analyses use. The analysis scripts consume the preprocessed epoch set directly.


### Condition coding is mostly unified

The current `visualizations/`, `statistics/`, and `inter-brain-sync/` scripts all key on the paper's string event ids (`T1P`, `T1Pn`, `T3P`, `T3Pn`, plus the duo codes where relevant). The older numeric-code path in preprocessing is separate and does not feed the later analysis scripts.

### Duplicated load-and-TFR loops

The same per-subject loop is copied across the TFR and ERD scripts. The statistics layer now centralises the core load / TFR / permutation logic, but the visualisation scripts still carry their own copies and should be aligned if the pipeline is cleaned up further.

### Inconsistent baseline windows

ERP and TFR / ERD scripts do not share the same baseline window. Both choices are defensible, but the difference should stay explicit in the methods rather than hidden in constants.
