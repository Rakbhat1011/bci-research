# BCI Research : EEG Analysis & Motor Imagery Classification

Interface (BCI) systems using EEG signal processing and machine learning.
---

## Projects

### 1. Motor Imagery Classifier (Public Dataset)
**Goal:** Classify left vs right hand motor imagery from EEG signals  
**Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset (public)  
**Method:** Band power features (Mu + Beta) + SVM classifier  

## Results

### Experiment 1 - Hands vs Feet Imagery (Runs 6 & 10)
| Subject | Accuracy |
|---------|----------|
| S1      | 76.7%    |
| S2      | 80.0%    |
| S3      | 56.7%    |
| S4      | 56.7%    |
| S5      | 43.3%    |
| **Mean**| **62.7%**|
| Chance  | 50.0%    |

### Experiment 2 - Left vs Right Hand Imagery (Runs 4 & 8)
| Subject | Accuracy |
|---------|----------|
| S1      | 66.7%    |
| S2      | 50.0%    |
| S3      | 66.7%    |
| S4      | 60.0%    |
| S5      | 40.0%    |
| **Mean**| **56.7%**|
| Chance  | 50.0%    |

**Why the difference?**  
Left vs right hand imagery is harder — both hands activate mirror-image 
regions of the motor cortex (C3 vs C4), making signals subtle and hard 
to distinguish. Hands vs feet activate completely different brain regions, 
making classification easier. Both are above chance, confirming the 
classifier is detecting real brain patterns.

**Key findings:**
- Mean accuracy of 62.7% across 5 subjects, well above 50% chance level
- Significant inter-subject variability (43–80%) — a known open challenge in BCI
- Most discriminative channels are occipital/parietal (O2, PO4, POz) rather 
  than expected motor cortex channels (C3/C4) — suggesting visual imagery 
  components contribute to classification

**Scripts:**
- motor_imagery_demo.py — full pipeline (download, preprocess, classify, plot)

**Plots:**
- motor_imagery_results.png - accuracy per subject + Mu power difference
- motor_imagery_channels.png - top discriminative channels + per-channel power

---

### 2. Seizure Detection (Private Dataset)
**Goal:** Detect seizure events from EEG band power features  
**Dataset:** Private clinical EEG dataset (17 participants)  
**Methods:** Random Forest, HDBSCAN, DBSCAN, UMAP  

**Key findings:**
- Random Forest + SMOTE: 99.98% seizure detection rate
- HDBSCAN outperforms DBSCAN for variable-density EEG data
- Unsupervised clustering success driven by seizure duration, not signal strength
- No single method generalises across all participants — adaptive approach needed

**Scripts:**
- seizure_classifier.py — Random Forest with SMOTE balancing
- seizure_hdbscan_perperson.py — HDBSCAN per participant
- dbscan_comparison.py — DBSCAN vs HDBSCAN comparison
- participant_comparison.py — Statistical analysis per participant
- umap_hdbscan.py — UMAP dimensionality reduction exploration
- visualise_seizure.py — EEG signal visualisation

---

### 3. Motor Imagery Paradigm
**Goal:** Build an experimental paradigm for EEG data collection  
**Tools:** Python + pygame  

A visual paradigm that presents motor imagery cues and logs timestamps 
to CSV — designed to sync with EEG recording hardware.

**Trial structure:**
Fixation (+) → 2s | Cue (OPEN/CLOSE HAND) → 4s | REST → 2s

**Script:** paradigm.py

---

## Setup

```bash
pip3 install mne numpy scipy scikit-learn matplotlib pygame hdbscan umap-learn
```

## How to Run Motor Imagery Demo

```bash
python3 motor_imagery_demo.py
```
Dataset downloads automatically via MNE-Python (~5MB per subject).

---

## Background

This research explores two tracks:
1. **Experimental paradigms** — building visual stimulus software for EEG data collection
2. **ML on EEG signals** — applying supervised and unsupervised learning to classify brain states

Key concepts: Motor Imagery, ERD (Event Related Desynchronization), 
Band Power Features, HDBSCAN clustering, inter-subject variability.
