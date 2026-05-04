import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Step 1: Load P300 dataset ────────────────────
# Using MNE's built-in ERP CORE dataset — proper P300 data
print("Downloading P300 dataset...")

from mne.datasets import erp_core
data_path = erp_core.data_path()

# Load subject 1, P3 task (the P300 oddball paradigm)
raw = mne.io.read_raw_fif(
    "/Users/rakshajagadish/Documents/Projects/BCI_Research/MNE-ERP-CORE-data/ERP-CORE_Subject-001_Task-Flankers_eeg.fif",
    preload=True, verbose=False
)

print(f"Channels: {len(raw.ch_names)}")
print(f"Sample rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")

# ── Step 2: Filter ────────────────────────────────
raw.filter(1., 20., verbose=False)
print("Applied 1-20 Hz filter")

# ── Step 3: Extract epochs ────────────────────────
events, event_dict = mne.events_from_annotations(raw, verbose=False)
print(f"\nEvents found: {event_dict}")

# In P3 task: targets (rare stimuli) produce P300
# Non-targets (frequent stimuli) do not
target_id    = {k: v for k, v in event_dict.items() if 'target' in k.lower()}
nontarget_id = {k: v for k, v in event_dict.items() if 'non' in k.lower() or 'frequent' in k.lower()}

print(f"Target events    : {target_id}")
print(f"Non-target events: {nontarget_id}")

selected = {**{'Target': list(target_id.values())[0]},
            **{'NonTarget': list(nontarget_id.values())[0]}} \
    if target_id and nontarget_id else event_dict

print(f"Using events: {selected}")

epochs = mne.Epochs(raw, events, selected,
                    tmin=-0.1, tmax=0.8,
                    baseline=(-0.1, 0),
                    preload=True, verbose=False)

print(f"\nEpochs: {len(epochs)}")
for k in selected:
    print(f"  {k}: {len(epochs[k])}")

# ── Step 4: Extract features ─────────────────────
X = epochs.get_data()
n_epochs, n_channels, n_times = X.shape
X_flat = X.reshape(n_epochs, n_channels * n_times)
y      = (epochs.events[:, 2] == list(selected.values())[0]).astype(int)

print(f"\nFeature matrix: {X_flat.shape}")
print(f"Labels: {np.unique(y, return_counts=True)}")

# ── Step 5: Classify ──────────────────────────────
print("\nTraining LDA classifier...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA())
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_flat, y, cv=cv, scoring='accuracy')

print(f"\n── Results ──────────────────────────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in scores]}")
print(f"Mean accuracy  : {scores.mean()*100:.1f}%")
print(f"Chance level   : 50.0%")

# ── Step 6: Plot ──────────────────────────────────
print("\nGenerating plots...")

# Use actual event names from this dataset
target_avg    = epochs['stimulus/compatible/target_left'].average()
nontarget_avg = epochs['response/left'].average()
times         = epochs.times * 1000

# Pick a central channel
ch_idx = 0
if 'Cz' in epochs.ch_names:
    ch_idx = epochs.ch_names.index('Cz')
elif 'CPz' in epochs.ch_names:
    ch_idx = epochs.ch_names.index('CPz')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(times, target_avg.data[ch_idx] * 1e6,
             color='red', linewidth=2, label='Target stimulus')
axes[0].plot(times, nontarget_avg.data[ch_idx] * 1e6,
             color='steelblue', linewidth=2, label='Response')
axes[0].axvline(0, color='black', linestyle='--',
                linewidth=0.8, label='Stimulus onset')
axes[0].axvline(300, color='gray', linestyle=':',
                linewidth=0.8, label='300ms')
axes[0].set_title(f"Average ERP Waveform\n(Channel: {epochs.ch_names[ch_idx]})")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude (µV)")
axes[0].legend()

axes[1].bar(range(1, 6), scores * 100, color='steelblue')
axes[1].axhline(scores.mean() * 100, color='green',
                linestyle='--', label=f'Mean ({scores.mean()*100:.1f}%)')
axes[1].axhline(50, color='red', linestyle='--', label='Chance (50%)')
axes[1].set_title("ERP Classification Accuracy\n(5-fold CV, LDA)")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig("p300_results.png", dpi=150)
print("Saved: p300_results.png")
print("\nDone!")