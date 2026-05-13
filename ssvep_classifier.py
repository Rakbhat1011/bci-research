import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Step 1: Load both subjects ────────────────────
data_dir = '/Users/rakshajagadish/Documents/Projects/BCI_Research/mne_data/ssvep-example-data'

all_epochs = []

for sub in ['sub-01', 'sub-02']:
    path = f'{data_dir}/{sub}/ses-01/eeg/{sub}_ses-01_task-ssvep_eeg.vhdr'
    raw  = mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    selected = {'S155': event_dict['Stimulus/S155'],
                'S255': event_dict['Stimulus/S255']}

    epochs = mne.Epochs(raw, events, selected,
                        tmin=0., tmax=4.,
                        baseline=None, preload=True, verbose=False)

    all_epochs.append(epochs)
    print(f"{sub}: {len(epochs)} epochs")

epochs_combined = mne.concatenate_epochs(all_epochs, verbose=False)
print(f"\nTotal epochs : {len(epochs_combined)}")
print(f"Epoch shape  : {epochs_combined.get_data().shape}")

# ── Step 2: Extract SSVEP features ───────────────
data   = epochs_combined.get_data()        # (40, 32, 4001)
sfreq  = epochs_combined.info['sfreq']
labels = (epochs_combined.events[:, 2] > epochs_combined.events[:, 2].min()).astype(int)

# FFT per epoch and channel
fft_vals = np.abs(np.fft.rfft(data, axis=2))   # (40, 32, 2001)
fft_freq = np.fft.rfftfreq(data.shape[2], 1/sfreq)

# Power at stimulus frequencies and harmonics
target_freqs = [12, 15, 24, 30]

features = []
for tf in target_freqs:
    idx = np.argmin(np.abs(fft_freq - tf))
    features.append(fft_vals[:, :, idx])

X = np.hstack(features)   # (40, 128)
y = labels

print(f"\nFeature matrix: {X.shape}")
print(f"Labels: {np.unique(y, return_counts=True)}")

# ── Step 3: Classify ──────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA())
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(f"\n── Results ──────────────────────────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in scores]}")
print(f"Mean accuracy  : {scores.mean()*100:.1f}%")
print(f"Chance level   : 50.0%")

# ── Step 4: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

oz_idx  = epochs_combined.ch_names.index('Oz') \
          if 'Oz' in epochs_combined.ch_names else 0

s155_avg = fft_vals[y==0, oz_idx, :].mean(axis=0)
s255_avg = fft_vals[y==1, oz_idx, :].mean(axis=0)

axes[0].plot(fft_freq, s155_avg, color='steelblue',
             linewidth=1.5, label='S155 (12 Hz)')
axes[0].plot(fft_freq, s255_avg, color='darkorange',
             linewidth=1.5, label='S255 (15 Hz)', alpha=0.8)
axes[0].axvline(12, color='steelblue', linestyle='--', linewidth=1)
axes[0].axvline(15, color='darkorange', linestyle='--', linewidth=1)
axes[0].set_xlim(5, 25)
axes[0].set_yscale('log')
axes[0].set_title('Mean FFT Spectrum per Class\n(Channel Oz — 5-25 Hz range, log scale)')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power')
axes[0].legend()

axes[1].bar(range(1, 6), scores * 100, color='steelblue')
axes[1].axhline(scores.mean() * 100, color='green',
                linestyle='--', label=f'Mean ({scores.mean()*100:.1f}%)')
axes[1].axhline(50, color='red', linestyle='--', label='Chance (50%)')
axes[1].set_title('SSVEP Classification Accuracy\n(5-fold CV, LDA)')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('ssvep_results.png', dpi=150)
print("Saved: ssvep_results.png")
print("Done!")