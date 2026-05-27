import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ── Step 1: Load both subjects ────────────────────
data_dir = '/Users/rakshajagadish/Documents/Projects/BCI_Research/mne_data/ssvep-example-data'

all_epochs = []

for sub in ['sub-01', 'sub-02']:
    path = f'{data_dir}/{sub}/ses-01/eeg/{sub}_ses-01_task-ssvep_eeg.vhdr'
    raw  = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    raw.filter(1., None, verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    selected = {'S155': event_dict['Stimulus/S155'],
                'S255': event_dict['Stimulus/S255']}

    epochs = mne.Epochs(raw, events, selected,
                        tmin=0., tmax=4.,
                        baseline=None, preload=True, verbose=False)
    all_epochs.append(epochs)

epochs_combined = mne.concatenate_epochs(all_epochs, verbose=False)
data   = epochs_combined.get_data()           # (40, 32, 4001)
sfreq  = epochs_combined.info['sfreq']
labels = (epochs_combined.events[:, 2] > epochs_combined.events[:, 2].min()).astype(int)

print(f"Epochs: {data.shape[0]} | Channels: {data.shape[1]} | Samples: {data.shape[2]}")
print(f"Labels: {np.unique(labels, return_counts=True)}")

# ── Step 2: Build CCA reference signals ──────────
# sine and cosine waves at the fundamental + harmonics
def make_reference(freq, sfreq, n_samples, n_harmonics=2):
    t = np.arange(n_samples) / sfreq
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs).T   # (n_samples, n_harmonics*2)

n_samples   = data.shape[2]
stim_freqs = [15.0, 12.0]   # Hz — S155=12Hz, S255=15Hz
n_harmonics = 2

references = [make_reference(f, sfreq, n_samples, n_harmonics)
              for f in stim_freqs]

print(f"\nReference signal shape: {references[0].shape}")
print(f"(n_samples x n_harmonics*2 = {n_samples} x {n_harmonics*2})")

# ── Step 3: CCA classification ────────────────────
# For each epoch, compute CCA correlation with each reference
# Predict the class with the highest correlation
def cca_predict(epoch_data, references):
    """
    epoch_data: (n_channels, n_samples)
    references: list of (n_samples, n_ref_signals)
    Returns: predicted class index (0 or 1)
    """
    correlations = []
    X = epoch_data.T   # (n_samples, n_channels)

    for ref in references:
        cca = CCA(n_components=1)
        cca.fit(X, ref)
        X_c, ref_c = cca.transform(X, ref)
        # Correlation between first canonical variates
        corr = np.corrcoef(X_c[:, 0], ref_c[:, 0])[0, 1]
        correlations.append(corr)

    return np.argmax(correlations)

# Predict all epochs
print("\nRunning CCA classification...")
predictions = np.array([cca_predict(data[i], references)
                        for i in range(len(data))])

# Calculate accuracy
accuracy = (predictions == labels).mean()
print(f"\n── CCA Results ──────────────────────────────")
print(f"Predictions: {predictions}")
print(f"True labels: {labels}")
print(f"CCA Accuracy: {accuracy*100:.1f}%")
print(f"Chance level: 50.0%")

# ── Step 4: Compare CCA vs LDA ────────────────────
# LDA on FFT features 
fft_vals = np.abs(np.fft.rfft(data, axis=2))
fft_freq = np.fft.rfftfreq(data.shape[2], 1/sfreq)

target_freqs = [12, 15, 24, 30]
features = []
for tf in target_freqs:
    idx = np.argmin(np.abs(fft_freq - tf))
    features.append(fft_vals[:, :, idx])
X_lda = np.hstack(features)

pipeline = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lda_scores = cross_val_score(pipeline, X_lda, labels, cv=cv, scoring='accuracy')

print(f"\n── LDA Results (for comparison) ─────────────")
print(f"LDA Mean Accuracy: {lda_scores.mean()*100:.1f}%")

# ── Step 5: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: CCA vs LDA accuracy comparison
methods    = ['CCA\n(no training)', 'LDA\n(5-fold CV)']
accuracies = [accuracy * 100, lda_scores.mean() * 100]
colors     = ['darkorange', 'steelblue']

bars = axes[0].bar(methods, accuracies, color=colors, width=0.4)
axes[0].axhline(50, color='red', linestyle='--', label='Chance (50%)')
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[0].set_title('SSVEP Classification\nCCA vs LDA')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 100)
axes[0].legend()

# Right: CCA correlation values for one epoch
# Show how CCA distinguishes between frequencies
sample_epoch = data[0]   # first epoch (label 0 = 12 Hz)
corrs_12 = []
corrs_15 = []

for i in range(len(data)):
    ep = data[i].T
    c12 = CCA(n_components=1)
    c12.fit(ep, references[0])
    x1, r1 = c12.transform(ep, references[0])
    corrs_12.append(np.corrcoef(x1[:, 0], r1[:, 0])[0, 1])

    c15 = CCA(n_components=1)
    c15.fit(ep, references[1])
    x2, r2 = c15.transform(ep, references[1])
    corrs_15.append(np.corrcoef(x2[:, 0], r2[:, 0])[0, 1])

corrs_12 = np.array(corrs_12)
corrs_15 = np.array(corrs_15)

axes[1].scatter(corrs_12[labels==0], corrs_15[labels==0],
                color='steelblue', s=80, label='S155 (12 Hz)', zorder=5)
axes[1].scatter(corrs_12[labels==1], corrs_15[labels==1],
                color='darkorange', s=80, label='S255 (15 Hz)', zorder=5)
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Equal correlation line')
axes[1].set_title('CCA Correlations per Epoch\n(above diagonal = predict 12 Hz)')
axes[1].set_xlabel('Correlation with 12 Hz reference')
axes[1].set_ylabel('Correlation with 15 Hz reference')
axes[1].legend()

plt.tight_layout()
plt.savefig('ssvep_cca_results.png', dpi=150)
print("\nSaved: ssvep_cca_results.png")
print("Done!")