import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mat = sio.loadmat('/Users/rakshajagadish/Documents/Projects/BCI_Research/p300_data/P300S01.mat')
data = mat['data']

X     = data['X'][0,0]
y     = data['y'][0,0]
trial = data['trial'][0,0]
flash = data['flash'][0,0]
Fs = int(mat['Fs'][0,0])

epoch_len = int(1.0 * Fs)  # 1 second
epochs, labels = [], []

for i in range(len(flash)):
    start = flash[i, 0]
    end   = start + epoch_len
    if end < len(X):
        epochs.append(X[start:end, :])
        labels.append(flash[i, 3])

epochs = np.array(epochs)   # (n_epochs, 250, 8)
labels = np.array(labels)

# ── Flatten epochs ────────────────────────────────
n_epochs, n_times, n_channels = epochs.shape
X_flat = epochs.reshape(n_epochs, n_times * n_channels)  

# Convert labels to 0 and 1
y_binary = labels - 1          # 1→0 (non-target), 2→1 (target)

print('Feature matrix:', X_flat.shape)
print('Targets:', y_binary.sum(), '| Non-targets:', (y_binary==0).sum())

# ── Classify ──────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # fill this in
    ('lda', LDA())       # fill this in
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_flat, y_binary, cv=cv, scoring='accuracy')

print('Fold accuracies:', [f'{s*100:.1f}%' for s in scores])
print('Mean accuracy  :', f'{scores.mean()*100:.1f}%')
print('Chance level   : 50.0%')


# ── Plot ──────────────────────────────────────────
times = np.arange(epoch_len) / Fs * 1000  # convert to ms

target_avg    = epochs[y_binary == 1].mean(axis=0)  # (250, 8)
nontarget_avg = epochs[y_binary == 0].mean(axis=0)  # (250, 8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot:
axes[0].plot(times, target_avg[:, 5], color='red', linewidth=2, label='Target')
axes[0].plot(times, nontarget_avg[:, 0], color='steelblue', linewidth=2, label='Non-Target')
axes[0].axvline(0, color='black', linestyle='--', label='Stimulus onset')
axes[0].axvline(300, color='gray', linestyle=':', label='300ms')
axes[0].set_title('Average ERP — Target vs Non-Target\n(Channel 1)')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend()

axes[1].bar(range(1, 6), scores * 100, color='steelblue')
axes[1].axhline(scores.mean() * 100, color='green', linestyle='--', label=f'Mean ({scores.mean()*100:.1f}%)')
axes[1].axhline(50, color='red', linestyle='--', label='Chance (50%)')
axes[1].set_title('P300 Oddball Classification Accuracy\n(5-fold CV, LDA)')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('p300_oddball_results.png', dpi=150)
print('Saved: p300_oddball_results.png')