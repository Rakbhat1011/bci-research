import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import Xdawn
import mne

data_dir = '/Users/rakshajagadish/Documents/Projects/BCI_Research/p300_data/'

all_scores_lda   = []
all_scores_xdawn = []

for i in range(1, 9):
    filepath = data_dir + f'P300S0{i}.mat'
    print(f'Processing subject {i}...', end=' ')

    # ── Load data ──────────────────────────────
    mat   = sio.loadmat(filepath)
    data  = mat['data']
    X     = data['X'][0,0]   # EEG matrix
    flash = data['flash'][0,0]   # flash events
    Fs    = int(mat['Fs'][0,0])   # sampling rate

    # ── Extract epochs ─────────────────────────
    epoch_len = int(1.0 * Fs)
    epochs, labels = [], []

    for j in range(len(flash)):
        start = flash[j, 0]
        end   = start + epoch_len
        if end < len(X):
            epochs.append(X[start:end, :])
            labels.append(flash[j, 3])

    epochs   = np.array(epochs)   # (n_epochs, 250, 8)
    labels   = np.array(labels) - 1  # 0=non-target, 1=target

    # ── Pipeline 1: LDA only ───────────────────
    n_epochs, n_times, n_channels = epochs.shape
    X_flat = epochs.reshape(n_epochs, n_times * n_channels)

    pipeline_lda = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LDA())
    ])
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_lda, X_flat, labels, cv=cv, scoring='accuracy')
    all_scores_lda.append(scores.mean())
    print(f'LDA={scores.mean()*100:.1f}%', end=' ')

    # ── Pipeline 2: xDAWN + LDA ────────────────
 # ── Pipeline 2: xDAWN + LDA ────────────────
    # Create a minimal MNE Info object for xDAWN
    info = mne.create_info(
        ch_names=[f'ch{k}' for k in range(n_channels)],
        sfreq=Fs, ch_types='eeg'
    )

    # Create MNE EpochsArray from numpy epochs
    epochs_mne = mne.EpochsArray(
        epochs.transpose(0, 2, 1),  # xDAWN needs (n_epochs, n_channels, n_times)
        info, tmin=0, verbose=False
    )

    # Fit xDAWN — finds best spatial filter for P300
    xd = Xdawn(n_components=4)
    xd.fit(epochs_mne)
    epochs_xd = xd.transform(epochs_mne)  # (n_epochs, n_components, n_times)

    # Flatten and classify
    X_xd = epochs_xd.reshape(n_epochs, -1)

    pipeline_xd = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LDA())
    ])
    scores_xd = cross_val_score(pipeline_xd, X_xd, labels, cv=cv, scoring='accuracy')
    all_scores_xdawn.append(scores_xd.mean())
    print(f'xDAWN+LDA={scores_xd.mean()*100:.1f}%', end=' ')

# ── Plot ───────────────────────────────────────
subjects = [f'S{i}' for i in range(1, 9)]
x = np.arange(len(subjects))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, [s*100 for s in all_scores_lda],   width, label='LDA',        color='steelblue')
ax.bar(x + width/2, [s*100 for s in all_scores_xdawn], width, label='xDAWN + LDA', color='darkorange')
ax.set_title('P300 Classification — LDA vs xDAWN+LDA\n(8 Subjects, 5-fold CV)')
ax.axhline(np.mean(all_scores_lda)*100, color='green', linestyle='--',
           label=f'Mean ({np.mean(all_scores_lda)*100:.1f}%)')
ax.axhline(50, color='red', linestyle='--', label='Chance (50%)')
#ax.set_title('P300 Classification Accuracy Across 8 Subjects\n(LDA, 5-fold CV)')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.savefig('p300_all_subjects.png', dpi=150)
print('Saved: p300_all_subjects.png')