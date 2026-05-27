import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ── Step 1: Load motor imagery data ──────────────
print("Loading motor imagery data...")
SUBJECTS = [1, 2, 3, 4, 5]
RUNS     = [4, 8]   # left vs right hand imagery

all_X, all_y = [], []

for subject in SUBJECTS:
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto', verbose=False)
                 for f in eegbci.load_data(subject, RUNS, verbose=False)]
    raw = concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(raw)
    raw.filter(8., 30., verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events,
                        {'Left': event_dict['T1'], 'Right': event_dict['T2']},
                        tmin=0., tmax=4., baseline=None,
                        preload=True, verbose=False)

    X = epochs.get_data()   # (n_trials, n_channels, n_times)
    y = (epochs.events[:, 2] - epochs.events[:, 2].min()).astype(int)

    all_X.append(X)
    all_y.append(y)
    print(f"Subject {subject}: {len(y)} trials")

X = np.vstack(all_X)   # (total_trials, 64, 641)
y = np.concatenate(all_y)

print(f"\nTotal: {X.shape} | Labels: {np.unique(y, return_counts=True)}")

# ── Step 2: Define EEGNet ─────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.5):
        super(EEGNet, self).__init__()

        # Block 1 — Temporal convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — Depthwise spatial convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # Block 3 — Separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate flattened size
        self.flat_size = self._get_flat_size(n_channels, n_times)

        # Classifier
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_times):
        x = torch.zeros(1, 1, n_channels, n_times)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)      # add channel dim: (batch, 1, channels, times)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ── Step 3: Train and evaluate with cross-val ─────
def train_evaluate(X, y, n_splits=5):
    n_channels, n_times = X.shape[1], X.shape[2]
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize per channel
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
        X_test  = scaler.transform(
            X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

        # Convert to tensors
        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.LongTensor(y_train)
        X_te = torch.FloatTensor(X_test)
        y_te = torch.LongTensor(y_test)

        # DataLoader
        loader = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=16, shuffle=True)

        # Model
        model     = EEGNet(n_channels, n_times)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for 50 epochs
        model.train()
        for epoch in range(50):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_te).argmax(dim=1)
            acc   = (preds == y_te).float().mean().item()
        scores.append(acc)
        print(f"  Fold {fold+1}: {acc*100:.1f}%")

    return np.array(scores)

print("\nTraining EEGNet (5-fold CV)...")
scores = train_evaluate(X, y)

print(f"\n── EEGNet Results ───────────────────────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in scores]}")
print(f"Mean accuracy  : {scores.mean()*100:.1f}%")
print(f"Std            : {scores.std()*100:.1f}%")
print(f"Chance level   : 50.0%")
print(f"\nPrevious LDA result: 60.0%")
print(f"Previous SVM result: 56.7%")

# ── Step 4: Plot ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

methods    = ['SVM\n(previous)', 'LDA\n(previous)', 'EEGNet\n(today)']
accuracies = [56.7, 60.0, scores.mean()*100]
colors     = ['#95a5a6', '#95a5a6', 'steelblue']

bars = ax.bar(methods, accuracies, color=colors, width=0.4)
ax.axhline(50, color='red', linestyle='--', label='Chance (50%)')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Motor Imagery — SVM vs LDA vs EEGNet\n(Left vs Right Hand, 5 subjects)')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.savefig('eegnet_results.png', dpi=150)
print("Saved: eegnet_results.png")
print("Done!")