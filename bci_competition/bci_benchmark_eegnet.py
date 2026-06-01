from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# ── Step 1: Load all 9 subjects ──────────────────
print("Loading BCI Competition IV Dataset 2a...")
dataset  = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

all_X, all_y = [], []
for subject in range(1, 10):
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    all_X.append(X)
    all_y.append(y)
    print(f"Subject {subject}: {X.shape[0]} epochs")

# ── Step 2: EEGNet definition ─────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=4,
                 F1=8, D=2, F2=16, dropout=0.5):
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.flat_size  = self._get_flat_size(n_channels, n_times)
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_times):
        x = torch.zeros(1, 1, n_channels, n_times)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ── Step 3: Band power features for LDA ──────────
def extract_band_power(X, sfreq=250):
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(X, sfreq=sfreq,
                                        fmin=8, fmax=30,
                                        verbose=False)
    mu_idx   = np.where((freqs >= 8)  & (freqs <= 12))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
    return np.hstack([psds[:, :, mu_idx].mean(axis=2),
                      psds[:, :, beta_idx].mean(axis=2)])

# ── Step 4: Train EEGNet with cross-val ──────────
def train_eegnet(X, y, n_splits=5, n_epochs=100):
    n_channels, n_times = X.shape[1], X.shape[2]
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
        X_test  = scaler.transform(
            X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.LongTensor(y_train)
        X_te = torch.FloatTensor(X_test)
        y_te = torch.LongTensor(y_test)

        loader    = DataLoader(TensorDataset(X_tr, y_tr),
                               batch_size=32, shuffle=True)
        model     = EEGNet(n_channels, n_times, n_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te).argmax(dim=1)
            acc   = (preds == y_te).float().mean().item()
        scores.append(acc)

    return np.mean(scores)

# ── Step 5: Run benchmark ─────────────────────────
print("\nRunning benchmark (LDA + EEGNet per subject)...")
le = LabelEncoder()

lda_scores    = []
eegnet_scores = []

for i, (X, y) in enumerate(zip(all_X, all_y)):
    subject = i + 1
    y_enc   = le.fit_transform(y)

    # LDA
    X_feat = extract_band_power(X)
    pipeline = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    lda_acc = cross_val_score(pipeline, X_feat, y_enc,
                              cv=cv, scoring='accuracy').mean()
    lda_scores.append(lda_acc)

    # EEGNet
    eegnet_acc = train_eegnet(X, y_enc, n_epochs=100)
    eegnet_scores.append(eegnet_acc)

    print(f"S{subject}: LDA={lda_acc*100:.1f}%  EEGNet={eegnet_acc*100:.1f}%")

print(f"\n── Mean across 9 subjects ───────────────────")
print(f"LDA    : {np.mean(lda_scores)*100:.1f}%")
print(f"EEGNet : {np.mean(eegnet_scores)*100:.1f}%")
print(f"Chance : 25.0%")

# ── Step 6: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

subjects = [f'S{i}' for i in range(1, 10)]
x        = np.arange(len(subjects))
width    = 0.35

axes[0].bar(x - width/2, [s*100 for s in lda_scores],
            width, label=f'LDA ({np.mean(lda_scores)*100:.1f}%)',
            color='steelblue')
axes[0].bar(x + width/2, [s*100 for s in eegnet_scores],
            width, label=f'EEGNet ({np.mean(eegnet_scores)*100:.1f}%)',
            color='darkorange')
axes[0].axhline(25, color='red', linestyle='--', label='Chance (25%)')
axes[0].set_title('LDA vs EEGNet — Per Subject\nBCI Competition IV Dataset 2a')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(subjects)
axes[0].set_ylim(0, 100)
axes[0].legend()

# Right: mean comparison with all classifiers from Day 1
all_methods    = ['LDA', 'SVM\n(linear)', 'SVM\n(RBF)', 'Random\nForest', 'EEGNet']
all_accuracies = [59.6, 58.1, 45.3, 44.8, np.mean(eegnet_scores)*100]
all_colors     = ['steelblue', 'darkorange', 'seagreen',
                  'mediumpurple', 'crimson']

bars = axes[1].bar(all_methods, all_accuracies, color=all_colors, alpha=0.85)
axes[1].axhline(25, color='red', linestyle='--', label='Chance (25%)')
for bar, acc in zip(bars, all_accuracies):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center',
                 fontsize=11, fontweight='bold')
axes[1].set_title('Full Benchmark — All Methods\nBCI Competition IV Dataset 2a')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('bci_benchmark_eegnet.png', dpi=150)
print("\nSaved: bci_benchmark_eegnet.png")
print("Done!")