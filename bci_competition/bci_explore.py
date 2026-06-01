from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# ── Step 2: Extract band power features ──────────
print("\nExtracting features...")

def extract_band_power(X, sfreq=250):
    from mne.time_frequency import psd_array_multitaper
    # Compute PSD and extract Mu (8-12Hz) and Beta (13-30Hz) power
    psds, freqs = psd_array_multitaper(X, sfreq=sfreq,
                                        fmin=8, fmax=30,
                                        verbose=False)
    # Mu band: 8-12 Hz
    mu_idx   = np.where((freqs >= 8)  & (freqs <= 12))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
    mu_power   = psds[:, :, mu_idx].mean(axis=2)    # (n_epochs, n_channels)
    beta_power = psds[:, :, beta_idx].mean(axis=2)
    return np.hstack([mu_power, beta_power])         # (n_epochs, n_channels*2)

le = LabelEncoder()

subject_results = []

for i, (X, y) in enumerate(zip(all_X, all_y)):
    subject = i + 1
    X_feat  = extract_band_power(X)
    y_enc   = le.fit_transform(y)

    # ── Classifiers ──────────────────────────────
    classifiers = {
        'LDA'          : Pipeline([('scaler', StandardScaler()), ('clf', LDA())]),
        'SVM (linear)' : Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', C=1.0))]),
        'SVM (RBF)'    : Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0))]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}

    for name, clf in classifiers.items():
        s = cross_val_score(clf, X_feat, y_enc, cv=cv, scoring='accuracy')
        scores[name] = s.mean()

    subject_results.append(scores)
    print(f"S{subject}: LDA={scores['LDA']*100:.1f}% | SVM-L={scores['SVM (linear)']*100:.1f}% | SVM-RBF={scores['SVM (RBF)']*100:.1f}% | RF={scores['Random Forest']*100:.1f}%")

# ── Step 3: Summary table ─────────────────────────
print("\n── Mean accuracy across all 9 subjects ──────")
clf_names = list(classifiers.keys())
for name in clf_names:
    mean_acc = np.mean([r[name] for r in subject_results])
    print(f"{name:20s}: {mean_acc*100:.1f}%")
print(f"{'Chance':20s}: 25.0%  (4 classes)")

# ── Step 4: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

subjects   = [f'S{i}' for i in range(1, 10)]
x          = np.arange(len(subjects))
width      = 0.2
colors     = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

# Per subject bar chart
for j, name in enumerate(clf_names):
    accs = [subject_results[i][name]*100 for i in range(9)]
    axes[0].bar(x + j*width, accs, width,
                label=name, color=colors[j], alpha=0.85)

axes[0].axhline(25, color='red', linestyle='--', label='Chance (25%)')
axes[0].set_title('4-Class Motor Imagery Benchmark\nBCI Competition IV Dataset 2a — Per Subject')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(subjects)
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=8)

# Mean accuracy comparison
mean_accs = [np.mean([r[name] for r in subject_results])*100
             for name in clf_names]
bars = axes[1].bar(clf_names, mean_accs, color=colors, alpha=0.85)
axes[1].axhline(25, color='red', linestyle='--', label='Chance (25%)')
for bar, acc in zip(bars, mean_accs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Mean Accuracy Across 9 Subjects\nBCI Competition IV Dataset 2a')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('bci_benchmark_results.png', dpi=150)
print("\nSaved: bci_benchmark_results.png")
print("Done!")