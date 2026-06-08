from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, FgMDM
from pyriemann.tangentspace import TangentSpace

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

# ── Step 2: Run benchmark ─────────────────────────
print("\nRunning Riemannian benchmark...")
le = LabelEncoder()

mdm_scores      = []
fgmdm_scores    = []
tangent_scores  = []
lda_scores      = []

for i, (X, y) in enumerate(zip(all_X, all_y)):
    subject = i + 1
    y_enc   = le.fit_transform(y)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Band power + LDA (baseline) ───────────────
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(X, sfreq=250,
                                        fmin=8, fmax=30,
                                        verbose=False)
    mu_idx   = np.where((freqs >= 8)  & (freqs <= 12))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
    X_feat   = np.hstack([psds[:, :, mu_idx].mean(axis=2),
                          psds[:, :, beta_idx].mean(axis=2)])
    lda_pipe = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
    lda_acc  = cross_val_score(lda_pipe, X_feat, y_enc,
                               cv=cv, scoring='accuracy').mean()

    # ── MDM — Minimum Distance to Mean ────────────
    # Simplest Riemannian classifier
    # Computes covariance matrix per epoch
    # Classifies by distance to class mean on manifold
    mdm_pipe = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('mdm', MDM(metric='riemann'))
    ])
    mdm_acc = cross_val_score(mdm_pipe, X, y_enc,
                              cv=cv, scoring='accuracy').mean()

    # ── FgMDM — Geodesic filtering + MDM ──────────
    # More powerful — filters noise before classifying
    fgmdm_pipe = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('fgmdm', FgMDM(metric='riemann'))
    ])
    fgmdm_acc = cross_val_score(fgmdm_pipe, X, y_enc,
                                cv=cv, scoring='accuracy').mean()

    # ── Tangent Space + LDA ────────────────────────
    # Maps covariance matrices to a flat tangent space
    # Then applies standard LDA on the flat features
    tangent_pipe = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('tangent', TangentSpace(metric='riemann')),
        ('lda', LDA())
    ])
    tangent_acc = cross_val_score(tangent_pipe, X, y_enc,
                                  cv=cv, scoring='accuracy').mean()

    lda_scores.append(lda_acc)
    mdm_scores.append(mdm_acc)
    fgmdm_scores.append(fgmdm_acc)
    tangent_scores.append(tangent_acc)

    print(f"S{subject}: LDA={lda_acc*100:.1f}%  "
          f"MDM={mdm_acc*100:.1f}%  "
          f"FgMDM={fgmdm_acc*100:.1f}%  "
          f"Tangent+LDA={tangent_acc*100:.1f}%")

print(f"\n── Mean across 9 subjects ───────────────────")
print(f"LDA (band power)  : {np.mean(lda_scores)*100:.1f}%")
print(f"MDM               : {np.mean(mdm_scores)*100:.1f}%")
print(f"FgMDM             : {np.mean(fgmdm_scores)*100:.1f}%")
print(f"Tangent Space+LDA : {np.mean(tangent_scores)*100:.1f}%")
print(f"Chance            : 25.0%")

# ── Step 3: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

subjects = [f'S{i}' for i in range(1, 10)]
x        = np.arange(len(subjects))
width    = 0.2

axes[0].bar(x - 1.5*width, [s*100 for s in lda_scores],
            width, label=f'LDA ({np.mean(lda_scores)*100:.1f}%)',
            color='steelblue')
axes[0].bar(x - 0.5*width, [s*100 for s in mdm_scores],
            width, label=f'MDM ({np.mean(mdm_scores)*100:.1f}%)',
            color='darkorange')
axes[0].bar(x + 0.5*width, [s*100 for s in fgmdm_scores],
            width, label=f'FgMDM ({np.mean(fgmdm_scores)*100:.1f}%)',
            color='seagreen')
axes[0].bar(x + 1.5*width, [s*100 for s in tangent_scores],
            width, label=f'Tangent+LDA ({np.mean(tangent_scores)*100:.1f}%)',
            color='mediumpurple')
axes[0].axhline(25, color='red', linestyle='--', label='Chance (25%)')
axes[0].set_title('Riemannian Methods vs LDA — Per Subject\nBCI Competition IV Dataset 2a')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(subjects)
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=8)

# Right: mean comparison including previous results
all_methods    = ['LDA', 'MDM', 'FgMDM', 'Tangent\n+LDA', 'EEGNet\n(prev)', 'Shallow\n(prev)']
all_accuracies = [np.mean(lda_scores)*100, np.mean(mdm_scores)*100,
                  np.mean(fgmdm_scores)*100, np.mean(tangent_scores)*100,
                  70.6, 66.7]
all_colors     = ['steelblue', 'darkorange', 'seagreen',
                  'mediumpurple', 'crimson', 'chocolate']

bars = axes[1].bar(all_methods, all_accuracies,
                   color=all_colors, alpha=0.85)
axes[1].axhline(25, color='red', linestyle='--', label='Chance (25%)')
for bar, acc in zip(bars, all_accuracies):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center',
                 fontsize=10, fontweight='bold')
axes[1].set_title('All Methods Comparison\nBCI Competition IV Dataset 2a')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('riemannian_results.png', dpi=150)
print("\nSaved: riemannian_results.png")
print("Done!")