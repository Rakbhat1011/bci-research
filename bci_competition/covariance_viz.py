from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pyriemann.classification import MDM


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
from mne.time_frequency import psd_array_multitaper

# ── Step 1: Load Subject 1 ────────────────────────
print("Loading data...")
dataset  = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])

print(f"EEG data shape : {X.shape}")
print(f"Labels         : {set(y)}")
print(f"Trials per class: {[(c, sum(y==c)) for c in set(y)]}")

# ── Step 2: Compute covariance matrices ───────────
cov = Covariances(estimator='oas')
C   = cov.fit_transform(X)

print(f"\nCovariance matrices shape: {C.shape}")
print(f"One matrix shape         : {C[0].shape}")
print(f"Is symmetric?            : {np.allclose(C[0], C[0].T)}")
print(f"Min eigenvalue           : {np.linalg.eigvalsh(C[0]).min():.6f}")

# ── Step 3: Visualise two covariance matrices ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pick one trial from left hand and one from right hand
left_idx  = np.where(y == 'left_hand')[0][0]
right_idx = np.where(y == 'right_hand')[0][0]

im1 = axes[0].imshow(C[left_idx],  cmap='RdBu_r', aspect='auto')
axes[0].set_title('Covariance Matrix — Left Hand\n(one trial, 22x22)')
axes[0].set_xlabel('Channel index')
axes[0].set_ylabel('Channel index')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(C[right_idx], cmap='RdBu_r', aspect='auto')
axes[1].set_title('Covariance Matrix — Right Hand\n(one trial, 22x22)')
axes[1].set_xlabel('Channel index')
plt.colorbar(im2, ax=axes[1])

plt.suptitle('EEG Covariance Matrices — BCI Competition IV Dataset 2a',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('covariance_matrices.png', dpi=150)
print("\nSaved: covariance_matrices.png")
print("Done!")

# ── Step 4: Encode labels ─────────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {le.classes_}")
print(f"Encoded: {set(y_enc)}")

# ── Step 5: MDM classifier ────────────────────────
# Pipeline: compute covariance → classify with MDM
mdm_pipe = Pipeline([
    ('cov', Covariances(estimator='oas')),
    ('mdm', MDM(metric='riemann'))
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(mdm_pipe, X, y_enc, cv=cv, scoring='accuracy')

print(f"\n── MDM Results — Subject 1 ──────────────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in scores]}")
print(f"Mean accuracy  : {scores.mean()*100:.1f}%")

chance = 1.0 / len(le.classes_)
print(f"Chance level   : {chance*100:.1f}%")


# ── Step 6: FgMDM ─────────────────────────────────
fgmdm_pipe = Pipeline([
    ('cov',   Covariances(estimator='oas')),
    ('fgmdm', FgMDM(metric='riemann'))
])
fgmdm_scores = cross_val_score(fgmdm_pipe, X, y_enc,
                               cv=cv, scoring='accuracy')
print(f"\n── FgMDM Results — Subject 1 ────────────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in fgmdm_scores]}")
print(f"Mean accuracy  : {fgmdm_scores.mean()*100:.1f}%")

# ── Step 7: Tangent Space + LDA ───────────────────
tangent_pipe = Pipeline([
    ('cov',     Covariances(estimator='oas')),
    ('tangent', TangentSpace(metric='riemann')),
    ('lda',     LDA())
])
tangent_scores = cross_val_score(tangent_pipe, X, y_enc,
                                 cv=cv, scoring='accuracy')
print(f"\n── Tangent Space + LDA — Subject 1 ─────────")
print(f"Fold accuracies: {[f'{s*100:.1f}%' for s in tangent_scores]}")
print(f"Mean accuracy  : {tangent_scores.mean()*100:.1f}%")

# ── Step 8: Band power LDA baseline ───────────────
psds, freqs = psd_array_multitaper(X, sfreq=250,
                                    fmin=8, fmax=30,
                                    verbose=False)
mu_idx   = np.where((freqs >= 8)  & (freqs <= 12))[0]
beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
X_feat   = np.hstack([psds[:, :, mu_idx].mean(axis=2),
                      psds[:, :, beta_idx].mean(axis=2)])

from sklearn.model_selection import cross_val_score as cvs
lda_pipe   = Pipeline([('scaler', StandardScaler()), ('lda', LDA())])
lda_scores = cvs(lda_pipe, X_feat, y_enc, cv=cv, scoring='accuracy')

print(f"\n── LDA (band power) — Subject 1 ────────────")
print(f"Mean accuracy  : {lda_scores.mean()*100:.1f}%")

# ── Step 9: Summary + Plot ────────────────────────
print(f"\n── Summary — Subject 1 ─────────────────────")
print(f"LDA (band power)  : {lda_scores.mean()*100:.1f}%")
print(f"MDM               : {scores.mean()*100:.1f}%")
print(f"FgMDM             : {fgmdm_scores.mean()*100:.1f}%")
print(f"Tangent Space+LDA : {tangent_scores.mean()*100:.1f}%")
print(f"Chance            : {chance*100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 5))

methods    = ['LDA\n(band power)', 'MDM', 'FgMDM', 'Tangent\nSpace+LDA']
accuracies = [lda_scores.mean()*100, scores.mean()*100,
              fgmdm_scores.mean()*100, tangent_scores.mean()*100]
colors     = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.85, width=0.5)
ax.axhline(chance*100, color='red', linestyle='--',
           label=f'Chance ({chance*100:.1f}%)')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center',
            fontsize=12, fontweight='bold')

ax.set_title('Riemannian Methods vs LDA — Subject 1\nBCI Competition IV Dataset 2a')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.savefig('riemannian_subject1.png', dpi=150)
print("\nSaved: riemannian_subject1.png")
print("Done!")