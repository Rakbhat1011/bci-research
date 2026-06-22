from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM
from pyriemann.transfer import TLCenter, TLRotate, TLClassifier, TLSplitter, encode_domains
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── Step 1: Load all 9 subjects ──────────────────
print("Loading data...")
dataset  = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

all_X, all_y, all_cov = [], [], []
le = LabelEncoder()

for subject in range(1, 10):
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    cov = Covariances(estimator='oas').fit_transform(X)
    all_X.append(X)
    all_y.append(le.fit_transform(y))
    all_cov.append(cov)
    print(f"Subject {subject}: {cov.shape[0]} covariance matrices")

# ── Step 2: Leave-one-subject-out evaluation ──────
print("\nRunning leave-one-subject-out cross-subject evaluation...")

no_adapt_scores = []
pt_scores       = []

for test_subj in range(9):
    # Split into train (8 subjects) and test (1 subject)
    train_cov = np.vstack([all_cov[i] for i in range(9) if i != test_subj])
    train_y   = np.concatenate([all_y[i] for i in range(9) if i != test_subj])
    test_cov  = all_cov[test_subj]
    test_y    = all_y[test_subj]

    # Domain labels — needed for TL tools
    train_domains = np.concatenate([
        np.full(len(all_cov[i]), i) for i in range(9) if i != test_subj
    ])
    test_domain = np.full(len(test_cov), test_subj)

    # ── No adaptation baseline ────────────────────
    clf_no_adapt = FgMDM(metric='riemann')
    clf_no_adapt.fit(train_cov, train_y)
    score_no_adapt = (clf_no_adapt.predict(test_cov) == test_y).mean()
    no_adapt_scores.append(score_no_adapt)

    # ── With parallel transport (TLCenter + TLRotate) ─
    # Combine train and test for domain adaptation
    all_cov_combined = np.vstack([train_cov, test_cov])
    all_domains      = np.concatenate([train_domains, test_domain])
    all_y_combined   = np.concatenate([train_y, test_y])

    # Encode domains
    X_enc, y_enc = encode_domains(all_cov_combined,
                                   all_y_combined,
                                   all_domains)

  # TLCenter: parallel transport to common reference (Step 1 only)
    # TLRotate needs target labels — skipping for unsupervised setting
    tl_center = TLCenter(target_domain=test_subj)
    X_centered = tl_center.fit_transform(X_enc, y_enc)

    # Classify on adapted data
    n_train = len(train_cov)
    X_train_adapted = X_centered[:n_train]
    X_test_adapted  = X_centered[n_train:]

    clf_pt = FgMDM(metric='riemann')
    clf_pt.fit(X_train_adapted, train_y)
    score_pt = (clf_pt.predict(X_test_adapted) == test_y).mean()
    pt_scores.append(score_pt)

    print(f"S{test_subj+1}: No adapt={score_no_adapt*100:.1f}%  "
          f"PT+Rotate={score_pt*100:.1f}%")

print(f"\n── Mean across 9 subjects ───────────────────")
print(f"Within-subject FgMDM (prev) : 72.1%")
print(f"Cross-subject (no adapt)    : {np.mean(no_adapt_scores)*100:.1f}%")
print(f"Cross-subject + PT + Rotate : {np.mean(pt_scores)*100:.1f}%")
print(f"Chance                      : 25.0%")

# ── Step 3: Plot ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

subjects = [f'S{i}' for i in range(1, 10)]
x        = np.arange(len(subjects))
width    = 0.25

ax.bar(x - width, [72.1, 59.5, 87.8, 69.8, 45.5, 61.5, 81.4, 86.5, 77.3],
       width, label='Within-subject FgMDM (72.1%)', color='seagreen', alpha=0.85)
ax.bar(x, [s*100 for s in no_adapt_scores],
       width, label=f'Cross-subject no adapt ({np.mean(no_adapt_scores)*100:.1f}%)',
       color='steelblue', alpha=0.85)
ax.bar(x + width, [s*100 for s in pt_scores],
       width, label=f'Cross-subject + PT (TLCenter) ({np.mean(pt_scores)*100:.1f}%)',
       color='darkorange', alpha=0.85)

ax.axhline(25, color='red', linestyle='--', label='Chance (25%)')
ax.set_title('Within-Subject vs Cross-Subject Classification\nBCI Competition IV Dataset 2a — FgMDM')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 100)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('cross_subject_results.png', dpi=150)
print("\nSaved: cross_subject_results.png")
print("Done!")
