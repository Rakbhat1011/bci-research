from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM
from pyriemann.transfer import TLCenter, encode_domains, decode_domains
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm
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

# ── Step 2: Leave-one-subject-out ─────────────────
print("\nRunning leave-one-subject-out cross-subject evaluation...")

no_adapt_scores = []
pt_scores       = []

for test_subj in range(9):
    train_cov = np.vstack([all_cov[i] for i in range(9) if i != test_subj])
    train_y   = np.concatenate([all_y[i] for i in range(9) if i != test_subj])
    test_cov  = all_cov[test_subj]
    test_y    = all_y[test_subj]

    train_domains = np.concatenate([
        np.full(len(all_cov[i]), i) for i in range(9) if i != test_subj
    ])
    test_domain = np.full(len(test_cov), test_subj)

    # No adaptation
    clf_no_adapt = FgMDM(metric='riemann')
    clf_no_adapt.fit(train_cov, train_y)
    score_no_adapt = (clf_no_adapt.predict(test_cov) == test_y).mean()
    no_adapt_scores.append(score_no_adapt)

    # TLCenter only
    all_cov_combined = np.vstack([train_cov, test_cov])
    all_domains      = np.concatenate([train_domains, test_domain])
    all_y_combined   = np.concatenate([train_y, test_y])

    X_enc, y_enc = encode_domains(all_cov_combined, all_y_combined, all_domains)

    tl_center  = TLCenter(target_domain=test_subj)
    X_centered = tl_center.fit_transform(X_enc, y_enc)

    n_train = len(train_cov)
    clf_pt  = FgMDM(metric='riemann')
    clf_pt.fit(X_centered[:n_train], train_y)
    score_pt = (clf_pt.predict(X_centered[n_train:]) == test_y).mean()
    pt_scores.append(score_pt)

    print(f"S{test_subj+1}: No adapt={score_no_adapt*100:.1f}%  "
          f"PT={score_pt*100:.1f}%")

print(f"\n── PT Results ───────────────────────────────")
print(f"Cross-subject no adapt : {np.mean(no_adapt_scores)*100:.1f}%")
print(f"Cross-subject + PT     : {np.mean(pt_scores)*100:.1f}%")

# ── Step 3: Manual PT + Rotation ─────────────────
print("\nRunning semi-supervised experiment (10 labeled target samples)...")

pt_rotate_scores = []
n_calibration    = 10

for test_subj in range(9):
    train_cov = np.vstack([all_cov[i] for i in range(9) if i != test_subj])
    train_y   = np.concatenate([all_y[i] for i in range(9) if i != test_subj])
    test_cov  = all_cov[test_subj]
    test_y    = all_y[test_subj]

    calib_cov = test_cov[:n_calibration]
    eval_cov  = test_cov[n_calibration:]
    eval_y    = test_y[n_calibration:]

    try:
        # Step 1 — Riemannian mean per domain
        mean_train  = mean_riemann(train_cov)
        mean_target = mean_riemann(calib_cov)

        # Step 2 — transport matrix: move test to train space
        M = np.real(sqrtm(mean_train) @ np.linalg.inv(sqrtm(mean_target)))

        # Transport eval covariances to train space
        eval_transported = np.array([M @ c @ M.T for c in eval_cov])

        # Classify in train space
        clf = FgMDM(metric='riemann')
        clf.fit(train_cov, train_y)
        score = (clf.predict(eval_transported) == eval_y).mean()
        pt_rotate_scores.append(score)
        print(f"S{test_subj+1}: Manual PT = {score*100:.1f}%")

    except Exception as e:
        print(f"S{test_subj+1}: Failed — {type(e).__name__}: {e}")
        pt_rotate_scores.append(np.nan)

# ── Step 4: Final summary ─────────────────────────
print(f"\n── Final Summary ────────────────────────────")
print(f"Within-subject FgMDM        : 72.1%")
print(f"Cross-subject no adapt      : {np.mean(no_adapt_scores)*100:.1f}%")
print(f"Cross-subject + PT          : {np.mean(pt_scores)*100:.1f}%")
print(f"Cross-subject + PT + Rotate : {np.nanmean(pt_rotate_scores)*100:.1f}%")
print(f"Chance                      : 25.0%")

# ── Step 5: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

subjects = [f'S{i}' for i in range(1, 10)]
x        = np.arange(len(subjects))
width    = 0.2

axes[0].bar(x - 1.5*width,
            [72.1,59.5,87.8,69.8,45.5,61.5,81.4,86.5,77.3],
            width, label='Within-subject FgMDM (72.1%)',
            color='seagreen', alpha=0.85)
axes[0].bar(x - 0.5*width,
            [s*100 for s in no_adapt_scores],
            width, label=f'No adapt ({np.mean(no_adapt_scores)*100:.1f}%)',
            color='steelblue', alpha=0.85)
axes[0].bar(x + 0.5*width,
            [s*100 for s in pt_scores],
            width, label=f'+ PT ({np.mean(pt_scores)*100:.1f}%)',
            color='darkorange', alpha=0.85)
axes[0].bar(x + 1.5*width,
            [s*100 for s in pt_rotate_scores],
            width, label=f'+ Manual PT ({np.nanmean(pt_rotate_scores)*100:.1f}%)',
            color='mediumpurple', alpha=0.85)
axes[0].axhline(25, color='red', linestyle='--', label='Chance (25%)')
axes[0].set_title('Cross-Subject Domain Adaptation — Per Subject\nBCI Competition IV Dataset 2a')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(subjects)
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=8)

methods = ['Within-subject\nFgMDM', 'Cross-subject\nno adapt',
           'Cross-subject\n+ PT', 'Cross-subject\n+ Manual PT']
means   = [72.1, np.mean(no_adapt_scores)*100,
           np.mean(pt_scores)*100, np.nanmean(pt_rotate_scores)*100]
colors  = ['seagreen', 'steelblue', 'darkorange', 'mediumpurple']

bars = axes[1].bar(methods, means, color=colors, alpha=0.85, width=0.5)
axes[1].axhline(25, color='red', linestyle='--', label='Chance (25%)')
for bar, acc in zip(bars, means):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center',
                 fontsize=11, fontweight='bold')
axes[1].set_title('Mean Accuracy — Within vs Cross-Subject\nBCI Competition IV Dataset 2a')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('cross_subject_results.png', dpi=150)
print("\nSaved: cross_subject_results.png")
print("Done!")