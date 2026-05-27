import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# ── Step 1: Load data ─────────────────────────────
print("Loading motor imagery data...")
SUBJECTS = [1, 2, 3, 4, 5]
RUNS     = [4, 8]

all_X, all_y = [], []

for subject in SUBJECTS:
    raw_files = [read_raw_edf(f, preload=True,
                              stim_channel='auto', verbose=False)
                 for f in eegbci.load_data(subject, RUNS, verbose=False)]
    raw = concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(raw)
    raw.filter(8., 30., verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events,
                        {'Left': event_dict['T1'], 'Right': event_dict['T2']},
                        tmin=0., tmax=4., baseline=None,
                        preload=True, verbose=False)

    # Band power features — same as before
    from mne.time_frequency import tfr_multitaper
    power = tfr_multitaper(epochs, freqs=np.arange(8, 31),
                           n_cycles=2, return_itc=False,
                           average=False, verbose=False)
    mu_power   = power.data[:, :, :5,  :].mean(axis=(2, 3))
    beta_power = power.data[:, :, 5:,  :].mean(axis=(2, 3))
    X = np.hstack([mu_power, beta_power])
    y = (epochs.events[:, 2] - epochs.events[:, 2].min()).astype(int)

    all_X.append(X)
    all_y.append(y)
    print(f"Subject {subject}: {len(y)} trials")

# ── Step 2: Train on subjects 1-4, test on 5 ─────
X_train = np.vstack([all_X[0]] + all_X[2:])
y_train = np.concatenate([all_y[0]] + all_y[2:])
X_test  = all_X[1]
y_test  = all_y[1]
# Normalize
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTraining on subjects 1-4: {len(y_train)} trials")
print(f"Testing on subject 2    : {len(y_test)} trials")

# ── Step 3: Static LDA ────────────────────────────
static_lda = LDA()
static_lda.fit(X_train, y_train)
static_acc = (static_lda.predict(X_test) == y_test).mean()
print(f"\nStatic LDA accuracy on subject 2: {static_acc*100:.1f}%")

# ── Step 4: Adaptive LDA ──────────────────────────
# Start with model trained on subjects 1-4
# After each test trial, update the model with the true label
# Predict BEFORE updating (so we always testing on unseen data)

# We do this by maintaining running class means and covariance
# Simple online update: weighted average of old and new statistics

class AdaptiveLDA:
    def __init__(self, base_model, learning_rate=0.1):
        self.means_    = base_model.means_.copy()      # (2, n_features)
        self.classes_  = base_model.classes_
        self.lr        = learning_rate
        self.n_features = base_model.means_.shape[1]

    def predict(self, x):
        # Predict using current means (nearest centroid)
        dists = [np.linalg.norm(x - m) for m in self.means_]
        return self.classes_[np.argmin(dists)]

    def update(self, x, y):
        # Update class mean for the true class
        class_idx = np.where(self.classes_ == y)[0][0]
        self.means_[class_idx] = ((1 - self.lr) * self.means_[class_idx]
                                  + self.lr * x)

# Run adaptive LDA trial by trial
adaptive_model = AdaptiveLDA(static_lda, learning_rate=0.1)

adaptive_preds  = []
static_preds    = []
adaptive_running = []
static_running   = []

for i, (x, y_true) in enumerate(zip(X_test, y_test)):
    # Predict with both models
    adapt_pred  = adaptive_model.predict(x)
    static_pred = static_lda.predict(x.reshape(1, -1))[0]

    adaptive_preds.append(adapt_pred == y_true)
    static_preds.append(static_pred == y_true)

    # Running accuracy
    adaptive_running.append(np.mean(adaptive_preds))
    static_running.append(np.mean(static_preds))

    # Update adaptive model with true label
    adaptive_model.update(x, y_true)

adaptive_acc = np.mean(adaptive_preds)
print(f"Adaptive LDA accuracy on subject 2: {adaptive_acc*100:.1f}%")
print(f"\nImprovement: {(adaptive_acc - static_acc)*100:+.1f}%")

# ── Step 5: Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: running accuracy over trials
trials = range(1, len(y_test) + 1)
axes[0].plot(trials, [a*100 for a in static_running],
             color='steelblue', linewidth=2, label=f'Static LDA ({static_acc*100:.1f}%)')
axes[0].plot(trials, [a*100 for a in adaptive_running],
             color='darkorange', linewidth=2, label=f'Adaptive LDA ({adaptive_acc*100:.1f}%)')
axes[0].axhline(50, color='red', linestyle='--', label='Chance (50%)')
axes[0].set_title('Running Accuracy on Subject 2\n(Static vs Adaptive LDA)')
axes[0].set_xlabel('Trial number')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 100)
axes[0].legend()

# Right: final comparison bar chart
methods    = ['Static LDA', 'Adaptive LDA']
accuracies = [static_acc*100, adaptive_acc*100]
colors     = ['steelblue', 'darkorange']

bars = axes[1].bar(methods, accuracies, color=colors, width=0.4)
axes[1].axhline(50, color='red', linestyle='--', label='Chance (50%)')
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[1].set_title('Final Accuracy Comparison\nSubject 2 (cross-subject)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig('adaptive_lda_results.png', dpi=150)
print("Saved: adaptive_lda_results.png")
print("Done!")