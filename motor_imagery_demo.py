import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ── Settings ─────────────────────────────────────
SUBJECTS = [1, 2, 3, 4, 5]   # 5 subjects
RUNS = [4, 8]           # imagery runs

# ── Helper: process one subject ──────────────────
def process_subject(subject):
    # Load data
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto', verbose=False)
                 for f in eegbci.load_data(subject, RUNS, verbose=False)]
    raw = concatenate_raws(raw_files)
    mne.datasets.eegbci.standardize(raw)  # standardize channel names

    # Bandpass filter 8-30 Hz
    raw.filter(8., 30., fir_design='firwin', verbose=False,
               skip_by_annotation='edge')

    # Extract epochs
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    imagery_events     = {'Left': event_dict['T1'], 'Right': event_dict['T2']}
    epochs = mne.Epochs(raw, events, imagery_events,
                        tmin=0., tmax=4., baseline=None,
                        preload=True, verbose=False)

    # Extract band power features
    power = tfr_multitaper(epochs, freqs=np.arange(8, 31),
                           n_cycles=2, return_itc=False,
                           average=False, verbose=False)

    mu_power   = power.data[:, :, :5,  :].mean(axis=(2, 3))
    beta_power = power.data[:, :, 5:,  :].mean(axis=(2, 3))
    X = np.hstack([mu_power, beta_power])
    y = epochs.events[:, 2] - epochs.events[:, 2].min()

    return X, y, epochs, mu_power

# ── Run across all subjects ───────────────────────
print(f"Processing {len(SUBJECTS)} subjects...\n")

all_scores     = []
all_mu_left    = []
all_mu_right   = []
channel_names  = None
sample_epochs  = None

for subject in SUBJECTS:
    print(f"Subject {subject}...", end=" ")
    X, y, epochs, mu_power = process_subject(subject)

    # Cross validation
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA())
        ])

    # Option 2: SVM
    # from sklearn.svm import SVC
    # pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, random_state=42))])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    all_scores.append(scores.mean())
    all_mu_left.append(mu_power[y==0].mean(axis=0))
    all_mu_right.append(mu_power[y==1].mean(axis=0))

    if channel_names is None:
        channel_names = epochs.ch_names
        sample_epochs = epochs

    print(f"accuracy = {scores.mean()*100:.1f}%")

all_scores  = np.array(all_scores)
mu_left     = np.array(all_mu_left).mean(axis=0)   # avg across subjects
mu_right    = np.array(all_mu_right).mean(axis=0)
diff_power  = mu_right - mu_left

print(f"\n── Overall Results ──────────────────────")
print(f"Per subject accuracy: {[f'{s*100:.1f}%' for s in all_scores]}")
print(f"Mean accuracy : {all_scores.mean()*100:.1f}%")
print(f"Std           : {all_scores.std()*100:.1f}%")
print(f"Chance level  : 50.0%")

# ── Plot 1: Accuracy per subject ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['green' if s > 0.6 else 'steelblue' for s in all_scores]
axes[0].bar([f"S{s}" for s in SUBJECTS], all_scores * 100, color=colors)
axes[0].axhline(50, color='red', linestyle='--', label='Chance (50%)')
axes[0].axhline(all_scores.mean() * 100, color='green',
                linestyle='--', label=f'Mean ({all_scores.mean()*100:.1f}%)')
axes[0].set_title("Motor Imagery Classification Accuracy\nLeft vs Right Hand (SVM, 5-fold CV)")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 100)
axes[0].legend()

# Mu power difference per channel
axes[1].bar(range(len(diff_power)), diff_power, color='darkorange')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title("Mu Band Power Difference\n(Right - Left hand, averaged across subjects)")
axes[1].set_xlabel("Channel index")
axes[1].set_ylabel("Power difference")

plt.tight_layout()
plt.savefig("motor_imagery_results_LDA.png", dpi=150)  # Plot 1
print("Saved: motor_imagery_results_LDA.png")

# ── Plot 2: Channel importance bar chart ─────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Top 10 channels with highest power difference
top10_idx   = np.argsort(np.abs(diff_power))[-10:]
top10_names = [channel_names[i] for i in top10_idx]
top10_vals  = diff_power[top10_idx]
colors      = ['darkorange' if v > 0 else 'steelblue' for v in top10_vals]

axes2[0].barh(top10_names, top10_vals, color=colors)
axes2[0].axvline(0, color='black', linewidth=0.8)
axes2[0].set_title("Top 10 Most Discriminative Channels\nMu Power (Right - Left hand)")
axes2[0].set_xlabel("Power difference\n(orange = Right higher, blue = Left higher)")

# Left vs Right mean Mu power across all channels
axes2[1].plot(mu_left,  color='steelblue',  linewidth=1, label='Left hand',  alpha=0.8)
axes2[1].plot(mu_right, color='darkorange', linewidth=1, label='Right hand', alpha=0.8)
axes2[1].set_title("Mean Mu Band Power per Channel\nLeft vs Right Hand (averaged across subjects)")
axes2[1].set_xlabel("Channel index")
axes2[1].set_ylabel("Power")
axes2[1].legend()

plt.tight_layout()
plt.savefig("motor_imagery_channels_LDA.png", dpi=150)  # Plot 2
print("Saved: motor_imagery_channels_LDA.png")
print("\nDone!")