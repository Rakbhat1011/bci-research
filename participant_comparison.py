import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ── Load all participants ────────────────────────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"
participants = defaultdict(list)

for filename in sorted(os.listdir(data_path)):
    if filename.endswith(".mat"):
        parts = filename.split("_")
        pn_id = [p for p in parts if p.startswith("PN")][0].split("-")[0]
        participants[pn_id].append(filename)

# ── Compute stats per participant ────────────────
stats = []

for pn_id in sorted(participants.keys()):
    p_features, p_labels = [], []

    for filename in participants[pn_id]:
        mat  = sio.loadmat(os.path.join(data_path, filename))
        sout = mat['Sout']
        gt   = mat['gt'].flatten()
        N    = sout.shape[2]
        p_features.append(sout.reshape(19 * 10, N).T)
        p_labels.append(gt)

    X = np.vstack(p_features)
    y = np.concatenate(p_labels)

    seizure_mask = y > 0
    normal_mask  = y == 0

    X_seizure = X[seizure_mask]
    X_normal  = X[normal_mask]

    # Mean band power during seizure vs normal
    mean_seizure = X_seizure.mean()
    mean_normal  = X_normal.mean()
    diff         = mean_seizure - mean_normal

    # Variance during seizure vs normal
    var_seizure = X_seizure.var()
    var_normal  = X_normal.var()

    # Seizure duration (number of seizure samples)
    seizure_duration = seizure_mask.sum()

    # Signal to noise ratio — how different is seizure from normal
    # (difference in means / std of normal)
    snr = abs(diff) / (X_normal.std() + 1e-8)

    stats.append({
        'id': pn_id,
        'mean_seizure': mean_seizure,
        'mean_normal': mean_normal,
        'diff': diff,
        'var_seizure': var_seizure,
        'var_normal': var_normal,
        'duration': seizure_duration,
        'snr': snr,
        'n_sessions': len(participants[pn_id]),
        'total_samples': len(y),
    })

    tag = " ← HDBSCAN worked" if pn_id in ["PN00", "PN17"] else ""
    print(f"{pn_id}{tag}")
    print(f"  Seizure mean: {mean_seizure:.4f}  Normal mean: {mean_normal:.4f}  Diff: {diff:+.4f}")
    print(f"  Seizure var:  {var_seizure:.4f}  Normal var:  {var_normal:.4f}")
    print(f"  Seizure duration: {seizure_duration} samples   SNR: {snr:.4f}")
    print()

# ── Plot comparison ──────────────────────────────
ids       = [s['id'] for s in stats]
diffs     = [s['diff'] for s in stats]
snrs      = [s['snr'] for s in stats]
durations = [s['duration'] for s in stats]

# Color PN00 and PN17 green, others blue
colors = ['green' if s['id'] in ['PN00', 'PN17'] else 'steelblue' for s in stats]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Mean power difference seizure vs normal
axes[0].bar(ids, diffs, color=colors)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title("Mean Band Power Difference (Seizure - Normal)\nGreen = HDBSCAN worked")
axes[0].set_ylabel("Difference")
axes[0].tick_params(axis='x', rotation=45)

# 2. SNR
axes[1].bar(ids, snrs, color=colors)
axes[1].set_title("Signal-to-Noise Ratio (how distinct seizure is from normal)")
axes[1].set_ylabel("SNR")
axes[1].tick_params(axis='x', rotation=45)

# 3. Seizure duration
axes[2].bar(ids, durations, color=colors)
axes[2].set_title("Total Seizure Samples per Participant")
axes[2].set_ylabel("Number of seizure samples")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("participant_comparison.png", dpi=150)
print("Saved: participant_comparison.png")
print("Done!")