import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load just ONE session — PN00-1 ──────────────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"
filename  = "Features_Bandpower00_PN00-1.h5.mat"

mat  = sio.loadmat(os.path.join(data_path, filename))
sout = mat['Sout']
gt   = mat['gt'].flatten()
N    = sout.shape[2]
X    = sout.reshape(19 * 10, N).T  # (262, 190)
y    = gt

print(f"Samples: {N}")
seizure_idx = np.where(y > 0)[0]
print(f"Seizure at samples: {seizure_idx[0]} to {seizure_idx[-1]}")

# ── Plot 1: Full recording overview ─────────────
mean_power = X.mean(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(mean_power, color='steelblue', linewidth=1, label='Mean Band Power')
axes[0].axvspan(seizure_idx[0], seizure_idx[-1],
                color='red', alpha=0.3, label='Seizure window')
axes[0].set_title("PN00 Session 1 — Mean EEG Band Power Over Time")
axes[0].set_xlabel("Time sample")
axes[0].set_ylabel("Mean band power")
axes[0].legend()

axes[1].plot(y, color='red', linewidth=1.5)
axes[1].set_title("Ground Truth Label (0 = Normal, 1 = Seizure)")
axes[1].set_xlabel("Time sample")
axes[1].set_ylabel("Label value")
axes[1].set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig("pn00_signal_overview.png", dpi=150)
print("Saved: pn00_signal_overview.png")

# ── Plot 2: Zoomed into the single seizure ───────
pad   = 30
start = max(0, seizure_idx[0] - pad)
end   = min(N, seizure_idx[-1] + pad)

X_zoom = X[start:end]
y_zoom = y[start:end]
t_zoom = np.arange(start, end)

# Top 5 most variable features in this window
top5_idx = np.argsort(X_zoom.var(axis=0))[-5:]

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Top: top 5 features
for idx in top5_idx:
    ch   = idx // 10
    band = idx % 10
    axes[0].plot(t_zoom, X_zoom[:, idx],
                 linewidth=1.2, label=f"Ch{ch} Band{band}")
axes[0].axvspan(seizure_idx[0], seizure_idx[-1],
                color='red', alpha=0.15, label='Seizure')
axes[0].set_title("Top 5 Most Variable Features — Zoomed into Seizure")
axes[0].set_ylabel("Band power")
axes[0].legend(fontsize=8)

# Middle: mean band power zoomed
axes[1].plot(t_zoom, mean_power[start:end],
             color='steelblue', linewidth=1.2)
axes[1].axvspan(seizure_idx[0], seizure_idx[-1],
                color='red', alpha=0.15, label='Seizure')
axes[1].set_title("Mean Band Power — Zoomed")
axes[1].set_ylabel("Mean band power")
axes[1].legend(fontsize=8)

# Bottom: label
axes[2].fill_between(t_zoom, y_zoom, color='red', alpha=0.5)
axes[2].set_title("Ground Truth Label")
axes[2].set_xlabel("Time sample")
axes[2].set_ylabel("Label value")
axes[2].set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig("pn00_seizure_zoom.png", dpi=150)
print("Saved: pn00_seizure_zoom.png")
print("\nDone!")