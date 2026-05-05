import numpy as np
import matplotlib.pyplot as plt

# ── Simulate P300 signal ────────────────────────
# Create a simple synthetic ERP to understand the shape
np.random.seed(42)
sfreq    = 256          # 256 samples per second
n_trials = 50           # 50 trials per condition
tmin, tmax = -0.1, 0.8  # -100ms to 800ms
times = np.arange(tmin, tmax, 1/sfreq)  # time axis

# ── P300 shape: Gaussian bump at 300ms ──────────
def make_p300(times, amplitude=5.0, latency=0.3, width=0.05):
    return amplitude * np.exp(-((times - latency)**2) / (2 * width**2))

# ── Generate target trials (with P300) ──────────
p300_component = make_p300(times, amplitude=6.0)
target_trials  = np.array([
    p300_component + np.random.randn(len(times)) * 2.0
    for _ in range(n_trials)
])

# ── Generate non-target trials (no P300) ────────
nontarget_trials = np.array([
    np.random.randn(len(times)) * 2.0
    for _ in range(n_trials)
])

# ── Average across trials (noise cancels out) ───
target_avg    = target_trials.mean(axis=0)
nontarget_avg = nontarget_trials.mean(axis=0)

# ── Plot ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: single trial vs average
axes[0].plot(times*1000, target_trials[0],
             color='lightcoral', alpha=0.7, label='Single trial (noisy)')
axes[0].plot(times*1000, target_avg,
             color='red', linewidth=2, label=f'Average ({n_trials} trials)')
axes[0].axvline(300, color='gray', linestyle=':', label='300ms')
axes[0].set_title('Single Trial vs Averaged P300')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend()

# Right: target vs non-target average
axes[1].plot(times*1000, target_avg,
             color='red', linewidth=2, label='Target (P300 present)')
axes[1].plot(times*1000, nontarget_avg,
             color='steelblue', linewidth=2, label='Non-Target (no P300)')
axes[1].axvline(0,   color='black', linestyle='--', label='Stimulus onset')
axes[1].axvline(300, color='gray',  linestyle=':',  label='300ms')
axes[1].set_title('Target vs Non-Target Average')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude (µV)')
axes[1].legend()

plt.tight_layout()
plt.savefig('p300_simulation.png', dpi=150)
print('Saved: p300_simulation.png')