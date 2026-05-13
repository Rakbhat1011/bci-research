import numpy as np
import matplotlib.pyplot as plt

# ── Settings ─────────────────────────────────────
sfreq     = 256          # sampling rate Hz
duration  = 5            # seconds
t         = np.arange(0, duration, 1/sfreq)  # time axis
noise_amp = 0.5          # noise level

# ── Four SSVEP stimuli frequencies ───────────────
freqs = {
    'Left  ( 8 Hz)':  8,
    'Right (10 Hz)': 10,
    'Up    (12 Hz)': 12,
    'Down  (15 Hz)': 15
}

# ── Simulate EEG response ─────────────────────────
# When looking at freq X, brain produces a sinusoid at X Hz
# plus random noise
np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for idx, (label, freq) in enumerate(freqs.items()):
    # Clean SSVEP signal at target frequency
    signal = np.sin(2 * np.pi * freq * t)

    # Add noise
    noisy  = signal + noise_amp * np.random.randn(len(t))

    # Compute FFT to find frequency content
    fft_vals = np.abs(np.fft.rfft(noisy))
    fft_freq = np.fft.rfftfreq(len(t), 1/sfreq)

    # Plot
    axes[idx].plot(fft_freq, fft_vals, color='steelblue', linewidth=1)
    axes[idx].axvline(freq, color='red', linestyle='--',
                      linewidth=2, label=f'Target: {freq} Hz')
    axes[idx].set_xlim(0, 40)
    axes[idx].set_title(f'SSVEP Response — {label}')
    axes[idx].set_xlabel('Frequency (Hz)')
    axes[idx].set_ylabel('Power')
    axes[idx].legend()

plt.suptitle('SSVEP Simulation — Brain Mirrors Stimulus Frequency',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ssvep_simulation.png', dpi=150)
print('Saved: ssvep_simulation.png')
print('Done!')