import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── All results collected across the week ─────────
# (from your previous runs)

methods = [
    'Random\nForest',
    'SVM\n(RBF)',
    'MDM',
    'SVM\n(linear)',
    'LDA\n(band power)',
    'Tangent\n+LDA',
    'ShallowConvNet\n(3 subjects)',
    'CSP+LDA',
    'EEGNet',
    'FgMDM',
]

means = [44.8, 45.3, 58.8, 58.1, 59.6, 62.4, 66.7, 68.0, 70.6, 72.1]

colors = [
    '#95a5a6',  # Random Forest — gray
    '#95a5a6',  # SVM RBF — gray
    '#e67e22',  # MDM — orange
    '#95a5a6',  # SVM linear — gray
    '#3498db',  # LDA — blue
    '#9b59b6',  # Tangent+LDA — purple
    '#e74c3c',  # ShallowConvNet — red
    '#f39c12',  # CSP+LDA — yellow
    '#e74c3c',  # EEGNet — red
    '#27ae60',  # FgMDM — green (winner)
]

# Category labels for legend
classical_patch  = mpatches.Patch(color='#3498db', label='Classical ML (band power)')
spatial_patch    = mpatches.Patch(color='#f39c12', label='Spatial filtering (CSP)')
riemannian_patch = mpatches.Patch(color='#27ae60', label='Riemannian geometry')
deep_patch       = mpatches.Patch(color='#e74c3c', label='Deep learning (CNN)')
weak_patch       = mpatches.Patch(color='#95a5a6', label='Classical ML (other)')

fig, ax = plt.subplots(figsize=(16, 7))

bars = ax.barh(methods, means, color=colors, alpha=0.88, height=0.6)

ax.axvline(25, color='red', linestyle='--', linewidth=1.5, label='Chance (25%)')
ax.axvline(np.max(means), color='green', linestyle=':', linewidth=1.5,
           label=f'Best: FgMDM ({np.max(means):.1f}%)')

for bar, acc in zip(bars, means):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 85)
ax.set_xlabel('Mean Accuracy (%) — 5-fold CV', fontsize=12)
ax.set_title('EEG Motor Imagery Benchmark — BCI Competition IV Dataset 2a\n'
             '4 Classes (Left Hand, Right Hand, Feet, Tongue) | 9 Subjects | 576 Trials/Subject',
             fontsize=13, fontweight='bold')

ax.legend(handles=[classical_patch, spatial_patch,
                   riemannian_patch, deep_patch, weak_patch],
          loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('final_benchmark.png', dpi=150, bbox_inches='tight')
print("Saved: final_benchmark.png")
print("Done!")