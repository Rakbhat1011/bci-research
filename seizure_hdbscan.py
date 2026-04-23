import scipy.io as sio
import numpy as np
import os
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, classification_report

# ── Step 1: Load all .mat files ──────────────────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"

all_features = []
all_labels   = []

for filename in os.listdir(data_path):
    if filename.endswith(".mat"):
        mat  = sio.loadmat(os.path.join(data_path, filename))
        sout = mat['Sout']
        gt   = mat['gt'].flatten()

        N        = sout.shape[2]
        features = sout.reshape(19 * 10, N).T  # (N, 190)
        labels   = (gt > 0).astype(int)         # 0=normal, 1=seizure

        all_features.append(features)
        all_labels.append(labels)

X = np.vstack(all_features)
y = np.concatenate(all_labels)

print(f"Total samples : {len(y)}")
print(f"Normal  (0)   : {(y==0).sum()}")
print(f"Seizure (1)   : {(y==1).sum()}")

# ── Step 2: Scale features ───────────────────────
# HDBSCAN works on distances — scaling makes sure
# no single feature dominates just because it's larger
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 3: Reduce dimensions with PCA ──────────
# 190 features is a lot — reduce to 10 key dimensions
# so HDBSCAN can find clusters more effectively
print("Reducing dimensions with PCA...")
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained by 10 components: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── Step 4: Run HDBSCAN ──────────────────────────
print("\nRunning HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,   # minimum samples to form a cluster
    min_samples=5,         # how conservative to be about outliers
    prediction_data=True
)
cluster_labels = clusterer.fit_predict(X_pca)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise    = (cluster_labels == -1).sum()

print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise}")
print(f"Cluster IDs    : {sorted(set(cluster_labels))}")

# ── Step 5: Compare clusters to real labels ──────
# Check if HDBSCAN's clusters match seizure vs normal
print("\n── How clusters map to real labels ──────────")
for cluster_id in sorted(set(cluster_labels)):
    mask          = cluster_labels == cluster_id
    total         = mask.sum()
    seizure_count = y[mask].sum()
    normal_count  = total - seizure_count
    label         = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"  {label}: {total} samples | Normal: {normal_count} | Seizure: {seizure_count}")

# ARI score — how well do clusters match real labels?
# 1.0 = perfect match, 0.0 = random, can be negative
ari = adjusted_rand_score(y, cluster_labels)
print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
print("(1.0 = perfect, 0.0 = random, higher is better)")

# ── Step 6: Visualise with PCA 2D plot ───────────
print("\nGenerating plot...")
pca_2d   = PCA(n_components=2)
X_2d     = pca_2d.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: coloured by HDBSCAN cluster
scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                            c=cluster_labels, cmap='tab10',
                            s=1, alpha=0.5)
axes[0].set_title("HDBSCAN Clusters")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
plt.colorbar(scatter1, ax=axes[0], label="Cluster ID")

# Right: coloured by real label (seizure vs normal)
scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                            c=y, cmap='RdYlGn',
                            s=1, alpha=0.5)
axes[1].set_title("Ground Truth (0=Normal, 1=Seizure)")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
plt.colorbar(scatter2, ax=axes[1], label="True Label")

plt.tight_layout()
plt.savefig("hdbscan_clusters.png", dpi=150)
print("Saved: hdbscan_clusters.png")
print("\nDone!")