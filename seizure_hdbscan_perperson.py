import scipy.io as sio
import numpy as np
import os
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict

# ── Step 1: Load files grouped by participant ────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"

participants = defaultdict(list)

for filename in sorted(os.listdir(data_path)):
    if filename.endswith(".mat"):
        # Extract participant ID e.g. PN00, PN14
        parts = filename.split("_")
        pn_id = [p for p in parts if p.startswith("PN")][0].split("-")[0]
        participants[pn_id].append(filename)

print(f"Found {len(participants)} participants: {sorted(participants.keys())}")

# ── Step 2: Run HDBSCAN per participant ──────────
results = []

for pn_id in sorted(participants.keys()):
    files = participants[pn_id]

    # Load all sessions for this participant
    p_features, p_labels = [], []
    for filename in files:
        mat  = sio.loadmat(os.path.join(data_path, filename))
        sout = mat['Sout']
        gt   = mat['gt'].flatten()
        N    = sout.shape[2]
        p_features.append(sout.reshape(19 * 10, N).T)
        p_labels.append((gt > 0).astype(int))

    X = np.vstack(p_features)
    y = np.concatenate(p_labels)

    n_seizure = y.sum()
    n_normal  = (y == 0).sum()

    # Scale
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    n_comp = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
    X_pca  = PCA(n_components=n_comp).fit_transform(X_scaled)

    # HDBSCAN
    clusterer     = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    cluster_labels = clusterer.fit_predict(X_pca)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    ari        = adjusted_rand_score(y, cluster_labels)

    # Check if any cluster is seizure-dominant
    best_seizure_pct = 0
    for cid in set(cluster_labels):
        if cid == -1:
            continue
        mask = cluster_labels == cid
        if mask.sum() > 0:
            pct = y[mask].sum() / mask.sum() * 100
            best_seizure_pct = max(best_seizure_pct, pct)

    results.append({
        'id': pn_id,
        'sessions': len(files),
        'samples': len(y),
        'normal': n_normal,
        'seizure': n_seizure,
        'clusters': n_clusters,
        'ari': ari,
        'best_seizure_pct': best_seizure_pct
    })

    print(f"{pn_id} | sessions={len(files)} | samples={len(y):4d} | "
          f"seizure={n_seizure:3d} | clusters={n_clusters} | "
          f"ARI={ari:.3f} | best seizure cluster={best_seizure_pct:.0f}%")

# ── Step 3: Summary plot ─────────────────────────
ids  = [r['id'] for r in results]
aris = [r['ari'] for r in results]
pcts = [r['best_seizure_pct'] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ARI per participant
colors = ['green' if a > 0.1 else 'steelblue' for a in aris]
axes[0].bar(ids, aris, color=colors)
axes[0].axhline(0.1, color='red', linestyle='--', label='ARI=0.1 threshold')
axes[0].set_title("ARI Score per Participant\n(higher = clusters match seizure labels better)")
axes[0].set_xlabel("Participant")
axes[0].set_ylabel("ARI Score")
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

# Best seizure cluster % per participant
colors2 = ['green' if p > 50 else 'steelblue' for p in pcts]
axes[1].bar(ids, pcts, color=colors2)
axes[1].axhline(50, color='red', linestyle='--', label='50% threshold')
axes[1].set_title("Best Seizure Cluster %\n(how pure is the most seizure-focused cluster)")
axes[1].set_xlabel("Participant")
axes[1].set_ylabel("% Seizure in best cluster")
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()

plt.tight_layout()
plt.savefig("hdbscan_perperson.png", dpi=150)
print("\nSaved: hdbscan_perperson.png")
print("\nDone!")