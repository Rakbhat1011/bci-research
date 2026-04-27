import scipy.io as sio
import numpy as np
import os
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import matplotlib.pyplot as plt

# ── Step 1: Load files grouped by participant ────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"
participants = defaultdict(list)

for filename in sorted(os.listdir(data_path)):
    if filename.endswith(".mat"):
        parts = filename.split("_")
        pn_id = [p for p in parts if p.startswith("PN")][0].split("-")[0]
        participants[pn_id].append(filename)

print(f"Found {len(participants)} participants\n")

# ── Step 2: Run both DBSCAN and HDBSCAN ──────────
results = []

for pn_id in sorted(participants.keys()):
    files = participants[pn_id]

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

    # Scale + PCA
    X_scaled = StandardScaler().fit_transform(X)
    X_pca    = PCA(n_components=10).fit_transform(X_scaled)

    # ── HDBSCAN ──
    hdb        = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    hdb_labels = hdb.fit_predict(X_pca)
    hdb_ari    = adjusted_rand_score(y, hdb_labels)
    hdb_n      = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    # ── DBSCAN ──
    # eps = neighbourhood radius, min_samples = min points to form cluster
    db        = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X_pca)
    db_ari    = adjusted_rand_score(y, db_labels)
    db_n      = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    # Best seizure cluster % for each
    def best_seizure_pct(labels, y):
        best = 0
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            if mask.sum() > 0:
                pct = y[mask].sum() / mask.sum() * 100
                best = max(best, pct)
        return best

    hdb_pct = best_seizure_pct(hdb_labels, y)
    db_pct  = best_seizure_pct(db_labels, y)

    results.append({
        'id': pn_id,
        'hdb_ari': hdb_ari, 'hdb_n': hdb_n, 'hdb_pct': hdb_pct,
        'db_ari':  db_ari,  'db_n':  db_n,  'db_pct':  db_pct,
    })

    print(f"{pn_id} | "
          f"HDBSCAN: ARI={hdb_ari:.3f} clusters={hdb_n} seizure%={hdb_pct:.0f}% | "
          f"DBSCAN:  ARI={db_ari:.3f} clusters={db_n} seizure%={db_pct:.0f}%")

# ── Step 3: Comparison plots ─────────────────────
ids      = [r['id'] for r in results]
hdb_aris = [r['hdb_ari'] for r in results]
db_aris  = [r['db_ari'] for r in results]
hdb_pcts = [r['hdb_pct'] for r in results]
db_pcts  = [r['db_pct'] for r in results]

x     = np.arange(len(ids))
width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ARI comparison
axes[0].bar(x - width/2, hdb_aris, width, label='HDBSCAN', color='steelblue')
axes[0].bar(x + width/2, db_aris,  width, label='DBSCAN',  color='darkorange')
axes[0].axhline(0.1, color='red', linestyle='--', label='ARI=0.1 threshold')
axes[0].set_title("ARI Score — DBSCAN vs HDBSCAN per Participant")
axes[0].set_ylabel("ARI Score")
axes[0].set_xticks(x)
axes[0].set_xticklabels(ids, rotation=45)
axes[0].legend()

# Seizure cluster % comparison
axes[1].bar(x - width/2, hdb_pcts, width, label='HDBSCAN', color='steelblue')
axes[1].bar(x + width/2, db_pcts,  width, label='DBSCAN',  color='darkorange')
axes[1].axhline(50, color='red', linestyle='--', label='50% threshold')
axes[1].set_title("Best Seizure Cluster % — DBSCAN vs HDBSCAN per Participant")
axes[1].set_ylabel("% Seizure in best cluster")
axes[1].set_xticks(x)
axes[1].set_xticklabels(ids, rotation=45)
axes[1].legend()

plt.tight_layout()
plt.savefig("dbscan_vs_hdbscan.png", dpi=150)
print("\nSaved: dbscan_vs_hdbscan.png")
print("Done!")