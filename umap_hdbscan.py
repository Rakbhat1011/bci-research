import scipy.io as sio
import numpy as np
import os
import hdbscan
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict

# ── Load all participants ────────────────────────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"
participants = defaultdict(list)

for filename in sorted(os.listdir(data_path)):
    if filename.endswith(".mat"):
        parts = filename.split("_")
        pn_id = [p for p in parts if p.startswith("PN")][0].split("-")[0]
        participants[pn_id].append(filename)

# ── Helper: run HDBSCAN with PCA vs UMAP ────────
from sklearn.decomposition import PCA

def run_pipeline(X, y, reducer, name):
    X_scaled  = StandardScaler().fit_transform(X)
    X_reduced = reducer.fit_transform(X_scaled)
    labels = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X_reduced)
    ari       = adjusted_rand_score(y, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return ari, n_clusters, X_reduced, labels

# ── Run per participant ──────────────────────────
results = []

print("Running PCA + HDBSCAN vs UMAP + HDBSCAN per participant...\n")

for pn_id in sorted(participants.keys()):
    p_features, p_labels = [], []

    for filename in participants[pn_id]:
        mat  = sio.loadmat(os.path.join(data_path, filename))
        sout = mat['Sout']
        gt   = mat['gt'].flatten()
        N    = sout.shape[2]
        p_features.append(sout.reshape(19 * 10, N).T)
        p_labels.append((gt > 0).astype(int))

    X = np.vstack(p_features)
    y = np.concatenate(p_labels)

    # PCA — 10 components
    pca_reducer  = PCA(n_components=10)
    ari_pca, n_pca, _, _ = run_pipeline(X, y, pca_reducer, "PCA")

    # UMAP — 10 components
    # n_neighbors: how many nearby points to consider (local structure)
    # min_dist: how tightly to pack points in reduced space
    umap_reducer = umap.UMAP(n_components=10, n_neighbors=30,
                          min_dist=0.5, random_state=42)
    ari_umap, n_umap, X_umap, umap_labels = run_pipeline(X, y, umap_reducer, "UMAP")

    results.append({
        'id': pn_id, 'y': y,
        'ari_pca': ari_pca, 'n_pca': n_pca,
        'ari_umap': ari_umap, 'n_umap': n_umap,
        'X_umap': X_umap, 'umap_labels': umap_labels
    })

    improved = " ✅ UMAP better!" if ari_umap > ari_pca + 0.05 else ""
    tag      = " ← worked before" if pn_id in ["PN00", "PN17"] else ""
    print(f"{pn_id}{tag}{improved}")
    print(f"  PCA  + HDBSCAN: ARI={ari_pca:.3f}  clusters={n_pca}")
    print(f"  UMAP + HDBSCAN: ARI={ari_umap:.3f}  clusters={n_umap}")
    print()

# ── Plot 1: ARI comparison bar chart ────────────
ids       = [r['id'] for r in results]
ari_pcas  = [r['ari_pca'] for r in results]
ari_umaps = [r['ari_umap'] for r in results]

x     = np.arange(len(ids))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width/2, ari_pcas,  width, label='PCA + HDBSCAN',  color='steelblue')
ax.bar(x + width/2, ari_umaps, width, label='UMAP + HDBSCAN', color='darkorange')
ax.axhline(0.1, color='red', linestyle='--', label='ARI=0.1 threshold')
ax.set_title("ARI Score — PCA vs UMAP before HDBSCAN (per participant)")
ax.set_ylabel("ARI Score")
ax.set_xticks(x)
ax.set_xticklabels(ids, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("umap_vs_pca_ari.png", dpi=150)
print("Saved: umap_vs_pca_ari.png")

# ── Plot 2: UMAP 2D scatter for PN00 and PN17 ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pn_id in zip(axes, ["PN00", "PN17"]):
    r = next(r for r in results if r['id'] == pn_id)

    # Re-run UMAP in 2D just for visualisation
    X_all = np.vstack([sio.loadmat(os.path.join(data_path, f))['Sout'].reshape(190, -1).T
                       for f in participants[pn_id]])
    y_all = np.concatenate([(sio.loadmat(os.path.join(data_path, f))['gt'].flatten() > 0).astype(int)
                             for f in participants[pn_id]])

    X_scaled = StandardScaler().fit_transform(X_all)
    X_2d     = umap.UMAP(n_components=2, n_neighbors=15,
                         min_dist=0.1, random_state=42).fit_transform(X_scaled)

    # Plot normal vs seizure
    ax.scatter(X_2d[y_all==0, 0], X_2d[y_all==0, 1],
               c='steelblue', s=3, alpha=0.5, label='Normal')
    ax.scatter(X_2d[y_all==1, 0], X_2d[y_all==1, 1],
               c='red', s=30, alpha=0.9, label='Seizure', zorder=5)
    ax.set_title(f"{pn_id} — UMAP 2D (red = seizure)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()

plt.tight_layout()
plt.savefig("umap_2d_pn00_pn17.png", dpi=150)
print("Saved: umap_2d_pn00_pn17.png")
print("\nDone!")