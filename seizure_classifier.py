import scipy.io as sio
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ── Step 1: Load all .mat files ──────────────────
data_path = "/Users/rakshajagadish/Documents/Projects/SeizureDataset"

all_features = []
all_labels   = []

for filename in os.listdir(data_path):
    if filename.endswith(".mat"):
        mat  = sio.loadmat(os.path.join(data_path, filename))
        sout = mat['Sout']           # shape: (19, 10, N)
        gt   = mat['gt'].flatten()   # shape: (N,)

        # Flatten 19 channels x 10 bands → 190 features per sample
        N = sout.shape[2]
        features = sout.reshape(19 * 10, N).T  # shape: (N, 190)

        # Binary labels: 0 = normal, 1 = seizure
        labels = (gt > 0).astype(int)

        all_features.append(features)
        all_labels.append(labels)

# Combine everything into one big dataset
X = np.vstack(all_features)
y = np.concatenate(all_labels)

print(f"Total samples : {len(y)}")
print(f"Normal (0)    : {(y==0).sum()}")
print(f"Seizure (1)   : {(y==1).sum()}")
print(f"Features/sample: {X.shape[1]}")


# ── Step 2: Balance the dataset using SMOTE ──────
# SMOTE creates synthetic seizure samples so both
# classes have equal representation
print("\nBalancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"After balancing:")
print(f"  Normal (0)  : {(y_balanced==0).sum()}")
print(f"  Seizure (1) : {(y_balanced==1).sum()}")

# ── Step 3: Split into train and test ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, random_state=42, stratify=y_balanced
)

# ── Step 4: Train Random Forest ──────────────────
print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ── Step 5: Evaluate ─────────────────────────────
y_pred = clf.predict(X_test)

print("\n── Results ──────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Normal", "Seizure"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  True Normal  detected as Normal:  {cm[0][0]}")
print(f"  True Normal  detected as Seizure: {cm[0][1]}")
print(f"  True Seizure detected as Seizure: {cm[1][1]}")
print(f"  True Seizure detected as Normal:  {cm[1][0]}")

# ── Step 6: Feature importance ───────────────────
importances = clf.feature_importances_
plt.figure(figsize=(12, 4))
plt.bar(range(190), importances, color='steelblue')
plt.title("Feature Importance (19 channels × 10 frequency bands)")
plt.xlabel("Feature index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nSaved: feature_importance.png")