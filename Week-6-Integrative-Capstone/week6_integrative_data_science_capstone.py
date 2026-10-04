"""
Week 6 - Integrative Data Science Capstone
End-to-end analysis using the public Breast Cancer Wisconsin
(Diagnostic) dataset.

Pipeline:
1. Data acquisition
2. Cleaning and quality checks
3. Exploratory data analysis
4. Supervised classification
5. Unsupervised clustering
6. Evaluation
7. Recommendations
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    silhouette_score
)

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df = df.rename(columns={"target": "diagnosis"})

# 0 = malignant, 1 = benign
df["diagnosis_label"] = df["diagnosis"].map({
    0: "Malignant",
    1: "Benign"
})

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. DATA QUALITY / CLEANING
# ------------------------------------------------------------
print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

feature_cols = [
    c for c in df.columns
    if c not in ["diagnosis", "diagnosis_label"]
]

X = df[feature_cols].copy()
y = df["diagnosis"].copy()

# ------------------------------------------------------------
# 3. EDA
# ------------------------------------------------------------
print("\nClass distribution:")
print(df["diagnosis_label"].value_counts())

print("\nDescriptive statistics:")
print(X.describe().T[["mean", "std", "min", "max"]])

print("\nFeature correlations with target:")
print(X.corrwith(y).sort_values().head(10))

# ------------------------------------------------------------
# 4. SUPERVISED LEARNING
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif, k=20)),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=SEED
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

print("\nCross-validation:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results["test_" + metric]
    print(
        metric,
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTest performance:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 5. UNSUPERVISED LEARNING
# ------------------------------------------------------------
cluster_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points"
]

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

k_values = range(2, 7)
silhouette_scores = []

for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20
    )
    labels = km.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    silhouette_scores.append(score)
    print(f"k={k}, silhouette={score:.4f}")

best_k = list(k_values)[np.argmax(silhouette_scores)]
print("\nSelected k:", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=SEED,
    n_init=20
)

df["cluster"] = kmeans.fit_predict(X_cluster)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nCluster malignant rates:")
print(
    df.groupby("cluster")["diagnosis"]
      .apply(lambda s: (s == 0).mean())
)

print("\nCluster profiles:")
print(
    df.groupby("cluster")[cluster_features].mean().round(3)
)

# ------------------------------------------------------------
# 6. INTERPRETATION
# ------------------------------------------------------------
print("\nThe supervised model answers:")
print("Which class is predicted for a new observation?")

print("\nThe unsupervised model answers:")
print("Which observations naturally group together based on selected measurements?")

print("\nNext steps:")
print("- Compare additional supervised algorithms.")
print("- Test clustering stability.")
print("- Validate on an independent external dataset.")
print("- Add domain-specific feature engineering.")
