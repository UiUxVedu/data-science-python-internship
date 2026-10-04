"""
Week 4 - Supervised Learning Model Implementation
Classification using the Breast Cancer Wisconsin Diagnostic dataset.

Run:
    python week4_supervised_learning.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

# 1. Load public dataset
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

X = df.drop(columns=["target"]).copy()
y = df["target"].copy()

print("Dataset shape:", df.shape)
print("\nTarget counts:")
print(y.value_counts())

# 2. Feature engineering
eps = 1e-9
X["radius_perimeter_ratio"] = X["mean radius"] / (X["mean perimeter"] + eps)
X["area_radius_ratio"] = X["mean area"] / (X["mean radius"] + eps)
X["concavity_compactness_ratio"] = (
    X["mean concavity"] / (X["mean compactness"] + eps)
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Modeling pipeline
# Scaling and feature selection are fitted only on training folds.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif, k=20)),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])

# 5. Five-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

print("\nCross-validation results:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results["test_" + metric]
    print(f"{metric:10s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# 6. Train final model
model.fit(X_train, y_train)

# 7. Test-set predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTest-set metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Malignant", "Benign"]
))

# 8. Permutation importance
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc"
)

importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance": perm.importances_mean
}).sort_values("importance", ascending=False)

print("\nTop 10 features by permutation importance:")
print(importance.head(10).to_string(index=False))
