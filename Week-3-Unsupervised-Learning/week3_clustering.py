import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.describe())

# Select clustering features
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method + silhouette scores
inertias = []
silhouette_scores = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

plt.plot(range(2, 9), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

plt.plot(range(2, 9), silhouette_scores, marker="o")
plt.title("Silhouette Scores")
plt.xlabel("Number of clusters")
plt.ylabel("Average silhouette score")
plt.show()

# Final model
k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Silhouette score:",
      silhouette_score(X_scaled, df["Cluster"]))

print(df["Cluster"].value_counts().sort_index())

# Cluster centres in original units
centers = scaler.inverse_transform(kmeans.cluster_centers_)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print(centers_df)

# Cluster profile
profile = df.groupby("Cluster")[features].agg(
    ["count", "mean", "median"]
)

print(profile)

# Visualization
plt.figure(figsize=(8, 6))

for cluster in sorted(df["Cluster"].unique()):
    part = df[df["Cluster"] == cluster]
    plt.scatter(
        part[features[0]],
        part[features[1]],
        label=f"Cluster {cluster}"
    )

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=180,
    label="Centroids"
)

plt.title("Customer Segments from K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.show()
