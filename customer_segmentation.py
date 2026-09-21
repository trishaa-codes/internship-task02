"""
Customer Segmentation Project
Run: python customer_segmentation.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_FILE = "customer_segmentation_data.csv"

df = pd.read_csv(DATA_FILE)

features = ['Age', 'AnnualIncome_k', 'PurchaseFrequency', 'AverageOrderValue', 'WebsiteVisitsPerMonth', 'DiscountUsagePct', 'SpendingScore']
X = StandardScaler().fit_transform(df[features])

# Evaluate k using elbow and silhouette methods
inertias = []
silhouettes = []

for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))

best_k = max(range(2, 9), key=lambda k: silhouettes[k-2])
print("Recommended k from silhouette score:", best_k)

# Final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print("\nCluster profile:")
print(df.groupby("Cluster")[features].mean().round(2))

# Main visualization
plt.figure(figsize=(9, 6))
for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(
        subset["AnnualIncome_k"],
        subset["SpendingScore"],
        s=30,
        alpha=0.65,
        label=f"Cluster {cluster}"
    )

plt.title("Customer Segmentation: Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
