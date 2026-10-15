import numpy as np
import matplotlib.pyplot as plt

# Week 38: Density-Based Spatial Clustering (DBSCAN from Scratch)
# Author: Ing. Arturo Javier Borbon Rojas

print("🌌 Initializing DBSCAN (Density-Based Clustering Engine)...")
print("-" * 65)

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        print(f"⚙️ Scanning spatial density with epsilon={self.eps} and MinPts={self.min_samples}...")
        
        # Labels: 0 = Unvisited, -1 = Noise, 1,2,3... = Cluster IDs
        labels = np.zeros(X.shape[0])
        cluster_id = 0

        for point_idx in range(X.shape[0]):
            # Skip if the point has already been visited
            if labels[point_idx] != 0:
                continue

            # Find all neighbors within the epsilon radius
            neighbors = self._region_query(X, point_idx)

            # If not enough neighbors, label as Noise (might be upgraded to Border later)
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                # Core point found, initialize a new cluster
                cluster_id += 1
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)

        print(f"   ➔ Scanning complete. Found {cluster_id} dense clusters.")
        return labels

    def _region_query(self, X, point_idx):
        """Calculates Euclidean distances using vectorized operations."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """Expands the cluster using a Breadth-First Search (BFS) approach."""
        labels[point_idx] = cluster_id
        i = 0
        
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # If it was previously marked as noise, it is a border point of this cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            # If it is completely unvisited
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                
                # If this new point is also a core point, expand the search area
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + new_neighbors
            
            i += 1

# --- Synthetic Educational Dataset (Non-spherical groups + Anomalies) ---
# Features: [Absences, Socio-emotional Stress Level]
np.random.seed(42)

# Cluster 1: High stress, high absences (At-Risk)
cluster_1 = np.random.normal(loc=[15, 8], scale=[1.5, 1.0], size=(40, 2))
# Cluster 2: Low stress, low absences (Safe/Regular)
cluster_2 = np.random.normal(loc=[3, 3], scale=[1.2, 0.8], size=(50, 2))
# Cluster 3: Moderate absences, low stress (Average)
cluster_3 = np.random.normal(loc=[8, 2], scale=[1.0, 0.5], size=(30, 2))

# Anomalies: Strange edge cases (e.g. 0 absences but massive stress burnout)
anomalies = np.array([[1, 9.5], [20, 1], [18, 9], [0, 8.5], [10, 10]])

X_data = np.vstack((cluster_1, cluster_2, cluster_3, anomalies))

# Apply Custom DBSCAN
# We require at least 5 points within a radius of 2.0 to form a dense region
dbscan = DBSCAN(eps=2.0, min_samples=5)
predictions = dbscan.fit_predict(X_data)

print("-" * 65)
noise_count = np.sum(predictions == -1)
print(f"📊 ISOLATION METRICS: Detected {noise_count} anomalous profiles (Noise).")

# --- Visualization ---
print("🎨 Generating Spatial Density Map...")
plt.figure(figsize=(9, 6))

# Plot clusters
unique_labels = set(predictions)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise/anomalies
        col = [0, 0, 0, 1]
        marker = 'x'
        label = "Anomaly / Noise"
    else:
        marker = 'o'
        label = f"Cluster {int(k)}"

    class_member_mask = (predictions == k)
    xy = X_data[class_member_mask]
    
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                s=80 if k != -1 else 100, edgecolor='k' if k != -1 else None, 
                linewidth=2 if k == -1 else 1, label=label, alpha=0.8)

plt.title("DBSCAN Optimization: Automatic Cluster Discovery & Noise Isolation", fontsize=14, fontweight='bold')
plt.xlabel("Total Absences", fontsize=12)
plt.ylabel("Socio-emotional Stress Level", fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()