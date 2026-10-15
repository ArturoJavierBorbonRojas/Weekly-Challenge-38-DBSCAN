# Weekly Challenge 38: Unsupervised Machine Learning (DBSCAN from Scratch)

## Description
This week, I continued exploring Unsupervised Learning and Anomaly Detection by implementing **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** strictly using NumPy.

Algorithms like K-Means are powerful, but they have a fatal flaw: they assume data is spherical and force you to pre-define the number of clusters ($K$). In real-world educational data analysis, student profiles form complex, irregular shapes, and we often have severe outliers (e.g., a student with perfect attendance but critical stress levels). DBSCAN solves this by crawling through the data spatially, discovering dense continuous regions of any shape, and mathematically isolating atypical points as "Noise."

## How it works
DBSCAN relies on two mathematical hyperparameters:
* **Epsilon ($\epsilon$):** The maximum distance between two points for them to be considered neighbors.
* **MinPts:** The minimum number of points required within the Epsilon radius to form a dense region (a cluster).

The algorithm categorizes points into three types using Euclidean distances:
1. **Core Points:** Have at least *MinPts* neighbors within their $\epsilon$-radius.
2. **Border Points:** Do not have enough neighbors themselves, but fall within the $\epsilon$-radius of a Core Point.
3. **Noise (Anomalies):** Points that do not belong to any cluster and are entirely isolated.

I implemented the spatial neighborhood expansion using a Breadth-First Search (BFS) algorithm to iteratively traverse the distance matrices.

## Technical Highlights
* **Pure Vectorization:** Distance queries between points are calculated using `np.linalg.norm` for optimal performance.
* **Dynamic Discovery:** The algorithm successfully identified the exact number of natural clusters without prior definition and isolated synthetic edge-cases natively.

## Dependencies
* Python 3.12
* NumPy
* Matplotlib
