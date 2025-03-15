# K-Means Clustering for Customer Segmentation in E-Commerce

![K-Means Clustering Animation](https://upload.wikimedia.org/wikipedia/commons/8/88/K-means_convergence.gif)

## Overview

This repository demonstrates how to use **K-Means Clustering** to segment customers for targeted marketing in the **E-commerce** industry. By analyzing customer data—specifically, annual spend and purchase frequency—we can discover natural groupings that inform personalized marketing strategies.

---

## 🎯 Problem Statement

**Customer Segmentation for Targeted Marketing**

- **Industry:** E-commerce  
- **Objective:** Segment customers into distinct clusters based on spending habits and purchase frequency.  
- **Benefit:** Enables personalized marketing campaigns, improved customer retention, and targeted promotional strategies.


## 🔍 Example & Code

Imagine you have customer data (e.g., annual spend, frequency) and want to segment them into distinct clusters. Below is a sample Python script that generates synthetic customer data, applies K-Means clustering, and visualizes the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate synthetic customer data
np.random.seed(42)
customers = np.vstack([
    np.random.normal([50, 500], [5, 50], size=(50, 2)),
    np.random.normal([70, 800], [5, 50], size=(50, 2)),
    np.random.normal([40, 300], [5, 50], size=(50, 2))
])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customers)

# Plot clusters and centroids
plt.figure(figsize=(8, 5))
plt.scatter(customers[:, 0], customers[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Centroids', marker='X')
plt.xlabel('Annual Spend ($)')
plt.ylabel('Frequency of Purchase')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.show()
