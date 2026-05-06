import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from funkcija_6_1 import generate_data

X = generate_data(500, 1)

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, n_init=10)
    oznaka = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=oznaka, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    plt.title(f"K = {k}")
    plt.show()
