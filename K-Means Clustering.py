# K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the Iris dataset
iris_dataset = datasets.load_iris()
iris_dataframe = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

# Find the optimal number of clusters using the elbow method
x_values = iris_dataframe.iloc[:, [0, 1, 2, 3]].values
within_cluster_sum_of_squares = []

for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_values)
    within_cluster_sum_of_squares.append(kmeans.inertia_)

# Plot the elbow method results
plt.plot(range(1, 11), within_cluster_sum_of_squares)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.show()

# Applying K-Means clustering with the optimal number of clusters (3)
optimal_num_clusters = 3
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_assignments = kmeans.fit_predict(x_values)

# Visualize the clusters
cluster_colors = ['red', 'blue', 'green']
cluster_labels = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']

for cluster_idx in range(optimal_num_clusters):
    plt.scatter(x_values[cluster_assignments == cluster_idx, 0], x_values[cluster_assignments == cluster_idx, 1], s=100, c=cluster_colors[cluster_idx], label=cluster_labels[cluster_idx])

# Plot the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')

plt.legend()
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.show()