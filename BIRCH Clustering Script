from sklearn.cluster import Birch
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd

# Instantiate the imputer
imputer = SimpleImputer(strategy='median')

# Load data
data = pd.read_csv('/Users/Nate/Desktop/School/Spring 2024/Capstone II/rice_data.csv')

# Select data features
X = data.select_dtypes(include=['float64', 'int64']).values

# Fit and transform data
X = imputer.fit_transform(X)

# Create and fit Birch clustering model
birch = Birch(threshold=0.5, n_clusters=None)
birch.fit(X)

# Get the unique cluster centers
unique_cluster_centers = birch.subcluster_centers_

# Get the number of clusters
num_clusters = len(unique_cluster_centers)

print("Number of clusters:", num_clusters)

# Predict cluster labels
labels = birch.predict(X)

# Retrieve cluster centers
cluster_centers = birch.subcluster_centers_

# Retrieve the number of data points in each cluster
cluster_sizes = birch.subcluster_labels_.shape[0]

# Retrieve the hierarchical structure of the clusters
cluster_hierarchy = birch.root_

# Retrieve memory usage information
memory_usage = birch.memory_usage_

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')

# Add legend
handles = []
for cluster_label in set(birch.labels_):
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(cluster_label), markersize=10, label=f'Cluster {cluster_label}'))

plt.legend(handles=handles, loc='best', title='Clusters')

plt.title('Birch Clustering')
plt.show()
