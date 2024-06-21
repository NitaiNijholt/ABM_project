import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Load the CSV file
file_path = 'sensitivity_analysis_results/combination_1/run_3.csv'
df = pd.read_csv(file_path)

# Aggregate the data
aggregated_data = df.groupby('agent_id').agg({
    # 'wealth': 'mean',
    # 'houses': 'mean',
    # 'wood': 'mean',
    # 'stone': 'mean',
    # 'income': 'mean',
    'action': lambda x: x.value_counts(normalize=True).to_dict()
}).reset_index()

# Convert action proportions to separate columns
action_df = aggregated_data['action'].apply(pd.Series).fillna(0)
aggregated_data = pd.concat([aggregated_data.drop(columns=['action']), action_df], axis=1)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(aggregated_data.drop(columns=['agent_id']))

# Apply PCA
pca = PCA()
pca_features = pca.fit_transform(scaled_features)

# Determine the optimal number of PCA components
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.grid(True)
plt.show()

# Apply PCA with optimal number of components
optimal_pca_components = np.argmax(cumulative_explained_variance >= 0.90) + 1
pca = PCA(n_components=optimal_pca_components)
pca_features = pca.fit_transform(scaled_features)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(pca_features)

# Determine the optimal number of clusters using Silhouette Score
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(tsne_features)
    silhouette_scores.append(silhouette_score(tsne_features, clusters))

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Apply K-Means with the optimal number of clusters
optimal_clusters = np.argmax(silhouette_scores) + 2  # since silhouette_scores starts from 2 clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(tsne_features)

# Add cluster labels to the DataFrame
aggregated_data['cluster'] = clusters

# Plot t-SNE clusters
colors = plt.get_cmap('tab10')
plt.figure(figsize=(10, 7))
for cluster in np.unique(clusters):
    plt.scatter(tsne_features[clusters == cluster, 0], tsne_features[clusters == cluster, 1], 
                c=[colors(cluster)], label=f'Cluster {cluster}', s=50, alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Agent Behavior Clusters')
plt.legend()
plt.show()

# Calculate and plot feature importances
# Transform the cluster centers back to the original feature space
kmeans_on_pca = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters_on_pca = kmeans_on_pca.fit_predict(pca_features)
cluster_centers_pca = kmeans_on_pca.cluster_centers_
cluster_centers_original = scaler.inverse_transform(pca.inverse_transform(cluster_centers_pca))

# Plot feature importances with rotated x-axis labels
fig, axs = plt.subplots(optimal_clusters, 1, figsize=(12, 8), constrained_layout=True)
for i in range(optimal_clusters):
    pd.Series(cluster_centers_original[i], index=aggregated_data.drop(columns=['agent_id', 'cluster']).columns).plot(kind='bar', ax=axs[i], color=colors(i))
    axs[i].set_title(f'Feature Importances for Cluster {i}')
    axs[i].set_xlabel('Features')
    axs[i].set_ylabel('Mean Value')
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)
plt.tight_layout(pad=3.0)
plt.show()

# Plot distribution of metrics by cluster
metrics = ['wealth', 'houses', 'wood', 'stone', 'income']
action_metrics = action_df.columns.tolist()

fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), constrained_layout=True)
for i, metric in enumerate(metrics):
    for cluster in aggregated_data['cluster'].unique():
        cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
        axs[i].hist(cluster_data[metric], bins=20, alpha=0.5, label=f'Cluster {cluster}', color=colors(cluster))
    axs[i].set_title(f'Distribution of {metric.capitalize()} by Cluster')
    axs[i].set_xlabel(metric.capitalize())
    axs[i].set_ylabel('Frequency')
    axs[i].legend()
plt.tight_layout(pad=3.0)
plt.show()

# Plot proportion of actions by cluster
fig, axs = plt.subplots(len(action_metrics), 1, figsize=(10, 3 * len(action_metrics)), constrained_layout=True)
for j, action in enumerate(action_metrics):
    for cluster in aggregated_data['cluster'].unique():
        cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
        axs[j].hist(cluster_data[action], bins=20, alpha=0.5, label=f'Cluster {cluster}', color=colors(cluster))
    axs[j].set_title(f'Proportion of {action.capitalize()} Action by Cluster')
    axs[j].set_xlabel(action.capitalize())
    axs[j].set_ylabel('Frequency')
    axs[j].legend()
plt.tight_layout(pad=3.0)
plt.show()

