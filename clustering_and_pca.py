import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the CSV file
file_path = 'data_saved.csv'
df = pd.read_csv(file_path)

# Aggregate the data
aggregated_data = df.groupby('agent_id').agg({
    'wealth': 'mean',
    'houses': 'mean',
    'wood': 'mean',
    'stone': 'mean',
    'income': 'mean',
    'action': lambda x: x.value_counts(normalize=True).to_dict()
}).reset_index()

# Convert action proportions to separate columns
action_df = aggregated_data['action'].apply(pd.Series).fillna(0)
aggregated_data = pd.concat([aggregated_data.drop(columns=['action']), action_df], axis=1)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(aggregated_data.drop(columns=['agent_id']))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the DataFrame
aggregated_data['cluster'] = clusters

# Define a consistent color palette
colors = {0: 'blue', 1: 'orange'}

# Apply PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Plot PCA clusters with reduced point size and transparency
plt.figure(figsize=(10, 7))
for cluster in np.unique(clusters):
    plt.scatter(pca_features[clusters == cluster, 0], pca_features[clusters == cluster, 1], 
                c=colors[cluster], label=f'Cluster {cluster}', s=50, alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Agent Behavior Clusters')
plt.legend()
plt.show()

# Plot explained variance ratio
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='gray')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

# Calculate and plot feature importances
cluster_centers = kmeans.cluster_centers_
feature_importances = pd.DataFrame(cluster_centers, columns=aggregated_data.drop(columns=['agent_id', 'cluster']).columns)
feature_importances['cluster'] = range(2)

# Plot feature importances with rotated x-axis labels
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
for i in range(2):
    feature_importances.iloc[i, :-1].plot(kind='bar', ax=axs[i], color=colors[i])
    axs[i].set_title(f'Feature Importances for Cluster {i}')
    axs[i].set_xlabel('Features')
    axs[i].set_ylabel('Mean Value')
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)
plt.tight_layout(pad=3.0)
plt.show()

# Plot distribution of metrics by cluster with adjusted bins and transparency
metrics = ['wealth', 'houses', 'wood', 'stone', 'income']
action_metrics = action_df.columns.tolist()

fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), constrained_layout=True)
for i, metric in enumerate(metrics):
    for cluster in aggregated_data['cluster'].unique():
        cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
        axs[i].hist(cluster_data[metric], bins=20, alpha=0.5, label=f'Cluster {cluster}', color=colors[cluster])
    axs[i].set_title(f'Distribution of {metric.capitalize()} by Cluster')
    axs[i].set_xlabel(metric.capitalize())
    axs[i].set_ylabel('Frequency')
    axs[i].legend()
plt.tight_layout(pad=3.0)
plt.show()

# Plot proportion of actions by cluster with adjusted bins and transparency
fig, axs = plt.subplots(len(action_metrics), 1, figsize=(10, 3 * len(action_metrics)), constrained_layout=True)
for j, action in enumerate(action_metrics):
    for cluster in aggregated_data['cluster'].unique():
        cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
        axs[j].hist(cluster_data[action], bins=20, alpha=0.5, label=f'Cluster {cluster}', color=colors[cluster])
    axs[j].set_title(f'Proportion of {action.capitalize()} Action by Cluster')
    axs[j].set_xlabel(action.capitalize())
    axs[j].set_ylabel('Frequency')
    axs[j].legend()
plt.tight_layout(pad=3.0)
plt.show()
