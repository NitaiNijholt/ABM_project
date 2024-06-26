import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the CSV file
file_path = 'sensitivity_analysis_results/combination_1/run_5.csv'
df = pd.read_csv(file_path)

# Calculate initial and final wealth for each agent to get the wealth differential
wealth_diff = df.groupby('agent_id')['wealth'].agg(['first', 'last']).reset_index()
wealth_diff['wealth_diff'] = wealth_diff['last'] - wealth_diff['first']

# Print minimum, maximum, and maximum of the maximum wealth per agent
min_wealth = df.groupby('agent_id')['wealth'].min()
max_wealth = df.groupby('agent_id')['wealth'].max()
max_of_max_wealth = max_wealth.max()

print("Minimum wealth per agent:\n", min_wealth)
print("Maximum wealth per agent:\n", max_wealth)
print("Maximum of the maximum wealth per agent:\n", max_of_max_wealth)

# Calculate mean income for each agent
mean_income = df.groupby('agent_id')['income'].mean().reset_index()

# Aggregate the counts of actions per agent
aggregated_data = df.groupby('agent_id').agg({
    'action': lambda x: x.value_counts().to_dict()
}).reset_index()

# Convert action counts to separate columns and remove "move" action
action_df = aggregated_data['action'].apply(pd.Series).fillna(0).drop(columns=['move'], errors='ignore')

# Merge wealth differential and mean income with action counts
aggregated_data = pd.concat([aggregated_data.drop(columns=['action']), action_df], axis=1)
aggregated_data = aggregated_data.merge(wealth_diff[['agent_id', 'wealth_diff']], on='agent_id')
aggregated_data = aggregated_data.merge(mean_income[['agent_id', 'income']], on='agent_id')

# Normalize the action features along with wealth differential and income
scaler = StandardScaler()
scaled_features = scaler.fit_transform(aggregated_data.drop(columns=['agent_id']))

# Perform hierarchical clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_full_tree=True)
clusters = hierarchical_clustering.fit_predict(scaled_features)

# Create a dendrogram
plt.figure(figsize=(10, 7))
linked = linkage(scaled_features, 'ward')
dendrogram(linked)
plt.axhline(y=linked[-(2-1), 2], color='r', linestyle='--')  # Draw a line at the height that corresponds to 2 clusters
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Choose the number of clusters (e.g., 3 clusters)
n_clusters = 3
hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
clusters = hierarchical_clustering.fit_predict(scaled_features)

# Add cluster labels to the DataFrame
aggregated_data['cluster'] = clusters

# Calculate mean values of scaled features within each cluster (normalized feature importances)
feature_means = pd.DataFrame(scaled_features, columns=aggregated_data.drop(columns=['agent_id', 'cluster']).columns).groupby(aggregated_data['cluster']).mean()
colors = plt.get_cmap('tab10')

# Plot normalized feature importance for each cluster
fig, axs = plt.subplots(n_clusters, 1, figsize=(12, 8), constrained_layout=True)
for i in range(n_clusters):
    feature_means.loc[i].plot(kind='bar', ax=axs[i], color=colors(i % 10))
    axs[i].set_title(f'Normalized Feature Importance for Cluster {i}')
    axs[i].set_xlabel('Features')
    axs[i].set_ylabel('Mean Standardized Value')
    axs[i].grid(True)
    axs[i].tick_params(axis='x', rotation=45)
plt.tight_layout(pad=3.0)
plt.show()

# Plot wealth differential and mean income distributions by cluster on log scale
fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
for cluster in np.unique(clusters):
    cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
    label = f'Cluster {cluster}'
    color = colors(cluster % 10)  # Ensure we don't exceed colormap range
    if 'wealth_diff' in aggregated_data.columns:
        axs[0].hist(cluster_data['wealth_diff'], bins=20, alpha=0.5, label=label, color=color)
        axs[0].set_yscale('log')
        axs[0].set_title('Distribution of Wealth Differential by Cluster (Log Scale)')
        axs[0].set_xlabel('Wealth Differential')
        axs[0].set_ylabel('Frequency')
    if 'income' in aggregated_data.columns:
        axs[1].hist(cluster_data['income'], bins=20, alpha=0.5, label=label, color=color)
        axs[1].set_yscale('log')
        axs[1].set_title('Distribution of Mean Income by Cluster (Log Scale)')
        axs[1].set_xlabel('Mean Income')
        axs[1].set_ylabel('Frequency')
axs[0].legend()
axs[1].legend()
plt.tight_layout(pad=3.0)
plt.show()
