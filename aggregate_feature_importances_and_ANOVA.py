import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def align_clusters(reference, target):
    """
    Align clusters in the target DataFrame to those in the reference DataFrame.
    """
    ref_means = reference.groupby('cluster').mean()
    tgt_means = target.groupby('cluster').mean()

    # Ensure both dataframes have the same columns by reindexing
    all_columns = ref_means.columns.union(tgt_means.columns)
    ref_means = ref_means.reindex(columns=all_columns, fill_value=0)
    tgt_means = tgt_means.reindex(columns=all_columns, fill_value=0)

    # Compute distances between each pair of clusters
    distances = cdist(ref_means, tgt_means, metric='euclidean')

    # Find the best matching clusters
    row_ind, col_ind = linear_sum_assignment(distances)

    # Create a mapping from target cluster numbers to reference cluster numbers
    cluster_mapping = {tgt_cluster: ref_cluster for tgt_cluster, ref_cluster in zip(col_ind, row_ind)}

    # Reassign clusters in the target DataFrame
    target['aligned_cluster'] = target['cluster'].map(cluster_mapping)
    
    return target

def perform_pairwise_tukey(df, feature_columns, cluster_column):
    """
    Perform pairwise Tukey HSD test for each feature between clusters.
    """
    pairwise_results = []
    for feature in feature_columns:
        tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[cluster_column], alpha=0.05)
        results_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        results_df['feature'] = feature
        pairwise_results.append(results_df)
    return pd.concat(pairwise_results, ignore_index=True)

def display_tukey_results(tukey_results):
    """
    Display the Tukey HSD results in a DataFrame and plot it.
    """
    tukey_results = tukey_results.rename(columns={'group1': 'Group 1', 'group2': 'Group 2'})
    tukey_results['Group 1'] = tukey_results['Group 1'].astype(int)
    tukey_results['Group 2'] = tukey_results['Group 2'].astype(int)

    print(tukey_results)
    
    plt.figure(figsize=(14, 10))
    plt.axis('off')
    table = plt.table(cellText=tukey_results.values, colLabels=tukey_results.columns, cellLoc='center', loc='center', edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()

def aggregate_feature_importances(directory):
    all_files = glob.glob(os.path.join(directory, "combination_*", "feature_analysis_run_*.csv"))

    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename)
        # Ensure the column order is consistent
        df = df[['buy', 'sell', 'start_building', 'continue_building', 'gather', 'wealth_diff', 'income', 'cluster']]
        dfs.append(df)

    # Use the first DataFrame as the reference for aligning clusters
    reference_df = dfs[0].copy()
    reference_df['aligned_cluster'] = reference_df['cluster']

    aligned_dfs = [reference_df]
    for df in dfs[1:]:
        aligned_df = align_clusters(reference_df, df)
        aligned_dfs.append(aligned_df)

    combined_df = pd.concat(aligned_dfs, ignore_index=True)

    # Perform Pairwise Tukey HSD tests
    feature_columns = ['buy', 'sell', 'start_building', 'continue_building', 'gather', 'wealth_diff', 'income', 'cluster']
    tukey_results = perform_pairwise_tukey(combined_df, feature_columns, 'aligned_cluster')
    tukey_file_path = os.path.join(directory, 'pairwise_tukey_results.csv')
    tukey_results.to_csv(tukey_file_path)
    print(f"Pairwise Tukey HSD results saved to {tukey_file_path}")

    # Display Tukey HSD results
    display_tukey_results(tukey_results)

    # Group by aligned cluster and compute summary statistics
    grouped = combined_df.groupby('aligned_cluster').agg(['mean', 'median', 'std'])

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Save the aggregated results
    aggregated_file_path = os.path.join(directory, 'aggregated_feature_importances.csv')
    grouped.to_csv(aggregated_file_path)
    print(f"Aggregated feature importances saved to {aggregated_file_path}")

    # Plot the aggregated feature importances with error bars (std deviation)
    colors = {0: 'blue', 1: 'green', 2: 'orange'}
    for stat in ['mean', 'median']:
        fig, axs = plt.subplots(grouped.index.nunique(), 1, figsize=(12, 8), constrained_layout=True)
        for cluster in grouped.index:
            data = grouped.loc[cluster, [col for col in grouped.columns if col.endswith(stat) and col != 'cluster_mean']]
            if stat == 'mean':
                std_data = grouped.loc[cluster, [col for col in grouped.columns if col.endswith('std') and col != 'cluster_std']]
                error_bars = std_data.values
                ax = axs[cluster]
                data.plot(kind='bar', ax=ax, yerr=error_bars, capsize=5, color=colors.get(cluster, 'black'))
            else:
                ax = axs[cluster]
                data.plot(kind='bar', ax=ax, color=colors.get(cluster, 'black'))
            ax.set_title(f'Feature Importance ({stat.capitalize()}) for Cluster {cluster}')
            ax.set_xlabel('Features')
            ax.set_ylabel(f'{stat.capitalize()} Value')
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(pad=3.0)
        plt.show()

# Directory where the feature analysis CSV files are saved
directory = 'sensitivity_analysis_results/final_exp_ev_static_tax'
aggregate_feature_importances(directory)
