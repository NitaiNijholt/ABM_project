import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def align_clusters(reference, target):
    ref_means = reference.groupby('cluster').mean()
    tgt_means = target.groupby('cluster').mean()

    all_columns = ref_means.columns.union(tgt_means.columns)
    ref_means = ref_means.reindex(columns=all_columns, fill_value=0)
    tgt_means = tgt_means.reindex(columns=all_columns, fill_value=0)

    distances = cdist(ref_means, tgt_means, metric='euclidean')

    row_ind, col_ind = linear_sum_assignment(distances)

    cluster_mapping = {tgt_cluster: ref_cluster for tgt_cluster, ref_cluster in zip(col_ind, row_ind)}

    target['aligned_cluster'] = target['cluster'].map(cluster_mapping)
    
    return target

def perform_pairwise_tukey(df, feature_columns, cluster_column):
    pairwise_results = []
    for feature in feature_columns:
        df_feature = df[[feature, cluster_column]].dropna()
        if df_feature[cluster_column].nunique() < 2:
            continue
        tukey = pairwise_tukeyhsd(endog=df_feature[feature], groups=df_feature[cluster_column], alpha=0.05)
        results_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        results_df['feature'] = feature
        pairwise_results.append(results_df)
    return pd.concat(pairwise_results, ignore_index=True)

def display_tukey_results(tukey_results):
    tukey_results = tukey_results.rename(columns={'group1': 'Group 1', 'group2': 'Group 2'})
    tukey_results['Group 1'] = tukey_results['Group 1'].astype(int)
    tukey_results['Group 2'] = tukey_results['Group 2'].astype(int)

    print(tukey_results)
    
    fig, ax = plt.subplots(figsize=(18, 12))  # Increase figure size
    ax.axis('off')
    table = ax.table(cellText=tukey_results.values, colLabels=tukey_results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)
    plt.tight_layout()
    plt.show()

def aggregate_feature_importances(directory):
    all_files = glob.glob(os.path.join(directory, "combination_*", "feature_analysis_run_*.csv"))

    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename)
        expected_columns = ['buy', 'sell', 'start_building', 'continue_building', 'gather', 'wealth_diff', 'income', 'cluster']
        available_columns = [col for col in expected_columns if col in df.columns]
        df = df[available_columns]

        if 'cluster' not in df.columns:
            raise ValueError("The 'cluster' column is missing from the dataset.")
        
        dfs.append(df)

    reference_df = dfs[0].copy()
    reference_df['aligned_cluster'] = reference_df['cluster']

    aligned_dfs = [reference_df]
    for df in dfs[1:]:
        aligned_df = align_clusters(reference_df, df)
        aligned_dfs.append(aligned_df)

    combined_df = pd.concat(aligned_dfs, ignore_index=True)

    feature_columns = [col for col in available_columns if col != 'cluster']
    tukey_results = perform_pairwise_tukey(combined_df, feature_columns, 'aligned_cluster')
    tukey_file_path = os.path.join(directory, 'pairwise_tukey_results.csv')
    tukey_results.to_csv(tukey_file_path)
    print(f"Pairwise Tukey HSD results saved to {tukey_file_path}")

    display_tukey_results(tukey_results)

    grouped = combined_df.groupby('aligned_cluster').agg(['mean', 'median', 'std'])

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    aggregated_file_path = os.path.join(directory, 'aggregated_feature_importances.csv')
    grouped.to_csv(aggregated_file_path)
    print(f"Aggregated feature importances saved to {aggregated_file_path}")

    colors = {0: 'blue', 1: 'green', 2: 'orange'}
    for stat in ['mean', 'median']:
        fig, axs = plt.subplots(len(grouped.index), 1, figsize=(15, 12), constrained_layout=True)
        for i, cluster in enumerate(grouped.index):
            data = grouped.loc[cluster, [col for col in grouped.columns if col.endswith(stat) and col != 'cluster_mean']]
            if stat == 'mean':
                std_data = grouped.loc[cluster, [col for col in grouped.columns if col.endswith('std') and col != 'cluster_std']]
                error_bars = std_data.values
                ax = axs[i]
                data.plot(kind='bar', ax=ax, yerr=error_bars, capsize=5, color=colors.get(cluster, 'black'))
            else:
                ax = axs[i]
                data.plot(kind='bar', ax=ax, color=colors.get(cluster, 'black'))
            ax.set_title(f'Feature Importance ({stat.capitalize()}) for Cluster {cluster}', fontsize=24)
            if i != len(grouped.index) - 1:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Features', fontsize=24)
            ax.set_ylabel(f'{stat.capitalize()} Value', fontsize=24)
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45, labelsize=24)
            ax.tick_params(axis='y', labelsize=24)
        plt.tight_layout(pad=3.0)
        plt.show()

# Directory where the feature analysis CSV files are saved
directory = 'sensitivity_analysis_results/New_final_exp_evolve_dynamic'
aggregate_feature_importances(directory)
