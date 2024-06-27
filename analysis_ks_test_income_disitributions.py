import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def gini_coefficient(wealths):
    """ Calculate the Gini coefficient of a list of wealths. """
    if len(wealths) == 0:
        return None
    sorted_wealths = np.sort(np.array(wealths))
    index = np.arange(1, len(wealths) + 1)
    n = len(wealths)
    return (np.sum((2 * index - n - 1) * sorted_wealths)) / (n * np.sum(sorted_wealths))

def load_and_compute_metrics(directory):
    """ Load CSV files from the directory and compute metrics for each run. """
    run_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    total_welfare = []
    gini_coefficients = []
    wealth_distributions = []
    
    for file_name in run_files:
        df = pd.read_csv(os.path.join(directory, file_name))
        wealth_per_agent = df.groupby('agent_id')['wealth'].mean()
        total_welfare.append(sum(wealth_per_agent))
        gini_coefficients.append(gini_coefficient(wealth_per_agent.values))
        wealth_distributions.append(wealth_per_agent.values)
        
    return {
        "total_welfare": total_welfare,
        "gini_coefficients": gini_coefficients,
        "wealth_distributions": wealth_distributions
    }

def plot_cdfs(metrics_static, metrics_dynamic):
    """ Plot the CDFs for wealth data from static and dynamic policies. """
    plt.figure(figsize=(12, 8))
    
    for wealth in metrics_static['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='blue', alpha=0.5)
    
    for wealth in metrics_dynamic['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='red', alpha=0.5)
    
    plt.title('CDF of Wealth Distributions by Tax Policy')
    plt.xlabel('Wealth')
    plt.ylabel('CDF')
    plt.legend(['Static Policy', 'Dynamic Policy'])
    plt.grid(True)
    plt.show()

def perform_ks_test_and_plot_table(static_dir, dynamic_dir):
    """ Perform KS tests and plot a statistics table for runs from two directories. """
    metrics_static = load_and_compute_metrics(static_dir)
    metrics_dynamic = load_and_compute_metrics(dynamic_dir)
    
    # Perform the KS test on total welfare and Gini coefficients
    ks_result_welfare = ks_2samp(metrics_static['total_welfare'], metrics_dynamic['total_welfare'])
    ks_result_gini = ks_2samp(metrics_static['gini_coefficients'], metrics_dynamic['gini_coefficients'])
    
    # Prepare the table data
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    table_data = [
        ['Metric', 'Static Policy Min', 'Static Policy Mean', 'Static Policy Max', 'Dynamic Policy Min', 'Dynamic Policy Mean', 'Dynamic Policy Max', 'KS Statistic', 'P-Value'],
        ['Welfare', f'{min(metrics_static["total_welfare"]):.2f}', f'{np.mean(metrics_static["total_welfare"]):.2f}', f'{max(metrics_static["total_welfare"]):.2f}', 
         f'{min(metrics_dynamic["total_welfare"]):.2f}', f'{np.mean(metrics_dynamic["total_welfare"]):.2f}', f'{max(metrics_dynamic["total_welfare"]):.2f}',
         f'{ks_result_welfare.statistic:.4f}', f'{ks_result_welfare.pvalue:.4f}'],
        ['Gini', f'{min(metrics_static["gini_coefficients"]):.4f}', f'{np.mean(metrics_static["gini_coefficients"]):.4f}', f'{max(metrics_static["gini_coefficients"]):.4f}', 
         f'{min(metrics_dynamic["gini_coefficients"]):.4f}', f'{np.mean(metrics_dynamic["gini_coefficients"]):.4f}', f'{max(metrics_dynamic["gini_coefficients"]):.4f}', 
         f'{ks_result_gini.statistic:.4f}', f'{ks_result_gini.pvalue:.4f}']
    ]
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center', colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.show()

    return {"KS Test Welfare": ks_result_welfare, "KS Test Gini": ks_result_gini}

# Example usage:
static_policy_dir = './path_to_static_policy_runs'
dynamic_policy_dir = './path_to_dynamic_policy_runs'

# Plot CDFs
metrics_static = load_and_compute_metrics(static_policy_dir)
metrics_dynamic = load_and_compute_metrics(dynamic_policy_dir)
plot_cdfs(metrics_static, metrics_dynamic)

# Perform KS tests and plot statistics table
ks_test_results = perform_ks_test_and_plot_table(static_policy_dir, dynamic_policy_dir)
print(ks_test_results)
