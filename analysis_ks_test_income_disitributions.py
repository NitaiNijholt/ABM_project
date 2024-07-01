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
    combination_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    total_welfare = []
    gini_coefficients = []
    wealth_distributions = []
    total_productivity = []
    
    for combination_dir in combination_dirs:
        combination_path = os.path.join(directory, combination_dir)
        run_files = [f for f in os.listdir(combination_path) if f.startswith('run_') and f.endswith('.csv')]
        
        for file_name in run_files:
            df = pd.read_csv(os.path.join(combination_path, file_name))
            initial_wealth_per_agent = df.groupby('agent_id').first()['wealth']
            final_wealth_per_agent = df.groupby('agent_id').last()['wealth']
            wealth_diff_per_agent = final_wealth_per_agent - initial_wealth_per_agent
            
            total_welfare.append(sum(final_wealth_per_agent))
            gini_coefficients.append(gini_coefficient(wealth_diff_per_agent.values))
            wealth_distributions.append(wealth_diff_per_agent.values)
            total_productivity.append(np.sum(final_wealth_per_agent.values))
        
    return {
        "total_welfare": total_welfare,
        "gini_coefficients": gini_coefficients,
        "wealth_distributions": wealth_distributions,
        "total_productivity": total_productivity
    }

def plot_cdfs(metrics_static, metrics_dynamic):
    """ Plot the CDFs for wealth data from static and dynamic policies. """
    plt.figure(figsize=(12, 8))
    
    for wealth in metrics_static['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='blue', alpha=0.5, label='Static Policy' if wealth is metrics_static['wealth_distributions'][0] else "")
    
    for wealth in metrics_dynamic['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='red', alpha=0.5, label='Dynamic Policy' if wealth is metrics_dynamic['wealth_distributions'][0] else "")
    
    plt.title('CDF of Wealth Distributions by Tax Policy')
    plt.xlabel('Wealth')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_ks_test_and_plot_table(static_dir, dynamic_dir):
    """ Perform KS tests and plot a statistics table for runs from two directories. """
    metrics_static = load_and_compute_metrics(static_dir)
    metrics_dynamic = load_and_compute_metrics(dynamic_dir)
    
    # Perform the KS test on total welfare, Gini coefficients, and productivity
    ks_result_welfare = ks_2samp(metrics_static['total_welfare'], metrics_dynamic['total_welfare'])
    ks_result_gini = ks_2samp(metrics_static['gini_coefficients'], metrics_dynamic['gini_coefficients'])
    ks_result_productivity = ks_2samp(metrics_static['total_productivity'], metrics_dynamic['total_productivity'])
    
    # Prepare the table data
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    table_data = [
        ['Metric', 'Static Policy Mean ± 1SD', 'Dynamic Policy Mean ± 1SD', 'KS Statistic', 'P-Value'],
        ['Welfare', f'{np.mean(metrics_static["total_welfare"]):.2f} ± {np.std(metrics_static["total_welfare"]):.2f}', 
         f'{np.mean(metrics_dynamic["total_welfare"]):.2f} ± {np.std(metrics_dynamic["total_welfare"]):.2f}', 
         f'{ks_result_welfare.statistic:.4f}', f'{ks_result_welfare.pvalue:.4f}'],
        ['Gini', f'{np.mean(metrics_static["gini_coefficients"]):.4f} ± {np.std(metrics_static["gini_coefficients"]):.4f}', 
         f'{np.mean(metrics_dynamic["gini_coefficients"]):.4f} ± {np.std(metrics_dynamic["gini_coefficients"]):.4f}', 
         f'{ks_result_gini.statistic:.4f}', f'{ks_result_gini.pvalue:.4f}'],
        ['Productivity', f'{np.mean(metrics_static["total_productivity"]):.2f} ± {np.std(metrics_static["total_productivity"]):.2f}', 
         f'{np.mean(metrics_dynamic["total_productivity"]):.2f} ± {np.std(metrics_dynamic["total_productivity"]):.2f}', 
         f'{ks_result_productivity.statistic:.4f}', f'{ks_result_productivity.pvalue:.4f}']
    ]
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center', colWidths=[0.2, 0.3, 0.3, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.show()

    return {
        "KS Test Welfare": ks_result_welfare, 
        "KS Test Gini": ks_result_gini,
        "KS Test Productivity": ks_result_productivity
    }

# Example usage:
static_policy_dir = './sensitivity_analysis_results/evolve_static'
dynamic_policy_dir = './sensitivity_analysis_results/evolve_dynamic'

# Plot CDFs
metrics_static = load_and_compute_metrics(static_policy_dir)
metrics_dynamic = load_and_compute_metrics(dynamic_policy_dir)
plot_cdfs(metrics_static, metrics_dynamic)

# Perform KS tests and plot statistics table
ks_test_results = perform_ks_test_and_plot_table(static_policy_dir, dynamic_policy_dir)
print(ks_test_results)

# Print Total Productivity for both policies
print(f"Total Productivity for Static Policy: {metrics_static['total_productivity']}")
print(f"Total Productivity for Dynamic Policy: {metrics_dynamic['total_productivity']}")
