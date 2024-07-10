import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, sem, t

def gini_coefficient(wealths):
    """ Calculate the Gini coefficient of a list of wealths. """
    if len(wealths) == 0:
        return None
    sorted_wealths = np.sort(np.array(wealths))
    index = np.arange(1, len(wealths) + 1)
    n = len(wealths)
    return (np.sum((2 * index - n - 1) * sorted_wealths)) / (n * np.sum(sorted_wealths))

def normalize(data):
    """ Normalize the data using min-max normalization. """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def compute_95_ci(data):
    """ Compute the 95% confidence interval for a given data array. """
    mean = np.mean(data)
    n = len(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + 0.95) / 2, n - 1)
    return mean, mean - h, mean + h

def load_and_compute_metrics(directory):
    """ Load CSV files from the directory and compute metrics for each run. """
    combination_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    total_welfare = []
    gini_coefficients = []
    wealth_distributions = []
    total_productivity = []
    social_equality = []
    social_welfare = []
    
    for combination_dir in combination_dirs:
        combination_path = os.path.join(directory, combination_dir)
        run_files = [f for f in os.listdir(combination_path) if f.startswith('run_') and f.endswith('.csv')]
        
        for file_name in run_files:
            df = pd.read_csv(os.path.join(combination_path, file_name))
            initial_wealth_per_agent = df.groupby('agent_id').first()['wealth']
            final_wealth_per_agent = df.groupby('agent_id').last()['wealth']
            wealth_diff_per_agent = final_wealth_per_agent - initial_wealth_per_agent
            
            normalized_final_wealth = normalize(final_wealth_per_agent)
            normalized_wealth_diff = normalize(wealth_diff_per_agent)
            
            gini = gini_coefficient(final_wealth_per_agent.values)
            prod = np.sum(final_wealth_per_agent.values)
            eq = 1 - (len(final_wealth_per_agent) / (len(final_wealth_per_agent) - 1)) * gini
            swf = eq * prod
            
            total_welfare.append(sum(final_wealth_per_agent))
            gini_coefficients.append(gini)
            wealth_distributions.append(normalized_wealth_diff.values)
            total_productivity.append(prod)
            social_equality.append(eq)
            social_welfare.append(swf)
        
    return {
        "total_welfare": total_welfare,
        "gini_coefficients": gini_coefficients,
        "wealth_distributions": wealth_distributions,
        "total_productivity": total_productivity,
        "social_equality": social_equality,
        "social_welfare": social_welfare
    }

def plot_cdfs(metrics_static, metrics_dynamic):
    """ Plot the CDFs for wealth data from static and dynamic policies. """
    plt.figure(figsize=(15, 12))
    
    for wealth in metrics_static['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='blue', alpha=0.5, label='Static Policy' if wealth is metrics_static['wealth_distributions'][0] else "")
    
    for wealth in metrics_dynamic['wealth_distributions']:
        sorted_wealth = np.sort(wealth)
        plt.plot(sorted_wealth, np.linspace(0, 1, len(sorted_wealth)), color='red', alpha=0.5, label='Dynamic Policy' if wealth is metrics_dynamic['wealth_distributions'][0] else "")
    
    plt.xlabel('Wealth (normalized)', fontsize=26)
    plt.ylabel('CDF', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)
    plt.grid(True)
    plt.show()

def perform_ks_test_and_plot_table(static_dir, dynamic_dir):
    """ Perform KS tests and plot a statistics table for runs from two directories. """
    metrics_static = load_and_compute_metrics(static_dir)
    metrics_dynamic = load_and_compute_metrics(dynamic_dir)
    
    # Perform the KS test on all relevant metrics
    ks_result_welfare = ks_2samp(metrics_static['total_welfare'], metrics_dynamic['total_welfare'])
    ks_result_gini = ks_2samp(metrics_static['gini_coefficients'], metrics_dynamic['gini_coefficients'])
    ks_result_productivity = ks_2samp(metrics_static['total_productivity'], metrics_dynamic['total_productivity'])
    ks_result_equality = ks_2samp(metrics_static['social_equality'], metrics_dynamic['social_equality'])
    ks_result_social_welfare = ks_2samp(metrics_static['social_welfare'], metrics_dynamic['social_welfare'])

    # Compute 95% confidence intervals
    welfare_mean, welfare_ci_low, welfare_ci_high = compute_95_ci(metrics_static['total_welfare'])
    gini_mean, gini_ci_low, gini_ci_high = compute_95_ci(metrics_static['gini_coefficients'])
    productivity_mean, productivity_ci_low, productivity_ci_high = compute_95_ci(metrics_static['total_productivity'])
    equality_mean, equality_ci_low, equality_ci_high = compute_95_ci(metrics_static['social_equality'])
    social_welfare_mean, social_welfare_ci_low, social_welfare_ci_high = compute_95_ci(metrics_static['social_welfare'])

    dynamic_welfare_mean, dynamic_welfare_ci_low, dynamic_welfare_ci_high = compute_95_ci(metrics_dynamic['total_welfare'])
    dynamic_gini_mean, dynamic_gini_ci_low, dynamic_gini_ci_high = compute_95_ci(metrics_dynamic['gini_coefficients'])
    dynamic_productivity_mean, dynamic_productivity_ci_low, dynamic_productivity_ci_high = compute_95_ci(metrics_dynamic['total_productivity'])
    dynamic_equality_mean, dynamic_equality_ci_low, dynamic_equality_ci_high = compute_95_ci(metrics_dynamic['social_equality'])
    dynamic_social_welfare_mean, dynamic_social_welfare_ci_low, dynamic_social_welfare_ci_high = compute_95_ci(metrics_dynamic['social_welfare'])
    
    # Prepare the table data
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis('off')
    table_data = [
        ['Metric', 'Static Policy Mean [95% CI]', 'Dynamic Policy Mean [95% CI]', 'KS Statistic', 'P-Value'],
        ['Gini', f'{gini_mean:.4f} [{gini_ci_low:.4f}, {gini_ci_high:.4f}]', 
         f'{dynamic_gini_mean:.4f} [{dynamic_gini_ci_low:.4f}, {dynamic_gini_ci_high:.4f}]', 
         f'{ks_result_gini.statistic:.4f}', f'{ks_result_gini.pvalue:.4f}'],
        ['Productivity', f'{productivity_mean:.2f} [{productivity_ci_low:.2f}, {productivity_ci_high:.2f}]', 
         f'{dynamic_productivity_mean:.2f} [{dynamic_productivity_ci_low:.2f}, {dynamic_productivity_ci_high:.2f}]', 
         f'{ks_result_productivity.statistic:.4f}', f'{ks_result_productivity.pvalue:.4f}'],
        ['Equality', f'{equality_mean:.4f} [{equality_ci_low:.4f}, {equality_ci_high:.4f}]', 
         f'{dynamic_equality_mean:.4f} [{dynamic_equality_ci_low:.4f}, {dynamic_equality_ci_high:.4f}]', 
         f'{ks_result_equality.statistic:.4f}', f'{ks_result_equality.pvalue:.4f}'],
        ['Social Welfare', f'{social_welfare_mean:.2f} [{social_welfare_ci_low:.2f}, {social_welfare_ci_high:.2f}]', 
         f'{dynamic_social_welfare_mean:.2f} [{dynamic_social_welfare_ci_low:.2f}, {dynamic_social_welfare_ci_high:.2f}]', 
         f'{ks_result_social_welfare.statistic:.4f}', f'{ks_result_social_welfare.pvalue:.4f}']
    ]
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center', colWidths=[0.2, 0.4, 0.4, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.show()

    return {
        "KS Test Gini": ks_result_gini,
        "KS Test Productivity": ks_result_productivity,
        "KS Test Equality": ks_result_equality,
        "KS Test Social Welfare": ks_result_social_welfare
    }

# Example usage:
static_policy_dir = './sensitivity_analysis_results/New_final_exp_ev_static'
dynamic_policy_dir = './sensitivity_analysis_results/New_final_exp_ev_dynamic'

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


def perform_ks_test_and_plot_table(static_dir, dynamic_dir):
    """ Perform KS tests and plot a statistics table for runs from two directories. """
    metrics_static = load_and_compute_metrics(static_dir)
    metrics_dynamic = load_and_compute_metrics(dynamic_dir)
    
    # Perform the KS test on all relevant metrics
    ks_result_welfare = ks_2samp(metrics_static['total_welfare'], metrics_dynamic['total_welfare'])
    ks_result_gini = ks_2samp(metrics_static['gini_coefficients'], metrics_dynamic['gini_coefficients'])
    ks_result_productivity = ks_2samp(metrics_static['total_productivity'], metrics_dynamic['total_productivity'])
    ks_result_equality = ks_2samp(metrics_static['social_equality'], metrics_dynamic['social_equality'])
    ks_result_social_welfare = ks_2samp(metrics_static['social_welfare'], metrics_dynamic['social_welfare'])

    # Compute 95% confidence intervals
    welfare_mean, welfare_ci_low, welfare_ci_high = compute_95_ci(metrics_static['total_welfare'])
    gini_mean, gini_ci_low, gini_ci_high = compute_95_ci(metrics_static['gini_coefficients'])
    productivity_mean, productivity_ci_low, productivity_ci_high = compute_95_ci(metrics_static['total_productivity'])
    equality_mean, equality_ci_low, equality_ci_high = compute_95_ci(metrics_static['social_equality'])
    social_welfare_mean, social_welfare_ci_low, social_welfare_ci_high = compute_95_ci(metrics_static['social_welfare'])

    dynamic_welfare_mean, dynamic_welfare_ci_low, dynamic_welfare_ci_high = compute_95_ci(metrics_dynamic['total_welfare'])
    dynamic_gini_mean, dynamic_gini_ci_low, dynamic_gini_ci_high = compute_95_ci(metrics_dynamic['gini_coefficients'])
    dynamic_productivity_mean, dynamic_productivity_ci_low, dynamic_productivity_ci_high = compute_95_ci(metrics_dynamic['total_productivity'])
    dynamic_equality_mean, dynamic_equality_ci_low, dynamic_equality_ci_high = compute_95_ci(metrics_dynamic['social_equality'])
    dynamic_social_welfare_mean, dynamic_social_welfare_ci_low, dynamic_social_welfare_ci_high = compute_95_ci(metrics_dynamic['social_welfare'])
    
    # Prepare the table data
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis('off')
    table_data = [
        ['Metric', 'Static Policy Mean [95% CI]', 'Dynamic Policy Mean [95% CI]', 'KS Statistic', 'P-Value'],
        ['Gini', f'{gini_mean:.4f} [{gini_ci_low:.4f}, {gini_ci_high:.4f}]', 
         f'{dynamic_gini_mean:.4f} [{dynamic_gini_ci_low:.4f}, {dynamic_gini_ci_high:.4f}]', 
         f'{ks_result_gini.statistic:.4f}', f'{ks_result_gini.pvalue:.4f}'],
        ['Productivity', f'{productivity_mean:.2f} [{productivity_ci_low:.2f}, {productivity_ci_high:.2f}]', 
         f'{dynamic_productivity_mean:.2f} [{dynamic_productivity_ci_low:.2f}, {dynamic_productivity_ci_high:.2f}]', 
         f'{ks_result_productivity.statistic:.4f}', f'{ks_result_productivity.pvalue:.4f}'],
        ['Equality', f'{equality_mean:.4f} [{equality_ci_low:.4f}, {equality_ci_high:.4f}]', 
         f'{dynamic_equality_mean:.4f} [{dynamic_equality_ci_low:.4f}, {dynamic_equality_ci_high:.4f}]', 
         f'{ks_result_equality.statistic:.4f}', f'{ks_result_equality.pvalue:.4f}'],
        ['Social Welfare', f'{social_welfare_mean:.2f} [{social_welfare_ci_low:.2f}, {social_welfare_ci_high:.2f}]', 
         f'{dynamic_social_welfare_mean:.2f} [{dynamic_social_welfare_ci_low:.2f}, {dynamic_social_welfare_ci_high:.2f}]', 
         f'{ks_result_social_welfare.statistic:.4f}', f'{ks_result_social_welfare.pvalue:.4f}']
    ]
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center', colWidths=[0.2, 0.4, 0.4, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.show()

    return {
        "KS Test Gini": ks_result_gini,
        "KS Test Productivity": ks_result_productivity,
        "KS Test Equality": ks_result_equality,
        "KS Test Social Welfare": ks_result_social_welfare
    }

# Example usage:
static_policy_dir = './sensitivity_analysis_results/New_final_exp_ev_static'
dynamic_policy_dir = './sensitivity_analysis_results/New_final_exp_ev_dynamic'

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
