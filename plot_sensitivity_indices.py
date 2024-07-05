import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot

def plot_sensitivity_indices(results_file_path, save_directory):
    # Load sensitivity analysis results from JSON
    with open(results_file_path, 'r') as f:
        sensitivity_results = json.load(f)
    
    # Create the directory for saving plots if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plotting the sensitivity indices
    for metric, Si in sensitivity_results.items():
        print(f"Plotting sensitivity indices for {metric}...")
        Si_df = pd.DataFrame(Si)

        # Plotting first-order sensitivity indices
        plt.figure(figsize=(12, 6))
        barplot(Si_df[['names', 'S1', 'S1_conf']])
        plt.title(f'First-order Sobol Sensitivity Indices for {metric}')
        plt.savefig(os.path.join(save_directory, f'first_order_sensitivity_{metric}.png'))
        plt.show()

        # Plotting total-order sensitivity indices
        plt.figure(figsize=(12, 6))
        barplot(Si_df[['names', 'ST', 'ST_conf']])
        plt.title(f'Total-order Sobol Sensitivity Indices for {metric}')
        plt.savefig(os.path.join(save_directory, f'total_order_sensitivity_{metric}.png'))
        plt.show()

if __name__ == "__main__":
    results_file_path = 'sensitivity_analysis_results.json'  # Path to the sensitivity analysis results JSON file
    save_directory = 'sensitivity_analysis_plots'  # Directory to save the plots
    
    plot_sensitivity_indices(results_file_path, save_directory)
