import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot

class SensitivityPlotter:
    def __init__(self, results_directory):
        self.results_directory = results_directory
        self.results_file_path = os.path.join(self.results_directory, 'sensitivity_analysis_results.json')
        
        if not os.path.exists(self.results_file_path):
            raise FileNotFoundError(f"Sensitivity results file not found in {self.results_directory}")

        self.load_results()

    def load_results(self):
        with open(self.results_file_path, 'r') as f:
            self.sensitivity_results = json.load(f)
            # Verify the structure of the JSON
            for metric, values in self.sensitivity_results.items():
                required_keys = {'S1', 'S1_conf', 'ST', 'ST_conf', 'names'}
                if not required_keys.issubset(values.keys()):
                    raise ValueError(f"Missing keys in the results for {metric}. Expected keys: {required_keys}")

    def plot_sensitivity_indices(self):
        for metric, Si in self.sensitivity_results.items():
            print(f"Plotting sensitivity indices for {metric}...")
            Si_df = pd.DataFrame(Si)

            # Ensure labels correspond to the variable names in JSON
            if 'names' not in Si_df.columns or 'S1' not in Si_df.columns or 'ST' not in Si_df.columns:
                raise ValueError(f"Incorrect data format for {metric}. Expected 'names', 'S1', 'S1_conf', 'ST', and 'ST_conf'.")

            # Plotting first-order sensitivity indices
            plt.figure(figsize=(12, 6))
            ax = barplot(Si_df[['names', 'S1', 'S1_conf']])
            ax.set_xticklabels(Si_df['names'], rotation=45, ha='right')
            plt.title(f'First-order Sobol Sensitivity Indices for {metric}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'first_order_sensitivity_{metric}.png'))
            plt.show()

            # Plotting total-order sensitivity indices
            plt.figure(figsize=(12, 6))
            ax = barplot(Si_df[['names', 'ST', 'ST_conf']])
            ax.set_xticklabels(Si_df['names'], rotation=45, ha='right')
            plt.title(f'Total-order Sobol Sensitivity Indices for {metric}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'total_order_sensitivity_{metric}.png'))
            plt.show()

# Usage
results_directory = 'sensitivity_analysis_results/global_sa_ev_static/'
plotter = SensitivityPlotter(results_directory)
plotter.plot_sensitivity_indices()
