import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
import os

def perform_ks_test_on_individual_runs(base_directory, given_csv_filename):
    # Construct the paths relative to the current script location
    run_files_path = os.path.join(base_directory, 'combination_1')
    given_csv_path = os.path.join('data', given_csv_filename)

    # Check if the directory exists
    if not os.path.exists(run_files_path):
        raise FileNotFoundError(f"The directory {run_files_path} does not exist. Check your path.")
    if not os.path.exists(given_csv_path):
        raise FileNotFoundError(f"The file {given_csv_path} does not exist. Check your path.")

    # List CSV files for individual runs
    run_files = [f for f in os.listdir(run_files_path) if f.startswith('run_') and f.endswith('.csv')]
    
    # Read the given income data and normalize it
    given_data = pd.read_csv(given_csv_path)
    given_income_data = given_data['income'].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    given_income_normalized = scaler.fit_transform(given_income_data).flatten()

    # Initialize plot for CDFs
    plt.figure(figsize=(12, 8))

    # Data structure to store KS results
    ks_results = []

    for run_file in run_files:
        run_file_path = os.path.join(run_files_path, run_file)
        simulation_data = pd.read_csv(run_file_path)
        agent_income_data = simulation_data.groupby('agent_id')['income'].mean().dropna().values.reshape(-1, 1)
        agent_income_normalized = scaler.fit_transform(agent_income_data).flatten()

        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(agent_income_normalized, given_income_normalized)
        ks_results.append([run_file.replace('.csv', ''), round(ks_statistic, 4), round(p_value, 4)])

        # Plot each run's CDF on the same figure, clean up the run name in the label
        plt.plot(np.sort(agent_income_normalized), np.linspace(0, 1, len(agent_income_normalized)), 
                 label=f'{run_file.replace("run_", "Run ").replace(".csv", "")}')

    # Plot CDF for the given income data
    plt.plot(np.sort(given_income_normalized), np.linspace(0, 1, len(given_income_normalized)), 
             label='Empirical Income Netherlands (CBS, 2022)', color='black', linestyle='--')

    # Finalize CDF plot
    plt.title(f'CDF Income EV agents Comparison Simulated vs. Empirical Data (10 Runs)')
    plt.xlabel('Normalized Income')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a DataFrame from the KS results
    ks_df = pd.DataFrame(ks_results, columns=['Run File', 'KS Statistic', 'P-Value'])

    # Plotting the table of KS test results
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the figure size as necessary
    ax.axis('off')  # Hide the axes

    # Add a table at the bottom of the axes
    the_table = ax.table(cellText=ks_df.values, colLabels=ks_df.columns, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.2)  # Adjust the scale if necessary

    plt.title('KS Test Statistics and P-Values for Each Run')
    plt.show()

# Usage example:
# Define relative paths based on the project structure
base_directory = './sensitivity_analysis_results/final_exp_EV_static_tax'
given_csv_filename = 'cleaned_income_data.csv'

perform_ks_test_on_individual_runs(base_directory, given_csv_filename)
