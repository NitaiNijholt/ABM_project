import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from agent import Agent
from grid import Grid
from market import Market
from orderbook import OrderBooks
from static_tax_policy import StaticTaxPolicy
from simulation import Simulation

class MultipleRunSimulator:
    def __init__(self, simulation_params, num_runs, save_directory):
        """
        Initialize the multiple run simulator with the given parameters.
        
        Parameters:
        - simulation_params (dict): A dictionary containing the parameters for the Simulation class.
                                     The values can be single values or lists of values.
        - num_runs (int): Number of simulation runs for each parameter combination.
        - save_directory (str): Directory to save the results of the simulations.
        """
        self.simulation_params = simulation_params
        self.num_runs = num_runs
        self.save_directory = save_directory

        # Create save directory if it does not exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Generate all parameter combinations
        self.param_combinations = self.generate_param_combinations(simulation_params)

    def generate_param_combinations(self, params):
        """
        Generate all combinations of parameters.
        
        Parameters:
        - params (dict): A dictionary containing parameter lists.
        
        Returns:
        - List of dictionaries with all parameter combinations.
        """
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def run_simulations(self):
        """
        Run the simulations multiple times for each parameter combination and save the results.
        """
        for combination_index, param_set in enumerate(self.param_combinations, start=1):
            combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
            if not os.path.exists(combination_dir):
                os.makedirs(combination_dir)

            # Save the parameter set
            params_file_path = os.path.join(combination_dir, 'params.json')
            serializable_param_set = {k: v for k, v in param_set.items() if self.is_json_serializable(v)}
            with open(params_file_path, 'w') as f:
                json.dump(serializable_param_set, f, indent=4)

            for run in range(1, self.num_runs + 1):
                print(f"Running simulation {run}/{self.num_runs} for parameter combination {combination_index}/{len(self.param_combinations)}...")
                sim = Simulation(**param_set)
                sim.run()
                self.save_run_data(sim.data, run, combination_index)

    def save_run_data(self, data, run_number, combination_index):
        """
        Save the data of a single run to a CSV file.
        
        Parameters:
        - data (list): The data to save.
        - run_number (int): The run number for naming the file.
        - combination_index (int): The index of the parameter combination.
        """
        df = pd.DataFrame(data)
        # Save the run data to CSV
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data for run {run_number} of combination {combination_index} saved to {file_path}.")

    def load_run_data(self, combination_index, run_number):
        """
        Load the data of a single run from a CSV file.
        
        Parameters:
        - combination_index (int): The index of the parameter combination.
        - run_number (int): The run number for loading the file.
        """
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        params_file_path = os.path.join(combination_dir, 'params.json')
        df = pd.read_csv(file_path)
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        
        # Add back the non-serializable objects
        params['grid'] = Grid(width=10, height=10)  # Adjust this as needed for your setup
        print(f"Data for run {run_number} of combination {combination_index} loaded from {file_path} with parameters loaded from {params_file_path}.")
        return df, params

    def is_json_serializable(self, v):
        """
        Check if a value is JSON serializable.
        
        Parameters:
        - v: The value to check.
        
        Returns:
        - bool: True if the value is JSON serializable, False otherwise.
        """
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    def aggregate_results(self, combinations=None, runs=None):
        """
        Aggregate results from selected runs for analysis.
        
        Parameters:
        - combinations (list): List of parameter combination indices to include in the aggregation.
                               If None, all combinations are included.
        - runs (list): List of run indices to include in the aggregation.
                       If None, all runs are included.
        
        Returns:
        - A DataFrame containing the aggregated data.
        """
        if combinations is None:
            combinations = range(1, len(self.param_combinations) + 1)
        if runs is None:
            runs = range(1, self.num_runs + 1)

        all_data = []
        for combination_index in combinations:
            for run_number in runs:
                df, _ = self.load_run_data(combination_index, run_number)
                all_data.append(df)

        aggregated_data = pd.concat(all_data, ignore_index=True)
        aggregated_file_path = os.path.join(self.save_directory, 'aggregated_results.csv')
        aggregated_data.to_csv(aggregated_file_path, index=False)
        print(f"Aggregated data saved to {aggregated_file_path}.")
        return aggregated_data

    def plot_aggregated_results(self, combinations=None, runs=None):
        """
        Plot aggregated results from selected runs.
        
        Parameters:
        - combinations (list): List of parameter combination indices to include in the plot.
                               If None, all combinations are included.
        - runs (list): List of run indices to include in the plot.
                       If None, all runs are included.
        """
        all_data = self.aggregate_results(combinations, runs)

        # Plot aggregated wealth over time
        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['wealth'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Wealth')
        plt.title('Aggregated Wealth Over Time')
        plt.grid(True)
        plt.show()

        # Plot aggregated number of houses over time
        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['houses'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Number of Houses')
        plt.title('Aggregated Number of Houses Over Time')
        plt.grid(True)
        plt.show()

        # Plot aggregated income over time
        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['income'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Income')
        plt.title('Aggregated Income Over Time')
        plt.grid(True)
        plt.show()

        # Plot distribution of wealth at the final timestep
        final_timestep = all_data[all_data['timestep'] == all_data['timestep'].max()]
        plt.hist(final_timestep['wealth'], bins=10, edgecolor='black')
        plt.xlabel('Wealth')
        plt.ylabel('Frequency')
        plt.title('Wealth Distribution at Final Timestep')
        plt.grid(True)
        plt.show()

        # Plot distribution of agent actions over time
        action_counts = all_data['action'].value_counts()
        action_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Distribution of Agent Actions')
        plt.grid(True)
        plt.show()

        # Plot aggregated income distribution over all timesteps
        incomes = all_data['income'].dropna()  # Drop any NaN values if they exist
        plt.hist(incomes, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.title('Aggregated Income Distribution Across All Timesteps')
        plt.grid(True)
        plt.show()

# Example usage
simulation_params = {
    'num_agents': [10, 20,30],
    'grid': [Grid(width=10, height=10), Grid(width=20, height=20), Grid(width=30, height=30)],
    'n_timesteps': [100],
    'num_resources': [50],
    'wood_rate': [1],
    'stone_rate': [1],
    'lifetime_mean': [80],
    'lifetime_std': [10],
    'resource_spawn_period': [1],
    'agent_spawn_period': [10],
    'order_expiry_time': [5],
    'save_file_path': [None],
    'tax_period': [30],
    'income_per_timestep': [1]
}

num_runs = 5
save_directory = 'multirun_simulation_results'

multiple_simulator = MultipleRunSimulator(simulation_params, num_runs, save_directory)
multiple_simulator.run_simulations()
multiple_simulator.plot_aggregated_results()

# Example to load data from a specific run
combination_index = 1
run_number = 1
run_data, run_params = multiple_simulator.load_run_data(combination_index, run_number)
print(run_data.head())
print(run_params)
