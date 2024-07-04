import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from agent import Agent  # Assuming these modules are correctly imported
from grid import Grid
from simulation_evolve import SimulationEvolve
from simulation import Simulation
from SALib.sample import saltelli
from SALib.analyze import sobol

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

class MultipleRunSimulator:
    def __init__(self, simulation_params, num_runs, save_directory, do_feature_analysis='no', evolve=False, dynamic_tax=True, dynamic_market=True, plot_per_run=False, sensitivity_analysis='no', num_base_samples=1000, sensitivity_metric='total_welfare'):
        self.simulation_params = simulation_params
        self.num_runs = num_runs
        self.save_directory = save_directory
        self.do_feature_analysis = do_feature_analysis
        self.evolve = evolve
        self.dynamic_tax = dynamic_tax
        self.dynamic_market = dynamic_market
        self.plot_per_run = plot_per_run
        self.sensitivity_analysis = sensitivity_analysis
        self.num_base_samples = num_base_samples
        self.sensitivity_metric = sensitivity_metric

        self.grid_width = simulation_params.get('grid_width', [40])[0]
        self.grid_height = simulation_params.get('grid_height', [40])[0]

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        if self.sensitivity_analysis == 'no':
            self.param_combinations = self.generate_param_combinations(simulation_params)
        else:
            valid_params = self.filter_numeric_params(simulation_params)
            problem = {
                'num_vars': len(valid_params),
                'names': list(valid_params.keys()),
                'bounds': [[min(v), max(v)] for v in valid_params.values()]
            }
            self.problem = problem
            self.sobol_samples = self.generate_sobol_samples(problem, self.num_base_samples)

    def filter_numeric_params(self, params):
        """ Filter out non-numeric parameters. """
        return {k: v for k, v in params.items() if all(isinstance(i, (int, float)) for i in v)}

    def generate_param_combinations(self, params):
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def generate_sobol_samples(self, problem, num_base_samples):
        return saltelli.sample(problem, num_base_samples)

    def run_simulations(self):
        all_metrics = []

        if self.sensitivity_analysis == 'no':
            for combination_index, param_set in enumerate(self.param_combinations, start=1):
                combination_metrics = self.run_simulation_combination(param_set, combination_index)
                all_metrics.extend(combination_metrics)
        else:  # Handle both 'global' and 'local' sensitivity analysis with Sobol sampling
            for sample_index, sample in enumerate(self.sobol_samples):
                param_set = dict(zip(self.problem['names'], sample))
                combination_metrics = self.run_simulation_combination(param_set, sample_index + 1)
                all_metrics.extend(combination_metrics)

            self.save_sensitivity_analysis_results(all_metrics)

    def run_simulation_combination(self, param_set, combination_index):
        # Convert specific parameters to integers
        int_params = ['num_agents', 'n_timesteps', 'num_resources', 'grid_width', 'grid_height', 'order_expiry_time', 'tax_period', 'income_per_timestep']
        for param in int_params:
            if param in param_set:
                param_set[param] = int(param_set[param])

        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        if not os.path.exists(combination_dir):
            os.makedirs(combination_dir)

        params_file_path = os.path.join(combination_dir, 'params.json')
        serializable_param_set = {k: v for k, v in param_set.items() if self.is_json_serializable(v)}
        with open(params_file_path, 'w') as f:
            json.dump(serializable_param_set, f, indent=4)

        all_metrics = []
        for run in range(1, self.num_runs + 1):
            print(f"Running simulation {run}/{self.num_runs} for combination {combination_index}...")
            print(f"Parameters: {param_set}")

            grid = Grid(width=self.grid_width, height=self.grid_height)

            # Extract individual parameters from param_set
            num_agents = param_set.get('num_agents', 30)
            n_timesteps = param_set.get('n_timesteps', 1000)
            num_resources = param_set.get('num_resources', 500)
            stone_rate = param_set.get('stone_rate', 1)
            wood_rate = param_set.get('wood_rate', 1)
            lifetime_mean = param_set.get('lifetime_mean', 80)
            lifetime_std = param_set.get('lifetime_std', 10)
            resource_spawn_rate = param_set.get('resource_spawn_rate', 0.5)
            order_expiry_time = param_set.get('order_expiry_time', 5)
            save_file_path = param_set.get('save_file_path', None)
            tax_period = param_set.get('tax_period', 1)
            income_per_timestep = param_set.get('income_per_timestep', 1)

            if self.evolve:
                sim = SimulationEvolve(num_agents=num_agents, n_timesteps=n_timesteps, num_resources=num_resources,
                                       stone_rate=stone_rate, wood_rate=wood_rate, lifetime_mean=lifetime_mean,
                                       lifetime_std=lifetime_std, resource_spawn_rate=resource_spawn_rate,
                                       order_expiry_time=order_expiry_time, save_file_path=save_file_path,
                                       tax_period=tax_period, income_per_timestep=income_per_timestep,
                                       grid=grid, dynamic_tax=self.dynamic_tax, dynamic_market=self.dynamic_market)
            else:
                sim = Simulation(num_agents=num_agents, n_timesteps=n_timesteps, num_resources=num_resources,
                                 stone_rate=stone_rate, wood_rate=wood_rate, lifetime_mean=lifetime_mean,
                                 lifetime_std=lifetime_std, resource_spawn_rate=resource_spawn_rate,
                                 order_expiry_time=order_expiry_time, save_file_path=save_file_path,
                                 tax_period=tax_period, income_per_timestep=income_per_timestep,
                                 grid=grid, dynamic_tax=self.dynamic_tax, dynamic_market=self.dynamic_market,
                                 show_time=True)
            sim.run()
            self.save_run_data(sim.data, run, combination_index)

            metrics = self.calculate_metrics(sim.data)
            all_metrics.append(metrics)

            if self.plot_per_run:
                self.plot_run_data(sim.data, run, combination_index)

        # Calculate and save average metrics
        average_metrics = self.calculate_average_metrics(all_metrics)
        average_metrics_csv_path = os.path.join(combination_dir, 'average_metrics.csv')
        average_metrics.to_csv(average_metrics_csv_path, index=False)
        print(f"Average metrics for combination {combination_index} saved to {average_metrics_csv_path}.")

        return all_metrics

    def calculate_average_metrics(self, metrics):
        metrics_df = pd.DataFrame(metrics)
        return metrics_df.mean().to_frame().T

    def save_run_data(self, data, run_number, combination_index):
        df = pd.DataFrame(data)
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data for run {run_number} of combination {combination_index} saved to {file_path}.")

    def calculate_metrics(self, data):
        df = pd.DataFrame(data)
        initial_wealth_per_agent = df.groupby('agent_id').first()['wealth']
        final_wealth_per_agent = df.groupby('agent_id').last()['wealth']
        wealth_diff_per_agent = final_wealth_per_agent - initial_wealth_per_agent

        normalized_final_wealth = normalize(final_wealth_per_agent)
        normalized_wealth_diff = normalize(wealth_diff_per_agent)

        gini = gini_coefficient(normalized_wealth_diff.values)
        prod = np.sum(normalized_final_wealth.values)
        eq = 1 - (len(normalized_final_wealth) / (len(normalized_final_wealth) - 1)) * gini
        swf = eq * prod

        metrics = {
            "total_welfare": sum(normalized_final_wealth),
            "gini_coefficient": gini,
            "total_productivity": prod,
            "social_equality": eq,
            "social_welfare": swf
        }

        # Adding feature importances if required
        if self.do_feature_analysis.lower() == 'yes':
            feature_analysis, _ = self.analyze_run_data(df)
            for feature in feature_analysis.columns:
                metrics[feature] = feature_analysis[feature].mean()

        return metrics

    def analyze_run_data(self, df):
        wealth_diff = df.groupby('agent_id')['wealth'].agg(['first', 'last']).reset_index()
        wealth_diff['wealth_diff'] = wealth_diff['last'] - wealth_diff['first']

        mean_income = df.groupby('agent_id')['income'].mean().reset_index()

        aggregated_data = df.groupby('agent_id').agg({
            'action': lambda x: x.value_counts().to_dict()
        }).reset_index()

        action_df = aggregated_data['action'].apply(pd.Series).fillna(0).drop(columns=['move'], errors='ignore')

        aggregated_data = pd.concat([aggregated_data.drop(columns=['action']), action_df], axis=1)
        aggregated_data = aggregated_data.merge(wealth_diff[['agent_id', 'wealth_diff']], on='agent_id')
        aggregated_data = aggregated_data.merge(mean_income[['agent_id', 'income']], on='agent_id')

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(aggregated_data.drop(columns=['agent_id']))

        hierarchical_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_full_tree=True)
        clusters = hierarchical_clustering.fit_predict(scaled_features)

        if self.plot_per_run:
            plt.figure(figsize=(10, 7))
            linked = linkage(scaled_features, 'ward')
            plt.axhline(y=linked[-(2-1), 2], color='r', linestyle='--')
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            plt.show()

        n_clusters = 3
        hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = hierarchical_clustering.fit_predict(scaled_features)

        aggregated_data['cluster'] = clusters

        feature_means = pd.DataFrame(scaled_features, columns=aggregated_data.drop(columns=['agent_id', 'cluster']).columns).groupby(aggregated_data['cluster']).mean()
        colors = plt.get_cmap('tab10')

        if self.plot_per_run:
            fig, axs = plt.subplots(n_clusters, 1, figsize=(12, 8), constrained_layout=True)
            for i in range(n_clusters):
                feature_means.loc[i].plot(kind='bar', ax=axs[i], color=colors(i % 10))
                axs[i].set_title(f'Normalized Feature Means for Cluster {i}')
                axs[i].set_xlabel('Features')
                axs[i].set_ylabel('Mean Standardized Value')
                axs[i].grid(True)
                axs[i].tick_params(axis='x', rotation=45)
            plt.tight_layout(pad=3.0)
            plt.show()

            fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
            for cluster in np.unique(clusters):
                cluster_data = aggregated_data[aggregated_data['cluster'] == cluster]
                label = f'Cluster {cluster}'
                color = colors(cluster % 10)
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

        feature_means['cluster'] = feature_means.index
        return feature_means.reset_index(drop=True), clusters

    def is_json_serializable(self, v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    def save_sensitivity_analysis_results(self, all_metrics):
        if self.sensitivity_analysis != 'no':
            # Extract the relevant metrics for sensitivity analysis
            Y = [metrics[self.sensitivity_metric] for metrics in all_metrics]

            Y = np.array(Y)
            Si = sobol.analyze(self.problem, Y, print_to_console=True)

            sensitivity_results = {
                'S1': Si['S1'].tolist(),
                'ST': Si['ST'].tolist(),
                'names': self.problem['names']
            }

            results_file_path = os.path.join(self.save_directory, 'sensitivity_analysis_results.json')
            with open(results_file_path, 'w') as f:
                json.dump(sensitivity_results, f, indent=4)
            print(f"Sensitivity analysis results saved to {results_file_path}")

# Example usage
constant_params = {
    'num_agents': [20, 50],  # Changed to a range
    'n_timesteps': [800, 1200],  # Changed to a range
    'num_resources': [400, 600],  # Changed to a range
    'grid_width': [30, 50],  # Changed to a range
    'grid_height': [30, 50],  # Changed to a range
    'stone_rate': [0.8, 1.2],  # Changed to a range
    'wood_rate': [0.8, 1.2],  # Changed to a range
    'lifetime_mean': [70, 90],  # Changed to a range
    'lifetime_std': [5, 15],  # Changed to a range
    'resource_spawn_rate': [0.4, 0.6],  # Changed to a range
    'order_expiry_time': [3, 7],  # Changed to a range
    'tax_period': [1, 2],  # Changed to a range
    'income_per_timestep': [0.5, 1.5]  # Changed to a range
}

# Remove the parameter with None value for sensitivity analysis
filtered_params = {k: v for k, v in constant_params.items() if None not in v}
combined_params = {**filtered_params}

evolve = False
dynamic_tax = False
dynamic_market = True

num_runs = 2
num_base_samples = 4  # Number of base samples for Saltelli sampling
sensitivity_metric = 'gini_coefficient'  # Change this to the metric you want to analyze

# Standard mode
# simulator_standard = MultipleRunSimulator(combined_params, num_runs=num_runs, save_directory='sensitivity_analysis_results/test_exp_v3_standard/', do_feature_analysis='yes', evolve=evolve, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market, plot_per_run=False, sensitivity_analysis='no')
# simulator_standard.run_simulations()

# Global sensitivity analysis mode
simulator_global_sa = MultipleRunSimulator(combined_params, num_runs=num_runs, save_directory='sensitivity_analysis_results/test_exp_v3_global_sa/', do_feature_analysis='yes', evolve=evolve, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market, plot_per_run=False, sensitivity_analysis='global', num_base_samples=num_base_samples, sensitivity_metric=sensitivity_metric)
simulator_global_sa.run_simulations()

# Local sensitivity analysis mode (also using Sobol sampling)
simulator_local_sa = MultipleRunSimulator(combined_params, num_runs=num_runs, save_directory='sensitivity_analysis_results/test_exp_v3_local_sa/', do_feature_analysis='yes', evolve=evolve, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market, plot_per_run=False, sensitivity_analysis='local', num_base_samples=num_base_samples, sensitivity_metric=sensitivity_metric)
simulator_local_sa.run_simulations()
