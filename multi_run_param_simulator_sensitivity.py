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
from simulation_evolve import Simulation as SimulationEvolve
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
        
        grid_params = simulation_params.get('grid', [{}])[0]
        self.grid_width = grid_params.get('grid_width', 40)
        self.grid_height = grid_params.get('grid_height', 40)

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        if self.sensitivity_analysis == 'no':
            self.param_combinations = self.generate_param_combinations(simulation_params)
        elif self.sensitivity_analysis == 'global':
            problem = {
                'num_vars': len(self.simulation_params),
                'names': list(self.simulation_params.keys()),
                'bounds': [[min(v), max(v)] for v in self.simulation_params.values()]
            }
            self.problem = problem
            self.sobol_samples = self.generate_sobol_samples(problem, self.num_base_samples)
        elif self.sensitivity_analysis == 'local':
            self.param_combinations = self.generate_local_param_combinations(simulation_params)

    def generate_param_combinations(self, params):
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def generate_local_param_combinations(self, params):
        base_combination = {k: v[0] for k, v in params.items()}
        local_combinations = []
        for param in params:
            for value in params[param]:
                local_combination = base_combination.copy()
                local_combination[param] = value
                local_combinations.append(local_combination)
        return local_combinations

    def generate_sobol_samples(self, problem, num_base_samples):
        return saltelli.sample(problem, num_base_samples)

    def run_simulations(self):
        if self.sensitivity_analysis == 'no':
            for combination_index, param_set in enumerate(self.param_combinations, start=1):
                self.run_simulation_combination(param_set, combination_index)
        elif self.sensitivity_analysis == 'global':
            for sample_index, sample in enumerate(self.sobol_samples):
                param_set = dict(zip(self.simulation_params.keys(), sample))
                self.run_simulation_combination(param_set, sample_index + 1)
        elif self.sensitivity_analysis == 'local':
            for combination_index, param_set in enumerate(self.param_combinations, start=1):
                self.run_simulation_combination(param_set, combination_index)

    def run_simulation_combination(self, param_set, combination_index):
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        if not os.path.exists(combination_dir):
            os.makedirs(combination_dir)

        params_file_path = os.path.join(combination_dir, 'params.json')
        serializable_param_set = {k: v for k, v in param_set.items() if self.is_json_serializable(v)}
        with open(params_file_path, 'w') as f:
            json.dump(serializable_param_set, f, indent=4)

        for run in range(1, self.num_runs + 1):
            print(f"Running simulation {run}/{self.num_runs} for combination {combination_index}...")
            print(f"Parameters: {param_set}")
            
            grid = Grid(width=self.grid_width, height=self.grid_height)
            
            param_set.pop('grid', None)
            
            if self.evolve:
                sim = SimulationEvolve(**param_set, grid=grid, dynamic_tax=self.dynamic_tax, dynamic_market=self.dynamic_market)
            else:
                sim = Simulation(**param_set, grid=grid, dynamic_tax=self.dynamic_tax, dynamic_market=self.dynamic_market, show_time=True)
            sim.run()
            self.save_run_data(sim.data, run, combination_index)
            
            if self.plot_per_run:
                self.plot_run_data(sim.data, run, combination_index)

    def save_run_data(self, data, run_number, combination_index):
        df = pd.DataFrame(data)
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data for run {run_number} of combination {combination_index} saved to {file_path}.")

        # Compute metrics and save
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

        metrics_file_path = os.path.join(combination_dir, f"metrics_run_{run_number}.json")
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics for run {run_number} of combination {combination_index} saved to {metrics_file_path}.")

        if self.do_feature_analysis.lower() == 'yes':
            feature_analysis, clusters = self.analyze_run_data(df)
            if feature_analysis is not None:
                fa_file_path = os.path.join(combination_dir, f"feature_analysis_run_{run_number}.csv")
                feature_analysis.to_csv(fa_file_path, index=False)
                print(f"Feature analysis for run {run_number} of combination {combination_index} saved to {fa_file_path}.")

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

    def load_run_data(self, combination_index, run_number):
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        params_file_path = os.path.join(combination_dir, 'params.json')
        df = pd.read_csv(file_path)
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        params['grid'] = Grid(width=self.grid_width, height=self.grid_height)
        print(f"Data for run {run_number} of combination {combination_index} loaded from {file_path} with parameters loaded from {params_file_path}.")
        return df, params

    def is_json_serializable(self, v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    def aggregate_results(self, combinations=None, runs=None):
        if combinations is None:
            combinations = range(1, len(self.param_combinations) + 1)
        if runs is None:
            runs = range(1, self.num_runs + 1)

        all_data = []
        feature_importances = []
        for combination_index in combinations:
            for run_number in runs:
                df, _ = self.load_run_data(combination_index, run_number)
                all_data.append(df)

                if self.do_feature_analysis.lower() == 'yes':
                    fa_file_path = os.path.join(self.save_directory, f"combination_{combination_index}", f"feature_analysis_run_{run_number}.csv")
                    if os.path.exists(fa_file_path):
                        feature_importances.append(pd.read_csv(fa_file_path))

        aggregated_data = pd.concat(all_data, ignore_index=True)
        aggregated_file_path = os.path.join(self.save_directory, 'aggregated_results.csv')
        aggregated_data.to_csv(aggregated_file_path, index=False)
        print(f"Aggregated data saved to {aggregated_file_path}.")

        if feature_importances:
            feature_importances = pd.concat(feature_importances, ignore_index=True)
            return aggregated_data, feature_importances
        else:
            return aggregated_data, None

    def plot_aggregated_results(self, combinations=None, runs=None):
        all_data, feature_importances = self.aggregate_results(combinations, runs)

        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['wealth'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Wealth')
        plt.title('Aggregated Wealth Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['houses'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Number of Houses')
        plt.title('Aggregated Number of Houses Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

        for agent_id in all_data['agent_id'].unique():
            agent_data = all_data[all_data['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['income'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Income')
        plt.title('Aggregated Income Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

        final_timestep = all_data[all_data['timestep'] == all_data['timestep'].max()]
        plt.hist(final_timestep['wealth'], bins=10, edgecolor='black')
        plt.xlabel('Wealth')
        plt.ylabel('Frequency')
        plt.title('Wealth Distribution at Final Timestep')
        plt.grid(True)
        plt.show()

        action_counts = all_data['action'].value_counts()
        action_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Distribution of Agent Actions')
        plt.grid(True)
        plt.show()

        incomes = all_data['income'].dropna()
        plt.hist(incomes, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.title('Aggregated Income Distribution Across All Timesteps')
        plt.grid(True)
        plt.show()

        if feature_importances is not None:
            feature_means = feature_importances.drop(columns=['cluster']).groupby('cluster').mean()
            feature_stds = feature_importances.drop(columns=['cluster']).groupby('cluster').std()

            for cluster in feature_means.index:
                means = feature_means.loc[cluster]
                stds = feature_stds.loc[cluster]
                plt.figure(figsize=(12, 6))
                plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
                plt.xlabel('Features')
                plt.ylabel('Mean Value')
                plt.title(f'Mean Feature Importances for Cluster {cluster}')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def save_sensitivity_analysis_results(self):
        if self.sensitivity_analysis == 'global':
            Y = []
            for combination_index in range(1, len(self.sobol_samples) + 1):
                combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
                for run_number in range(1, self.num_runs + 1):
                    metrics_file_path = os.path.join(combination_dir, f"metrics_run_{run_number}.json")
                    with open(metrics_file_path, 'r') as f:
                        metrics = json.load(f)
                    Y.append(metrics[self.sensitivity_metric])

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
    'num_agents': [30],  
    'n_timesteps': [1000],
    'num_resources': [500],
    'grid': [{'grid_width': 40, 'grid_height': 40}],
    'stone_rate': [1],
    'wood_rate': [1],
    'lifetime_mean': [80],
    'lifetime_std': [10],
    'resource_spawn_rate': [0.5],
    'order_expiry_time': [5],
    'save_file_path': [None],
    'tax_period': [1],
    'income_per_timestep': [1]
}

combined_params = {**constant_params}

evolve = False
dynamic_tax = False
dynamic_market = True

num_runs = 10
num_base_samples = 1000  # Number of base samples for Saltelli sampling
sensitivity_metric = 'total_welfare'  # Change this to the metric you want to analyze

# Standard mode
simulator_standard = MultipleRunSimulator(combined_params, num_runs=num_runs, save_directory='sensitivity_analysis_results/test_exp_v3_standard/', do_feature_analysis='yes', evolve=evolve, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market, plot_per_run=False, sensitivity_analysis='no')
simulator_standard.run_simulations()
aggregated_data_standard, feature_importances_standard = simulator_standard.aggregate_results()

# Global sensitivity analysis mode
simulator_global_sa = MultipleRunSimulator(combined_params, num_runs=num_runs, save_directory='sensitivity_analysis_results/test_exp_v3_global_sa/', do_feature_analysis='yes', evolve=evolve, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market, plot_per_run=False, sensitivity_analysis='global', num_base_samples=num_base_samples, sensitivity_metric=sensitivity_metric)
simulator_global_sa.run_simulations()
aggregated_data_global_sa, feature_importances_global_sa = simulator_global_sa.aggregate_results()
simulator_global_sa.save_sensitivity_analysis_results()
