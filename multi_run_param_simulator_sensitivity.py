import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import linear_sum_assignment
from agent import Agent 
from grid import Grid
from simulation_evolve import Simulation as SimulationEvolve
from simulation import Simulation
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
from sklearn.cluster import AgglomerativeClustering


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
    def __init__(self, simulation_params, num_runs, save_directory, do_feature_analysis='no', evolve=False, dynamic_tax=True, dynamic_market=True, plot_per_run=False, sensitivity_analysis='no', num_base_samples=1000, reference_vectors_path='reference_vector.csv'):
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

        self.grid_width = simulation_params.get('grid_width', [40])[0]
        self.grid_height = simulation_params.get('grid_height', [40])[0]

        # Load reference vectors from CSV
        self.reference_vectors = self.load_reference_vectors(reference_vectors_path)

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

    def load_reference_vectors(self, reference_vectors_path):
        df = pd.read_csv(reference_vectors_path)
        feature_columns = [col for col in df.columns if 'mean' in col]
        self.reference_columns = [col.replace('_mean', '') for col in feature_columns]
        reference_vectors = df[feature_columns].values
        
        # Normalize the reference vectors so that each vector's features sum to 1
        reference_vectors = reference_vectors / reference_vectors.sum(axis=1, keepdims=True)
        
        return reference_vectors

    def filter_numeric_params(self, params):
        """ Filter out non-numeric parameters. """
        return {k: v for k, v in params.items() if all(isinstance(i, (int, float)) for i in v)}

    def generate_param_combinations(self, params):
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def generate_sobol_samples(self, problem, num_base_samples):
        # Generate Sobol samples based on the number of base samples and problem definition
        return saltelli.sample(problem, num_base_samples)

    def run_simulations(self):
        all_metrics = []

        if self.sensitivity_analysis == 'no':
            for combination_index, param_set in enumerate(self.param_combinations, start=1):
                combination_metrics, combination_distances = self.run_simulation_combination(param_set, combination_index)
                all_metrics.extend(combination_metrics)
        else:  # Handle 'global' sensitivity analysis with Sobol sampling
            for sample_index, sample in enumerate(self.sobol_samples):
                param_set = dict(zip(self.problem['names'], sample))
                combination_metrics, combination_distances = self.run_simulation_combination(param_set, sample_index + 1)
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
        all_distances = []
        
        for run in range(1, self.num_runs + 1):
            print(f"Running simulation {run}/{self.num_runs} for combination {combination_index}...")
            print(f"Parameters: {param_set}")

            

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
            house_cost = param_set.get('house_cost', 2)
            house_cost = (house_cost, house_cost)
            max_house_num = param_set.get('max_house_num', 3)

            grid = Grid(width=self.grid_width, height=self.grid_height, house_cost=house_cost,  max_house_num=3)

            # house cost


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

            metrics, feature_importances, distances = self.calculate_metrics(sim.data, run, combination_index)
            combined_metrics = {**metrics, **feature_importances, **distances}
            all_metrics.append(combined_metrics)
            all_distances.append(distances)

            if self.plot_per_run:
                self.plot_run_data(sim.data, run, combination_index)

        return all_metrics, all_distances

    def calculate_average_metrics(self, metrics):
        metrics_df = pd.DataFrame(metrics)
        return metrics_df.mean().to_frame().T
    

    def save_run_data(self, data, run_number, combination_index):
        df = pd.DataFrame(data)
        combination_dir = os.path.join(self.save_directory, f"combination_{combination_index}")
        if not os.path.exists(combination_dir):
            os.makedirs(combination_dir)
        
        # Save raw data
        file_path = os.path.join(combination_dir, f"run_{run_number}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data for run {run_number} of combination {combination_index} saved to {file_path}.")

        # Calculate and save metrics
        metrics, feature_importances, distances = self.calculate_metrics(data, run_number, combination_index)
        
        # Combine metrics and feature importances into a single dictionary
        combined_metrics = {**metrics, **feature_importances, **distances}
        
        metrics_file_path = os.path.join(combination_dir, f"metrics_{run_number}.csv")
        metrics_df = pd.DataFrame([combined_metrics])
        metrics_df.to_csv(metrics_file_path, index=False)
        print(f"Metrics for run {run_number} of combination {combination_index} saved to {metrics_file_path}.")

        # Debug: Print vectors to verify correctness
        print("Saved Metrics DataFrame:")
        print(metrics_df)

    def calculate_metrics(self, data, run_number, combination_index):
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

        feature_importances = {}
        distances = {}
        if self.do_feature_analysis.lower() == 'yes':
            features, clusters = self.analyze_run_data(df)
            feature_importances = {f'{col}_cluster_{i}': val for i in features.index for col, val in features.loc[i].items()}
            
            # Ensure features and reference_vectors are aligned and have the same shape
            features_columns = features.columns.tolist()
            reference_vectors_columns = self.reference_columns
            
            # Find common columns
            common_columns = list(set(features_columns) & set(reference_vectors_columns))
            
            # Debug: Print column names and values before aligning
            print("Feature Vector Columns before aligning:", features_columns)
            print("Features before aligning:")
            print(features)
            print("Reference Vector Columns before aligning:", reference_vectors_columns)
            print("Reference Vectors before aligning:")
            print(pd.DataFrame(self.reference_vectors, columns=self.reference_columns))
            
            # Align the features and reference vectors
            features = features[common_columns].values
            reference_vectors = pd.DataFrame(self.reference_vectors, columns=self.reference_columns)[common_columns].values
            
            # Align clusters to the reference
            aligned_features = self.align_clusters(features, reference_vectors)
            
            # Debug: Print column names and values after aligning
            print("Feature Vector Columns after aligning:", common_columns)
            print("Features after aligning:")
            print(aligned_features)
            print("Reference Vector Columns after aligning:", common_columns)
            print("Reference Vectors after aligning:")
            print(reference_vectors)
            
            for i in range(len(aligned_features)):
                print(f"Feature vector for cluster {i}: {aligned_features[i]}")
                print(f"Reference vector for cluster {i}: {reference_vectors[i]}")
                dist = euclidean(aligned_features[i], reference_vectors[i])
                distances[f'euclidean_distance_cluster_{i}'] = dist

        # Ensure no NaN values in metrics
        for key in metrics:
            if np.isnan(metrics[key]):
                metrics[key] = 0.0

        return metrics, feature_importances, distances

    def align_clusters(self, target, reference):
        """
        Align clusters in the target array to those in the reference array.
        """
        # Replace NaN or infinite values in target with 0
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        # Replace NaN or infinite values in reference with 0
        reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute distances between each pair of clusters
        distances = cdist(target, reference, metric='euclidean')

        # Find the best matching clusters
        row_ind, col_ind = linear_sum_assignment(distances)

        # Reorder target clusters to match reference clusters
        aligned_target = target[row_ind]

        return aligned_target

    def analyze_run_data(self, df):
        # Compute the difference in wealth for each agent
        wealth_diff = df.groupby('agent_id')['wealth'].agg(['first', 'last']).reset_index()
        wealth_diff['wealth_diff'] = wealth_diff['last'] - wealth_diff['first']

        # Compute the mean income for each agent
        mean_income = df.groupby('agent_id')['income'].mean().reset_index()

        # Aggregate data for actions
        aggregated_data = df.groupby('agent_id').agg({
            'action': lambda x: x.value_counts().to_dict()
        }).reset_index()

        # Convert the action counts into separate columns
        action_df = aggregated_data['action'].apply(pd.Series).fillna(0)

        # Merge the aggregated data with wealth differences and mean income
        aggregated_data = pd.concat([aggregated_data.drop(columns=['action']), action_df], axis=1)
        aggregated_data = aggregated_data.merge(wealth_diff[['agent_id', 'wealth_diff']], on='agent_id')
        aggregated_data = aggregated_data.merge(mean_income[['agent_id', 'income']], on='agent_id')

        # Ensure all expected columns are present
        expected_columns = ['buy', 'sell', 'start_building', 'gather', 'wealth_diff', 'income']
        for col in expected_columns:
            if col not in aggregated_data.columns:
                aggregated_data[col] = 0

        # Drop the 'move', 'build', and 'continue_building' columns if they exist
        columns_to_drop = ['move', 'build', 'continue_building']
        for col in columns_to_drop:
            if col in aggregated_data.columns:
                aggregated_data = aggregated_data.drop(columns=[col])

        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(aggregated_data.drop(columns=['agent_id']))

        # Re-cluster into a specific number of clusters
        n_clusters = 3
        agglo_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = agglo_clustering.fit_predict(scaled_features)

        aggregated_data['cluster'] = clusters

        # Compute the feature values for each cluster and normalize them
        features = pd.DataFrame(scaled_features, columns=aggregated_data.drop(columns=['agent_id', 'cluster']).columns).groupby(clusters).mean()
        features = features.div(features.sum(axis=1), axis=0)  # Normalize so that each cluster's features sum to 1

        # Print the features for debugging
        print("Normalized Features for Clusters:")
        print(features)

        return features.reset_index(drop=True), clusters

    def is_json_serializable(self, v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    def save_sensitivity_analysis_results(self, all_metrics):
        if self.sensitivity_analysis != 'no':
            # Extract the relevant metrics for sensitivity analysis
            metrics_keys = all_metrics[0].keys()  # Corrected to access the first metric dictionary
            sobol_results = {}

            for metric in metrics_keys:
                # Collect all metric values across runs
                Y = [metrics[metric] for metrics in all_metrics]
                Y = np.array(Y)

                # Debugging: Print the shape and a sample of the data
                print(f"Metric: {metric}")
                print(f"Shape of collected data for {metric}: {Y.shape}")
                print(f"Sample data for {metric}: {Y[:10]}")  # Print first 10 values as a sample

                # Handle cases where Y contains NaNs
                if np.any(np.isnan(Y)):
                    print(f"Warning: NaNs detected in metric '{metric}'. Replacing NaNs with 0.")
                    Y = np.nan_to_num(Y, nan=0.0)

                # Skip metrics that are constant (all values are the same)
                if np.all(Y == Y[0]):
                    print(f"Skipping metric '{metric}' for sensitivity analysis because it is constant.")
                    continue

                Si = sobol.analyze(self.problem, Y, print_to_console=True)

                print(f"Sensitivity analysis for {metric}:")
                print("First-order sensitivity indices (S1):", Si['S1'])
                print("First-order confidence intervals (S1_conf):", Si.get('S1_conf', [None]*len(Si['S1'])))
                print("Total-order sensitivity indices (ST):", Si['ST'])
                print("Total-order confidence intervals (ST_conf):", Si.get('ST_conf', [None]*len(Si['ST'])))
                print("Parameter names:", self.problem['names'])

                # Ensure S1_conf and ST_conf are numpy arrays before calling tolist()
                Si['S1_conf'] = np.array(Si.get('S1_conf', [None]*len(Si['S1'])))
                Si['ST_conf'] = np.array(Si.get('ST_conf', [None]*len(Si['ST'])))

                sobol_results[metric] = {
                    'S1': Si['S1'].tolist(),
                    'S1_conf': Si['S1_conf'].tolist(),
                    'ST': Si['ST'].tolist(),
                    'ST_conf': Si['ST_conf'].tolist(),
                    'names': self.problem['names']
                }

            results_file_path = os.path.join(self.save_directory, 'sensitivity_analysis_results.json')
            with open(results_file_path, 'w') as f:
                json.dump(sobol_results, f, indent=4)
            print(f"Sensitivity analysis results saved to {results_file_path}")

    def plot_sensitivity_indices(self):
        results_file_path = os.path.join(self.save_directory, 'sensitivity_analysis_results.json')
        with open(results_file_path, 'r') as f:
            sensitivity_results = json.load(f)
        
        for metric, Si in sensitivity_results.items():
            print(f"Plotting sensitivity indices for {metric}...")
            Si_df = pd.DataFrame(Si)

            # Plotting first-order sensitivity indices
            plt.figure(figsize=(12, 6))
            barplot(Si_df[['names', 'S1', 'S1_conf']])
            plt.title(f'First-order Sobol Sensitivity Indices for {metric}')
            plt.savefig(os.path.join(self.save_directory, f'first_order_sensitivity_{metric}.png'))
            plt.show()

            # Plotting total-order sensitivity indices
            plt.figure(figsize=(12, 6))
            barplot(Si_df[['names', 'ST', 'ST_conf']])
            plt.title(f'Total-order Sobol Sensitivity Indices for {metric}')
            plt.savefig(os.path.join(self.save_directory, f'total_order_sensitivity_{metric}.png'))
            plt.show()

    def run_local_sensitivity_analysis(self):
        self.sensitivity_analysis = 'local'
        # Use the num_base_samples passed to the simulator
        self.run_simulations()

        results_file_path = os.path.join(self.save_directory, 'sensitivity_analysis_results.json')
        with open(results_file_path, 'r') as f:
            sensitivity_results = json.load(f)

        print("Sensitivity results loaded from JSON:", sensitivity_results)

        for metric, Si in sensitivity_results.items():
            Si_df = pd.DataFrame(Si)

            plt.figure(figsize=(12, 6))
            barplot(Si_df[['names', 'S1', 'S1_conf']])
            plt.title(f'First-order Sobol Sensitivity Indices for {metric}')
            plt.savefig(os.path.join(self.save_directory, f'first_order_sensitivity_local_{metric}.png'))
            plt.show()

            plt.figure(figsize=(12, 6))
            barplot(Si_df[['names', 'ST', 'ST_conf']])
            plt.title(f'Total-order Sobol Sensitivity Indices for {metric}')
            plt.savefig(os.path.join(self.save_directory, f'total_order_sensitivity_local_{metric}.png'))
            plt.show()

            threshold = 0.05
            influential_params = [Si['names'][i] for i in range(len(Si['S1'])) if Si['S1'][i] > threshold or Si['ST'][i] > threshold]

            print(f"Influential parameters for {metric}: {influential_params}")

constant_params = {
    'num_agents': [5, 50],  # Changed to a range
    'n_timesteps': [100, 1200],  # Changed to a range
    'num_resources': [50, 5000],  # Changed to a range
    'grid_width': [40, 100],  # Changed to a range
    'grid_height': [40, 100],  # Changed to a range
    # 'stone_rate': [0.5, 5],  # Changed to a range
    # 'wood_rate': [0.5, 5],  # Changed to a range
    'lifetime_mean': [70, 90],  # Changed to a range
    'lifetime_std': [5, 15],  # Changed to a range
    'resource_spawn_rate': [0.1, 5],  # Changed to a range
    'order_expiry_time': [3, 7],  # Changed to a range
    'tax_period': [1, 10],  # Changed to a range
    'income_per_timestep': [0.5, 5],  # Changed to a range
    'house_cost': [1, 10], # Changed to a range
    'num_houses': [1, 10] 

}

# Remove the parameter with None value for sensitivity analysis
filtered_params = {k: v for k, v in constant_params.items() if None not in v}
combined_params = {**filtered_params}

evolve = False
dynamic_tax = True
dynamic_market = True

num_runs = 10
num_base_samples = 128  # Number of base samples for Saltelli sampling

# Set save directory
save_directory='sensitivity_analysis_results/global_sa_ev_dynamic_v4/'

# Calculate the number of combinations
num_parameters = len(filtered_params)
total_combinations = num_base_samples * (num_parameters + 2)
print(f"Total number of combinations: {total_combinations}")

# Global sensitivity analysis mode with reduced parameter set
simulator_global_sa = MultipleRunSimulator(
    combined_params,
    num_runs=num_runs,
    save_directory=save_directory,
    do_feature_analysis='yes',
    evolve=evolve,
    dynamic_tax=dynamic_tax,
    dynamic_market=dynamic_market,
    plot_per_run=False,
    sensitivity_analysis='global',
    num_base_samples=num_base_samples,
    reference_vectors_path='reference_vec_ev_dynamic.csv',
)

simulator_global_sa.run_simulations()
simulator_global_sa.plot_sensitivity_indices()
