import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from agent import Agent
from agent_static_market import Agent_static_market
# from agent_static_market import Agent_static_market as Agent
from grid import Grid
from market import Market
from orderbook import OrderBooks
from static_tax_policy import StaticTaxPolicy
from dynamic_tax_policy import DynamicTaxPolicy
import psutil

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB, VMS: {mem_info.vms / 1024 ** 2:.2f} MB")


class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1, num_resources=0, wood_rate=1, stone_rate=1, 
                 lifetime_mean=80, lifetime_std=10, resource_spawn_rate=0.5, order_expiry_time=5, 
                 save_file_path=None, tax_period=1, income_per_timestep=1, show_time=False, dynamic_tax=True, dynamic_market=True):
        """
        Initialize the simulation with the given parameters.

        Parameters
        ----------
        num_agents : int
            Number of agents in the simulation.
        grid : Grid
            Grid object representing the environment.
        n_timesteps : int, optional
            Number of timesteps for the simulation (default is 1).
        num_resources : int, optional
            Number of initial resources (default is 0).
        wood_rate : float, optional
            Initial rate of wood (default is 1).
        stone_rate : float, optional
            Initial rate of stone (default is 1).
        lifetime_mean : float, optional
            Mean of the agents' lifetime distribution (default is 80).
        lifetime_std : float, optional
            Standard deviation of the agents' lifetime distribution (default is 10).
        resource_spawn_rate : float, optional
            Rate at which resources spawn on the grid (default is 0.5).
        order_expiry_time : int, optional
            The amount of timesteps an order stays in the market until it expires (default is 5).
        save_file_path : str, optional
            File path to save the simulation results (default is None).
        tax_period : int, optional
            Periodicity of tax collection (default is 1).
        income_per_timestep : int, optional
            Income per timestep for the agents (default is 1).
        show_time : bool, optional
            Whether to display the current timestep during the simulation (default is False).
        dynamic_tax : bool, optional
            Whether to use dynamic tax policies (default is True).
        dynamic_market : bool, optional
            Whether to use dynamic market prices (default is True).
        """
        self.t = 0
        self.grid = grid
        self.num_agents = num_agents
        self.n_timesteps = n_timesteps
        self.num_resources = num_resources
        self.wealth_over_time = {}
        self.houses_over_time = {}
        self.gathered_at_timesteps = {}
        self.bought_at_timesteps = {}
        self.sold_at_timesteps = {}
        self.taxes_paid_at_timesteps = {}
        self.lifetime_mean = lifetime_mean
        self.lifetime_std = lifetime_std
        self.resource_spawn_rate = resource_spawn_rate
        self.save_file_path = save_file_path
        self.tax_period = tax_period
        self.writer = None
        self.show_time = show_time
        self.income_per_timestep = income_per_timestep
        self.data = []
        self.initial_wealth = []
        self.average_tax_values = {}
        self.equality = {}
        self.productivity = {}
        self.social_welfare = {}
        self.total_discounted_welfare_change = {}
        self.dynamic_tax = dynamic_tax
        self.dynamic_market = dynamic_market

        # Initialize Dynamic market
        self.wood_order_book = OrderBooks(self.get_agents_dict(), 'wood', order_expiry_time, self.grid.agents)
        self.stone_order_book = OrderBooks(self.get_agents_dict(), 'stone', order_expiry_time, self.grid.agents)
        
        # Initialize Static price market
        self.market = Market(wood_rate, stone_rate)
        self.tax_policy = None

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"
        
        # Load wealth distribution data
        self.wealth_distribution = self.load_distribution_data('data/distribution_data.json')

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # Initialize resources
        self.initialize_resources()

    def load_distribution_data(self, file_path):
        """
        Load wealth distribution data from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing wealth distribution data.

        Returns
        -------
        dict
            The wealth distribution data.
        """
        with open(file_path, 'r') as file:
            distribution_data = json.load(file)
        return distribution_data

    def make_agent(self, agent_id):
        """
        Create and place an agent on the grid.

        Parameters
        ----------
        agent_id : int
            Unique identifier for the agent.
        """
        mask_agent = self.grid.agent_matrix == 0
        mask_house = self.grid.house_matrix == 0

        common_zeros = np.where(mask_agent & mask_house)

        if common_zeros[0].size > 0:
            random_index = np.random.choice(len(common_zeros[0]))
            position = (common_zeros[0][random_index], common_zeros[1][random_index])
        else:
            return
        
        # Sample wealth
        if self.wealth_distribution:
            middle_values = self.wealth_distribution['middle_values']
            normalized_percentages = self.wealth_distribution['normalized_percentages']
            wealth = np.random.choice(middle_values, p=normalized_percentages)
        else:
            wealth = 0  
            
        self.initial_wealth.append(wealth)
        
        if self.dynamic_market:
            agent = Agent(self, agent_id, position, self.grid, self.market, lifetime_mean=self.lifetime_mean, lifetime_std=self.lifetime_std, creation_time=self.t, wealth=wealth, income_per_timestep=self.income_per_timestep)
        else:
            agent = Agent_static_market(self, agent_id, position, self.grid, self.market, lifetime_mean=self.lifetime_mean, lifetime_std=self.lifetime_std, creation_time=self.t, wealth=wealth, income_per_timestep=self.income_per_timestep)
        self.grid.agents[agent_id] = agent
        self.grid.agent_matrix[position] = agent_id

    def get_random_position(self):
        """
        Get a random position on the grid.

        Returns
        -------
        tuple
            Random (x, y) position on the grid.
        """
        x = np.random.randint(self.grid.width)
        y = np.random.randint(self.grid.height)
        return (x, y)

    def timestep(self):
        """
        Execute one timestep in the simulation.

        This method updates the agents, order books, and market prices, and handles resource spawning and tax collection.
        """
        self.t += 1

        if self.t % 100 == 0:
            print(f"Current timestep: {self.t}")

        agents = list(self.grid.agents.values())
        
        # Update order books with current agents' state
        self.wood_order_book.agents_dict = self.get_agents_dict()
        self.stone_order_book.agents_dict = self.get_agents_dict()
        
        for agent in agents:
            agent.step()
            self.data.append({
                'timestep': self.t,
                'agent_id': agent.agent_id,
                'wealth': agent.wealth,
                'houses': len(agent.houses),
                'wood': agent.wood,
                'stone': agent.stone,
                'income': agent.income,
                'position': agent.position,
                'action': agent.current_action
            })
            agent.taxes_paid_at_timesteps.append(0)
        self.spawn_resources()
        self.grid.update_house_incomes()
        if self.t % self.tax_period == 0:
            self.tax_policy.apply_taxes()

        # Update order books with current agents' state
        self.wood_order_book.agents_dict = self.get_agents_dict()
        self.stone_order_book.agents_dict = self.get_agents_dict()
        
        # Increment timestep and remove expired orders
        self.wood_order_book.increment_timestep()
        self.stone_order_book.increment_timestep()

        # Update market prices
        self.market.update_price()
        
    def run(self, show_time=False):
        """
        Run the simulation for the specified number of timesteps.

        Parameters
        ----------
        show_time : bool, optional
            Whether to display the current timestep during the simulation (default is False).
        """
        self.show_time = show_time
        if self.dynamic_tax:
            self.tax_policy = DynamicTaxPolicy(self.grid, self)
        else:
            self.tax_policy = StaticTaxPolicy(self.grid, self)
        for t in range(self.n_timesteps):
            if self.show_time:
                print(f"\nTimestep {t+1}:")
            self.timestep()
        # Save the results to a CSV file after the simulation
        self.save_results(self.save_file_path)

    def initialize_resources(self):
        """
        Initialize resources on the grid.

        This method places resources randomly on the grid.
        """
        for _ in range(self.num_resources):
            self.grid.resource_matrix_wood[self.get_random_position()] += 1
            self.grid.resource_matrix_stone[self.get_random_position()] += 1

    def spawn_resources(self):
        """
        Spawn resources on the grid randomly.
        """
        num_wood = np.sum(self.grid.resource_matrix_wood)
        num_stone = np.sum(self.grid.resource_matrix_stone)

        for _ in range(int(self.resource_spawn_rate * (self.num_resources - num_stone))):
            stone_position = self.get_random_position()
            if self.grid.if_no_houses(stone_position):
                self.grid.resource_matrix_stone[stone_position] += 1

        for _ in range(int(self.resource_spawn_rate * (self.num_resources - num_wood))):
            wood_position = self.get_random_position()
            if self.grid.if_no_houses(wood_position):
                self.grid.resource_matrix_wood[wood_position] += 1

    def spawn_agents(self):
        """
        Spawn agents on the grid randomly.
        """
        current_num_agents = len(self.grid.agents)
        last_agent_id = max(self.grid.agents.keys())
        for _ in range(self.num_agents - current_num_agents):
            last_agent_id += 1
            self.make_agent(max(self.grid.agents.keys()))
            
    def get_agents_dict(self):
        """
        Get a dictionary of agents' resources.

        Returns
        -------
        dict
            Dictionary with agent IDs as keys and their resources as values.
        """
        return {agent_id: {'wealth': agent.wealth, 'wood': agent.wood, 'stone': agent.stone} 
                for agent_id, agent in self.grid.agents.items()}



    def save_results(self, file_path):
        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index=False)

    def load_results(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def plot_results(self, file_path=None):
        if file_path:
            df = self.load_results(file_path)
        else:
            df = pd.DataFrame(self.data)
        
        # Plot Market price over time
        price_history_wood = []
        price_history_stone = []
        for transaction in self.wood_order_book.transactions:
            price_history_wood.append(transaction['price'])
        for transaction in self.stone_order_book.transactions:
            price_history_stone.append(transaction['price'])

        # Plotting the price history
        plt.figure(figsize=(10, 6))
        plt.plot(price_history_wood, marker='o', markersize= 3, linestyle='-', color='g', label = 'wood')
        plt.plot(price_history_stone, marker='o', markersize = 3, linestyle='-', color='b', label ='stone')
        plt.title('Price History of Wood and Stone Transactions')
        plt.xlabel('Transaction Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot wealth over time
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['wealth'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Wealth')
        plt.title('Wealth EV agent Over Time')
        plt.grid(True)
        plt.show()

        # Plot number of houses over time
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['houses'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Number of Houses')
        plt.title('Number of Houses Over Time')
        plt.grid(True)
        plt.show()

        # Plot income over time
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['income'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Income')
        plt.title('Income Over Time')
        plt.grid(True)
        plt.show()

        # Plot distribution of wealth at the final timestep
        final_timestep = df[df['timestep'] == self.n_timesteps]
        plt.hist(final_timestep['wealth'], bins=10, edgecolor='black')
        plt.xlabel('Wealth')
        plt.ylabel('Frequency')
        plt.title('Wealth Distribution at Final Timestep')
        plt.grid(True)
        plt.show()

        # Plot distribution of agent actions over time
        action_counts = df['action'].value_counts()
        action_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Distribution of Agent Actions')
        plt.grid(True)
        plt.show()

        # Plot aggregated income distribution over all timesteps
        self.plot_aggregated_income_distribution(df)

    def plot_aggregated_income_distribution(self, df=None):
        if df is None:
            df = pd.DataFrame(self.data)
        incomes = df['income'].dropna()  # Drop any NaN values if they exist
        plt.hist(incomes, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.title('Aggregated Income Distribution Across All Timesteps')
        plt.grid(True)
        plt.show()

    def plot_action_distribution(self):
        df = pd.DataFrame(self.data)
        action_counts = df['action'].value_counts()
        action_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Distribution of Agent Actions')
        plt.grid(True)
        plt.show()

    def plot_initial_wealth_distribution(self):
        plt.hist(self.initial_wealth, bins=10, edgecolor='black')
        plt.xlabel('Wealth')
        plt.ylabel('Frequency')
        plt.title('Initial Wealth Distribution')
        plt.grid(True)
        plt.show()

    def plot_average_tax_values(self):
        timesteps = list(self.average_tax_values.keys())
        average_tax_values = list(self.average_tax_values.values())
        plt.plot(timesteps, average_tax_values)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Tax Amount')
        plt.title('Average Tax Amount Over Time')
        plt.grid(True)
        plt.show()

    def plot_equality_over_time(self):
        timesteps = list(self.equality.keys())
        equality_values = list(self.equality.values())
        plt.plot(timesteps, equality_values)
        plt.xlabel('Timesteps')
        plt.ylabel('Equality')
        plt.title('Equality Over Time')
        plt.grid(True)
        plt.show()


    def plot_productivity_over_time(self):
        timesteps = list(self.productivity.keys())
        productivity_values = list(self.productivity.values())
        plt.plot(timesteps, productivity_values)
        plt.xlabel('Timesteps')
        plt.ylabel('Productivity')
        plt.title('Productivity Over Time')
        plt.grid(True)
        plt.show()

    def plot_social_welfare_over_time(self):
        timesteps = list(self.social_welfare.keys())
        social_welfare_values = list(self.social_welfare.values())
        plt.plot(timesteps, social_welfare_values)
        plt.xlabel('Timesteps')
        plt.ylabel('Social Welfare')
        plt.title('Social Welfare Over Time')
        plt.grid(True)
        plt.show()

    def plot_total_discounted_welfare_change_over_time(self):
        timesteps = list(self.total_discounted_welfare_change.keys())
        total_discounted_welfare_change = list(self.total_discounted_welfare_change.values())
        plt.plot(timesteps, total_discounted_welfare_change)
        plt.xlabel('Timesteps')
        plt.ylabel('Total Discounted Welfare Change')
        plt.title('Total Discounted Welfare Change Over Time')
        plt.grid(True)
        plt.show()
