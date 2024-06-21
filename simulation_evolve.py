import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from intelligent_agent import Agent
from grid import Grid
from market import Market
from orderbook import OrderBooks
from dynamic_tax_policy import DynamicTaxPolicy as TaxPolicy
from scipy.stats import gamma, lognorm


class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1, num_resources=0, wood_rate=1, stone_rate=1, 
                 lifetime_mean=80, lifetime_std=10, resource_spawn_rate=0.5, order_expiry_time=5, 
                 save_file_path=None, tax_period=1, lifetime_distribution='gamma', income_per_timestep=1, show_time=False):
        """
        order_expiry_time (int): The amount of timesteps an order stays in the market until it expires
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
        self.mutation_probability = 0.001
        self.k = 2

        self.lifetime_distribution = lifetime_distribution

        self.agent_dict = {}
        self.action_failure = 0

        self.moving = 0
        self.failed_moving = 0
        self.gathering = 0
        self.failed_gathering = 0
        self.buy = 0
        self.failed_buy = 0
        self.sell = 0
        self.failed_sell = 0
        self.build = 0
        self.failed_build = 0

        # Initialize Static price market
        self.market = Market(wood_rate, stone_rate)
        self.tax_policy = None

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"
        
        # Load wealth distribution data
        # MAKE SURE THIS LINE IS BEFORE AGENTS ARE CREATED
        self.wealth_distribution = self.load_distribution_data('data/distribution_data.json')

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # # Plot initial wealth distribution
        # self.plot_initial_wealth_distribution()
        
        # Initialize resources
        self.initialize_resources()

    def load_distribution_data(self, file_path):
        with open(file_path, 'r') as file:
            distribution_data = json.load(file)
        return distribution_data
    
    def line_recombination(self, mating_pool, n_offspring=2):
        offspring =  np.zeros((n_offspring, len(mating_pool[0])))

        for individual in offspring:
            alpha = np.random.uniform(-0.25, 1.25)
            for i in range(len(individual)):
                individual[i] = mating_pool[0][i] + alpha * (mating_pool[1][i] - mating_pool[0][i])
        return offspring
    
    def limits(self, weight):
        if weight > 1:
            return 1
        if weight < -1:
            return -1
        return weight
    
    def tournament(self):
        '''
        Implements the tournament selection algorithm. 
        It draws randomly with replacement k individuals and returns the fittest individual.
        '''

        # Select k random indexes from the population
        k_indexes = np.random.randint(1, self.num_agents + 1, self.k)

        # # Extract the agent_ids corresponding to the selected indexes
        # selected_agents = [self.grid.agents[index] for index in k_indexes]
        # print(selected_agents)
        winner = max(k_indexes, key=lambda key: self.agent_dict[key])
        return self.grid.agents[winner]

    def mutate(self, agent):

        # Mutates the offspring
        for i in range(len(agent)):
            if np.random.uniform() <= self.mutation_probability:
                agent[i] = np.random.uniform(-1, 1)
                # individual[i] -= np.random.normal(1, self.mutation_stepsize)
        return agent

    def reproduce(self):
        total_offspring = []
        for reproduction in range(int(self.num_agents / 2)):
            if np.random.rand() <= 0.05:
                n_offspring = 2
            else:
                n_offspring = 1
    
            # Make mating pool according to tournament selection
            mating_pool = np.array([self.tournament() for _ in range(2)])

            parent_1, parent_2 = mating_pool[0], mating_pool[1]
            offspring = self.line_recombination([parent_1.network, parent_2.network], n_offspring)

            for new_agent in offspring:
                # Mutates and ensures no weight is outside the range [-1, 1]
                new_agent = self.mutate(new_agent)
                # new_agent = [self.limits(weight) for weight in new_agent]
                total_offspring.append(new_agent)
                if n_offspring == 1:
                    total_offspring.append(mating_pool[np.argmax(parent.fitness for parent in mating_pool)].network)
        
        # Kill old generation
        agents = list(self.grid.agents.values())
        for agent in agents:
            agent.die()

        # Place new generation
        for agent_id, new_agent in enumerate(total_offspring):
            self.make_agent(agent_id + 1, network=new_agent)

    def generate_lifetime(self):
        """
        Generate a lifetime based on the given distribution.
        """
        if self.lifetime_distribution == 'gamma':
            scale = self.lifetime_std**2 / self.lifetime_mean
            shape = self.lifetime_mean / scale
            return int(gamma.rvs(a=shape, scale=scale, size=1)[0])
        elif self.lifetime_distribution == 'lognormal':
            sigma = np.sqrt(np.log(1 + (self.lifetime_std / self.lifetime_mean)**2))
            mu = np.log(self.lifetime_mean) - sigma**2 / 2
            return int(lognorm.rvs(s=sigma, scale=np.exp(mu), size=1)[0])

    def make_agent(self, agent_id, network=None, position=None):
        """
        Note that on a grid cell, there can now only be 1 agent!
        """ 
        
        mask_agent = self.grid.agent_matrix == 0
        mask_house = self.grid.house_matrix == 0

        common_zeros = np.where(mask_agent & mask_house)

        if common_zeros[0].size > 0 and not position:
            random_index = np.random.choice(len(common_zeros[0]))
            position = (common_zeros[0][random_index], common_zeros[1][random_index])
        
        # Sample wealth
        if self.wealth_distribution:
            middle_values = self.wealth_distribution['middle_values']
            normalized_percentages = self.wealth_distribution['normalized_percentages']
            wealth = np.random.choice(middle_values, p=normalized_percentages)
        else:
            wealth = 0  
            
        self.initial_wealth.append(wealth)
        
        agent = Agent(self, agent_id, position, self.grid, self.market, creation_time=self.t, wealth = wealth, income_per_timestep=self.income_per_timestep, network=network)
        self.grid.agents[agent_id] = agent
        self.grid.agent_matrix[position] = agent_id
        self.agent_dict[agent_id] = 999999999

    def get_random_position(self):
        x = np.random.randint(self.grid.width)
        y = np.random.randint(self.grid.height)
        return (x, y)

    def timestep(self):
        self.t += 1
        agents = list(self.grid.agents.values())
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
        if self.t % self.tax_period == 0:
            self.tax_policy.apply_taxes()
        self.grid.update_house_incomes()
        
        # Update market prices
        self.market.update_price()

        
    def run(self, show_time=False):
        epochs = 25


        self.show_time = show_time
        self.tax_policy = TaxPolicy(self.grid, self)
        with open('agents.csv', 'w') as file:
            self.writer = csv.writer(file)
            for epoch in range(epochs):

                t_max = self.generate_lifetime()
                for t in range(t_max):
                    if self.show_time:
                        print(f"\nTimestep {t+1}:")
                    self.timestep()
                for agent in self.grid.agents.values():
                    agent.update_fitness()
                
                print(f"Success rate: {1-(self.action_failure / self.num_agents / t_max)}")
                print(f'{np.around((self.build+self.failed_build) / self.num_agents / t_max, 3)}, {np.around((self.gathering+self.failed_gathering) / self.num_agents / t_max, 3)}, {np.around((self.moving+self.failed_moving) / self.num_agents / t_max, 3)}, {np.around((self.buy+self.failed_buy) / self.num_agents / t_max, 3)}, {np.around((self.sell+self.failed_sell) / self.num_agents / t_max, 3)}')
                self.action_failure = 0
                self.moving = 0
                self.failed_moving = 0
                self.gathering = 0
                self.failed_gathering = 0
                self.buy = 0
                self.failed_buy = 0
                self.sell = 0
                self.failed_sell = 0
                self.build = 0
                self.failed_build = 0
                self.reproduce()
                print(f"########################################## NEW EPOCH {epoch + 1} ############################################")


        # Save the results to a CSV file after the simulation
        # self.save_results(self.save_file_path)

    def initialize_resources(self):
        """
        Initialize resources on the grid. This method places resources randomly on the grid.
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
        # print(f"Number of wood resources: {num_wood}")
        # print(f"Number of stone resources: {num_stone}")

        for _ in range(self.resource_spawn_rate * (self.num_resources - num_stone)):
            stone_position = self.get_random_position()
            if self.grid.if_no_houses(stone_position):
                self.grid.resource_matrix_stone[stone_position] += 1

        for _ in range(self.resource_spawn_rate * (self.num_resources - num_wood)):
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

        # Plot wealth over time
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            plt.plot(agent_data['timestep'], agent_data['wealth'], label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Wealth')
        plt.title('Wealth Over Time')
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
        plt.tight_layout()
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
        total_discounted_welfare_change_values = list(self.total_discounted_welfare_change.values())
        plt.plot(timesteps, total_discounted_welfare_change_values)
        plt.xlabel('Timesteps')
        plt.ylabel('Total Discounted Welfare Change')
        plt.title('Total Discounted Welfare Change Over Time')
        plt.grid(True)
        plt.show()
