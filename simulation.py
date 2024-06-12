import numpy as np
from agent import Agent
from grid import Grid
from market import Market
import matplotlib.pyplot as plt
import json
import csv


class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1, num_resources=0, wood_rate=1, stone_rate=1, lifetime_mean=80, resource_spawn_period=1, agent_spawn_period=10):
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
        self.resource_spawn_period = resource_spawn_period
        self.agent_spawn_period = agent_spawn_period
        self.writer = None

        # Initialize market
        self.market = Market(wood_rate, stone_rate)

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"
        
        # Load wealth distribution data
        # MAKE SURE THIS LINE IS BEFORE AGENTS ARE CREATED
        self.wealth_distribution = self.load_distribution_data('data/distribution_data.json')

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # Initialize resources
        self.initialize_resources()

    def load_distribution_data(self, file_path):
        with open(file_path, 'r') as file:
            distribution_data = json.load(file)
        return distribution_data

    def make_agent(self, agent_id):
        """
        Note that on a grid cell, there can now only be 1 agent!
        """ 
        position = self.get_random_position()

        # Safe, because of assert statement above
        while not self.grid.if_no_agents_houses(position):
            position = self.get_random_position()
        
        # Sample wealth
        if self.wealth_distribution:
            middle_values = self.wealth_distribution['middle_values']
            normalized_percentages = self.wealth_distribution['normalized_percentages']
            wealth = np.random.choice(middle_values, p=normalized_percentages)
        else:
            wealth = 0  
            
        agent = Agent(self, agent_id, position, self.grid, self.market, lifetime_mean=self.lifetime_mean, 
                      creation_time=self.t, wealth = wealth)
        self.grid.agents[agent_id] = agent
        self.grid.agent_matrix[position] = agent_id

        # Place agent on the grid
        self.grid.agent_matrix[position] = agent_id

    def get_random_position(self):
        x = np.random.randint(self.grid.width)
        y = np.random.randint(self.grid.height)
        return (x, y)

    def timestep(self):
        self.t += 1
        agents = list(self.grid.agents.values())
        for agent in agents:
            agent.step()
        if self.t % self.resource_spawn_period == 0:
            self.spawn_resources()
        
        # Buggy so commented out
        # if self.t % self.agent_spawn_period == 0:
        #     self.spawn_agents()
        
    def run(self):
        with open('agent_data.csv', mode='a', newline='') as file:
            self.writer = csv.writer(file)
            for t in range(self.n_timesteps):
                print(f"\nTimestep {t+1}:")
                self.timestep()

    def initialize_resources(self):
        """
        Initialize resources on the grid. This method places resources randomly on the grid.
        """
        for _ in range(self.num_resources):
            wood_position = self.get_random_empty_position()
            self.grid.resource_matrix_wood[wood_position] += 1

            stone_position = self.get_random_empty_position()
            self.grid.resource_matrix_stone[stone_position] += 1

    def get_random_empty_position(self):
        """
        Get a random position on the grid that is not occupied by an agent or a house.
        """
        position = self.get_random_position()
        while not self.grid.if_no_agents_houses(position):
            position = self.get_random_position()
        return position

    def spawn_resources(self):
        """
        Spawn resources on the grid randomly.
        """
        num_wood = np.sum(self.grid.resource_matrix_wood)
        num_stone = np.sum(self.grid.resource_matrix_stone)
        # print(f"Number of wood resources: {num_wood}")
        # print(f"Number of stone resources: {num_stone}")###################################################################

        for _ in range(self.num_resources - num_wood):
            wood_position = self.get_random_position()
            if self.grid.if_no_agents_houses(wood_position):
                self.grid.resource_matrix_wood[wood_position] += 1

        for _ in range(self.num_resources - num_stone):
            stone_position = self.get_random_position()
            if self.grid.if_no_agents_houses(stone_position):
                self.grid.resource_matrix_stone[stone_position] += 1

    def spawn_agents(self):
        """
        Spawn agents on the grid randomly.
        """
        current_num_agents = len(self.grid.agents)
        last_agent_id = max(self.grid.agents.keys())
        for _ in range(self.num_agents - current_num_agents):
            last_agent_id += 1
            self.make_agent(max(self.grid.agents.keys()))

    def plot_wealth_over_time(self):
        """
        Plot wealth of agents over time.
        """
        for agent_id, wealth_history in self.wealth_over_time.items():
            plt.plot(range(len(wealth_history)), wealth_history, marker='o', label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Wealth')
        plt.title('Wealth Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_houses_over_time(self):
        """
        Plot number of houses of agents over time.
        """
        for agent_id, houses_history in self.houses_over_time.items():
            plt.plot(range(len(houses_history)), houses_history, marker='o', label=f'Agent {agent_id}')
        plt.xlabel('Timesteps')
        plt.ylabel('Number of Houses')
        plt.title('Number of Houses Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()




