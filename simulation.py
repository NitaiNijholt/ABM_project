import numpy as np
from agent import Agent
from grid import Grid
from market import Market
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1, num_resources=1, wood_rate=1, stone_rate=1, life_expectancy=80, resource_spawn_period=1, agent_spawn_period=1):
        self.t = 0
        self.grid = grid
        self.num_agents = num_agents
        self.n_timesteps = n_timesteps
        self.num_resources = num_resources
        self.wealth_over_time = {agent_id: [] for agent_id in range(1, num_agents + 1)}
        self.houses_over_time = {agent_id: [] for agent_id in range(1, num_agents + 1)}
        self.life_expectancy = life_expectancy
        self.resource_spawn_period = resource_spawn_period
        self.agent_spawn_period = agent_spawn_period

        # Initialize market
        self.market = Market(wood_rate, stone_rate)

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # Initialize resources
        self.initialize_resources()
        print('resources initialized')

    def make_agent(self, agent_id):
        """
        Note that on a grid cell, there can now only be 1 agent!
        """ 
        position = self.get_random_position()

        # Safe, because of assert statement above
        while not self.grid.if_no_agent(position):
            position = self.get_random_position()

        agent = Agent(self, agent_id, position, self.grid, self.market, life_expectancy=self.life_expectancy, creation_time=self.t)
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
            self.wealth_over_time[agent.agent_id].append(agent.wealth)
            self.houses_over_time[agent.agent_id].append(len(agent.houses))
        if self.t % self.resource_spawn_period == 0:
            self.spawn_resources()
        if self.t % self.agent_spawn_period == 0:
            self.spawn_agents()
        
    def run(self):
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
            print(f"Placed wood at {wood_position}")

            stone_position = self.get_random_empty_position()
            self.grid.resource_matrix_stone[stone_position] += 1
            print(f"Placed stone at {stone_position}")

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
        Spawn resources on the grid randomly. If the new resource position is already
        occupied, the resource will not be placed to keep resources density constant.
        """
        num_wood = np.sum(self.grid.resource_matrix_wood)
        num_stone = np.sum(self.grid.resource_matrix_stone)
        print(f"Number of wood resources: {num_wood}")
        print(f"Number of stone resources: {num_stone}")

        for _ in range(self.num_resources - num_wood):
            wood_position = self.get_random_position()
            if self.grid.if_empty(wood_position):
                self.grid.resource_matrix_wood[wood_position] = 1

        for _ in range(self.num_resources - num_stone):
            stone_position = self.get_random_position()
            if self.grid.if_empty(stone_position):
                self.grid.resource_matrix_stone[stone_position] = 1

    def spawn_agents(self):
        """
        Spawn agents on the grid randomly.
        """
        current_num_agents = len(self.grid.agents)
        last_agent_id = max(self.grid.agents.keys())
        for _ in range(self.num_agents - current_num_agents):
            last_agent_id += 1
            self.make_agent(last_agent_id)

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




