import numpy as np
from agent import Agent

class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1, num_resources=1):
        self.t = 0
        self.grid = grid
        self.num_agents = num_agents
        self.n_timesteps = n_timesteps
        self.num_resources = num_resources  # Ensure this is set

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # Initialize resources
        self.initialize_resources()

        # Prints initial state of system for debugging purposes
        print(self.grid.agent_matrix)

    def make_agent(self, agent_id):
        """
        Note that on a grid cell, there can now only be 1 agent!
        """ 
        position = self.get_random_position()

        # Safe, because of assert statement above
        while not self.grid.if_empty(position):
            position = self.get_random_position()

        agent = Agent(agent_id, position, self.grid)
        self.grid.agents[agent_id] = agent

        # Place agent on the grid
        self.grid.agent_matrix[position] = agent_id

    def get_random_position(self):
        x = np.random.randint(self.grid.width)
        y = np.random.randint(self.grid.height)
        return (x, y)

    def timestep(self):

        # Iterate over all agents
        for agent in self.grid.agents.values():
            # Move agent
            agent.random_move()
        
        # Prints state of system after timestep for debugging purposes
        print(self.grid.agent_matrix)
    
    def run(self):
        for t in range(self.n_timesteps):
            self.timestep()

    def initialize_resources(self):
        """
        Initialize resources on the grid. This method places resources randomly on the grid.
        """
        for _ in range(self.num_resources):
            wood_position = self.get_random_empty_position()
            self.grid.resource_matrix_wood[wood_position] = 1
            print(f"Placed wood at {wood_position}")

            stone_position = self.get_random_empty_position()
            self.grid.resource_matrix_stone[stone_position] = 1
            print(f"Placed stone at {stone_position}")

    def get_random_empty_position(self):
        """
        Get a random empty position on the grid that is not occupied by an agent or a resource.
        """
        position = self.get_random_position()
        while not self.grid.if_empty(position):
            position = self.get_random_position()
        return position