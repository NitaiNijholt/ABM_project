import numpy as np
from agent import Agent


class Simulation:
    def __init__(self, num_agents, grid, n_timesteps=1):
        self.t = 0
        self.grid = grid
        self.num_agents = num_agents
        self.n_timesteps = n_timesteps

        assert num_agents <= self.grid.width * self.grid.height, "Number of agents cannot be larger than gridpoints"

        # Create number of agents
        for agent_id in range(1, num_agents + 1):
            self.make_agent(agent_id)
        
        # Prints inital state of system for debugging purposes
        print(self.grid.agent_matrix)

    def make_agent(self, agent_id):
        """
        Note that on a gridcell, there can now only be 1 agent!
        """ 

        position = self.get_random_position()

        # Safe, becasue of assert statement above
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
