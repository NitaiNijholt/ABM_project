from grid import Grid
from simulation import Simulation
from market import Market
from agent import Agent


# +
# Params
height = 5
width = 5
num_agents = 1
n_timesteps = 25
num_resources = 8
house_cost = (2, 2)  # Define the cost of building a house
wood_rate = 2  # Define the exchange rate for wood to wealth
stone_rate = 3  # Define the exchange rate for stone to wealth

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Run simulation
sim.run()

# Check the expected income increase by building a house for each agent
for id, agent in sim.grid.agents.items():
    print(f"\nAgent {id} can increase {agent.expected_income_building()} income each time step by building a house")