from grid import Grid
from simulation import Simulation
from market import Market
from agent import Agent


# +
# Params
height = 5
width = 5
num_agents = 3
n_timesteps = 2
num_resources = 1
house_cost = (2, 2)  # Define the cost of building a house
wood_rate = 2  # Define the exchange rate for wood to wealth
stone_rate = 3  # Define the exchange rate for stone to wealth

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Run simulation
sim.run()
# -

# Initialize grid and simulation
grid = Grid(width, height, house_cost)
sim = Simulation(num_agents, grid, n_timesteps, num_resources)

# Run simulation
sim.run()

# Print resource matrices for debugging purposes
print("Wood resource matrix:")
print(grid.resource_matrix_wood)
print("Stone resource matrix:")
print(grid.resource_matrix_stone)
print("House matrix:")
print(grid.house_matrix)

# +
market = Market(wood_rate=2, stone_rate=3)

# Manually create an agent and place it on the grid
agent_id = 5
initial_position = (2, 2)
agent = Agent(agent_id, initial_position, grid, market, wealth=50, wood=10, stone=5)

# Agent performing steps including trading
agent.step()
