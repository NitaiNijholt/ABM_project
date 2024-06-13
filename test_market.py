from grid import Grid
from simulation import Simulation

# Params
height = 10
width = 10
num_agents = 4
n_timesteps = 10
num_resources = 50
house_cost = (2, 2)  # Define the cost of building a house
lifetime_mean = 50
lifetime_std = 5
wood_rate = 0.5  # Define the initial rate of wood
stone_rate = 0.5  # Define the initial rate of stone

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, lifetime_mean=lifetime_mean, lifetime_std=lifetime_std, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate, save_file_path='data/test_market.csv')

sim.run()

# Print the grid
print("House matrix:")
print(sim.grid.house_matrix)
print("Wood matrix:")
print(sim.grid.resource_matrix_wood)
print("Stone matrix:")
print(sim.grid.resource_matrix_stone)
print("Agent matrix:")
print(sim.grid.agent_matrix)

# Print the market
print(f"\nMarket:")
print(f"Wood to buy: {sim.market.wood_to_buy}")
print(f"Wood to sell: {sim.market.wood_to_sell}")
print(f"Updated wood rate: {sim.market.wood_rate}")
print(f"Stone to buy: {sim.market.stone_to_buy}")
print(f"Stone to sell: {sim.market.stone_to_sell}")
print(f"Updated stone rate: {sim.market.stone_rate}")

# Print the earning rates
for id, agent in grid.agents.items():
    print(f"\nAgent {id}:")
    print(agent.earning_rates)
