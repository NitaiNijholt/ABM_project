from grid import Grid
from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt

# Params
height = 20
width = 20
num_agents = 6
n_timesteps = 1000
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
# print("Agent matrix:")
# print(sim.grid.agent_matrix)

# Print the market
print(f"\nMarket:")
print(f"Wood to buy: {sim.market.wood_to_buy}")
print(f"Wood to sell: {sim.market.wood_to_sell}")
print(f"Updated wood rate: {sim.market.wood_rate}")
print(f"Stone to buy: {sim.market.stone_to_buy}")
print(f"Stone to sell: {sim.market.stone_to_sell}")
print(f"Updated stone rate: {sim.market.stone_rate}")

plt.figure()
plt.plot(sim.market.wood_rate_history, label='Wood rate')
plt.plot(sim.market.stone_rate_history, label='Stone rate')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Rate')
plt.title('Resource rates over time')
plt.show()

plt.figure()
plt.plot(np.cumsum(sim.market.wood_buy_history), label='Wood buy')
plt.plot(np.cumsum(sim.market.wood_sell_history), label='Wood sell')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Amount')
plt.title('Resource transactions over time')
plt.show()

plt.figure()
plt.plot(np.cumsum(sim.market.stone_buy_history), label='Stone buy')
plt.plot(np.cumsum(sim.market.stone_sell_history), label='Stone sell')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Amount')
plt.title('Resource transactions over time')
plt.show()

# TODO: Why are the rates always polarized?
# TODO: Why the buys and sells are balanced for stone but not for wood?