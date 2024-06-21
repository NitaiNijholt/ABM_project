from grid import Grid
from simulation import Simulation
import time

start = time.time()


# +
# Params
height = 20
width = 20
num_agents = 50


n_timesteps = 1000
num_resources = 800
house_cost = (2,2)  # Define the cost of building a house
wood_rate = 2  # Define the exchange rate for wood to wealth
stone_rate = 2  # Define the exchange rate for stone to wealth

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Run simulation
sim.run()

print(grid.house_matrix)

sim.plot_results()


end = time.time()

print(f"Running time: {end-start} seconds")

# # Check the expected income increase by building a house for each agent
# for id, agent in sim.grid.agents.items():
#     print(f"\nAgent {id} can get {agent.expected_income_building()} income by building a house")
#     print(f"Agent {id} can get {agent.expected_income_buying()} income by buying resources")
#     print(f"Agent {id} can get {agent.expected_income_selling()} income by selling resources")