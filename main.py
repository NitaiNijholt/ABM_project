from grid import Grid
from simulation_evolve import Simulation
import time
import matplotlib.pyplot as plt

start = time.time()


# +
# Params
height = 80
width = 80
num_agents = 500
n_timesteps = 100
num_resources = 15000
house_cost = (2, 2)  # Define the cost of building a house
lifetime_mean = 50
lifetime_std = 1

wood_rate = 2  # Define the initial rate of wood
stone_rate = 2  # Define the initial rate of stone
income_per_timestep = 1

# Initialize grid
grid = Grid(width, height, house_cost)

filepath = 'data/test_stuff_2.csv'

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate, income_per_timestep=income_per_timestep, save_file_path=filepath, show_time=True)

# Run simulation
sim.run()

end = time.time()

print(f"Running time: {end-start} seconds")

plt.figure()
plt.plot(sim.market.wood_rate_history, label='Wood rate')
plt.plot(sim.market.stone_rate_history, label='Stone rate')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Rate')
plt.title('Resource rates over time')
plt.show()

# print(grid.house_matrix)

sim.plot_results()


# # Check the expected income increase by building a house for each agent
# for id, agent in sim.grid.agents.items():
#     print(f"\nAgent {id} can get {agent.expected_income_building()} income by building a house")
#     print(f"Agent {id} can get {agent.expected_income_buying()} income by buying resources")
#     print(f"Agent {id} can get {agent.expected_income_selling()} income by selling resources")