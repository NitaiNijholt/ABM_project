from grid import Grid
from simulation import Simulation
import time
import matplotlib.pyplot as plt

start = time.time()




# +
# Params
height = 100
width = 100
num_agents = 100
n_timesteps = 500
num_resources = 1500
house_cost = (10, 10)  # Define the cost of building a house
lifetime_mean = 100
lifetime_std = 20

wood_rate = 5  # Define the initial rate of wood
stone_rate = 5  # Define the initial rate of stone
income_per_timestep = 5

# Initialize grid
grid = Grid(width, height, house_cost)

filepath = 'data/test_stuff_2.csv'

dynamic_tax = True
dynamic_market = True
# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, lifetime_mean=lifetime_mean, lifetime_std=lifetime_std, stone_rate=stone_rate, income_per_timestep=income_per_timestep, save_file_path=filepath, show_time=True, dynamic_tax=dynamic_tax, dynamic_market=dynamic_market)

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
# -


