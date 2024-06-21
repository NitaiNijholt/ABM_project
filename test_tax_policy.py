# Description: Test the tax policy of the simulation
from simulation import Simulation
from grid import Grid
import time


start = time.time()


height = 50
width = 50
num_agents = 100
n_timesteps = 200
num_resources = 1000
house_cost = (2, 2)  # Define the cost of building a house
lifetime_mean = 80
lifetime_std = 10

wood_rate = 1.  # Define the initial rate of wood
stone_rate = 1.  # Define the initial rate of stone
income_per_timestep = 5

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Run simulation
sim.run()


# # Initialize grid and agents
# grid = Grid(50, 50, (1000, 1000))
# sim = Simulation(num_agents=100, n_timesteps=300, grid=grid)
sim.plot_average_tax_values()
sim.plot_equality_over_time()
sim.plot_productivity_over_time()
sim.plot_social_welfare_over_time()
sim.plot_total_discounted_welfare_change_over_time()

# sim.plot_results()    

end = time.time()

print(f"Running time: {end-start} seconds")