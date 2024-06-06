from grid import Grid
from simulation import Simulation
        

# Params
height = 5
width = 5
num_agents = 3
n_timesteps = 2
num_resources = 1
house_cost = (2, 2)  # Define the cost of building a house

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