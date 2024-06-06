from grid import Grid
from simulation import Simulation
        


height = 2
width = 2
grid = Grid(width, height)


num_agents = 3
n_timesteps = 2
sim = Simulation(num_agents, grid, n_timesteps)

sim.run()