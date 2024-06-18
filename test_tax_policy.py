# Description: Test the tax policy of the simulation
from simulation import Simulation
from grid import Grid

# Initialize grid and agents
grid = Grid(4, 4, (2, 2))
sim = Simulation(num_agents=2, n_timesteps=100, grid=grid)
sim.run(show_time=True)
sim.plot_average_tax_values()
sim.plot_equality_over_time()
sim.plot_productivity_over_time()
sim.plot_social_welfare_over_time()
sim.plot_total_discounted_welfare_change_over_time()

# sim.plot_results()