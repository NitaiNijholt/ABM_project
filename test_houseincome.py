from grid import Grid
from simulation import Simulation
from agent import Agent
import numpy as np
from house import House


def test_income_from_houses():
    width, height = 5, 5
    num_agents = 2
    n_timesteps = 20  # Increased timesteps for better visualization
    num_resources = 20
    house_cost = (2, 2)

    grid = Grid(width, height, house_cost)
    sim = Simulation(num_agents, grid, n_timesteps, num_resources)

    grid.resource_matrix_wood[1, 1] = 2
    grid.resource_matrix_stone[2, 2] = 2

    print("Initial Wood Resource Matrix:")
    print(grid.resource_matrix_wood)
    print("Initial Stone Resource Matrix:")
    print(grid.resource_matrix_stone)

    sim.run()

    house_built = np.any(grid.house_matrix == 1)
    assert house_built, "No house was built during the simulation."

    initial_wealth = sim.grid.agents[1].wealth
    sim.run()  # Run additional timesteps to collect income
    final_wealth = sim.grid.agents[1].wealth

    print("Initial wealth:", initial_wealth)
    print("Final wealth:", final_wealth)

    assert final_wealth > initial_wealth, "Agent did not collect income from the house."

    sim.plot_wealth_over_time()
    sim.plot_houses_over_time()

if __name__ == "__main__":
    test_income_from_houses()
