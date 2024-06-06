from grid import Grid
from simulation import Simulation
from agent import Agent
import numpy as np


def test_house_building():
    width, height = 5, 5
    num_agents = 1
    n_timesteps = 20
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

    print("Final Wood Resource Matrix:")
    print(grid.resource_matrix_wood)
    print("Final Stone Resource Matrix:")
    print(grid.resource_matrix_stone)
    print("House matrix:")
    print(grid.house_matrix)

    house_built = np.any(grid.house_matrix == 1)
    print("House built:", house_built)
    assert house_built, "No house was built during the simulation."

if __name__ == "__main__":
    test_house_building()

