from grid import Grid
from simulation import Simulation
from agent import Agent
import numpy as np


def test_gathering():
    # Set up the grid and simulation parameters
    width, height = 5, 5
    num_agents = 1
    n_timesteps = 10
    num_resources = 1

    # Initialize grid and simulation
    grid = Grid(width, height)
    sim = Simulation(num_agents, grid, n_timesteps, num_resources)

    # Place initial resources manually for a controlled test
    grid.resource_matrix_wood[1, 1] = 1
    grid.resource_matrix_stone[2, 2] = 1

    # Print initial resource matrices for reference
    print("Initial Wood Resource Matrix:")
    print(grid.resource_matrix_wood)
    print("Initial Stone Resource Matrix:")
    print(grid.resource_matrix_stone)

    # Run the simulation
    sim.run()

    # Print final resource matrices for reference
    print("Final Wood Resource Matrix:")
    print(grid.resource_matrix_wood)
    print("Final Stone Resource Matrix:")
    print(grid.resource_matrix_stone)

if __name__ == "__main__":
    test_gathering()
