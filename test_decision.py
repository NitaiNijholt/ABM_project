from grid import Grid
from simulation import Simulation

# Params
height = 4
width = 4
num_agents = 2
n_timesteps = 10
num_resources = 10
house_cost = (2, 2)  # Define the cost of building a house
wood_rate = 2  # Define the exchange rate for wood to wealth
stone_rate = 3  # Define the exchange rate for stone to wealth

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Test decision making
for t in range(n_timesteps):
    print("-"*50)
    print(f"Timestep {t+1}:")
    for id, agent in grid.agents.items():
        print(f"\nAgent {id}:")
        print(f"Wood: {agent.wood}")
        print(f"Stone: {agent.stone}")
        print(f"Wealth: {agent.wealth}")
        print(f"House: {agent.houses}")
        print(f"Action: {agent.current_action}")
    print("\nAgent matrix:")
    print(grid.agent_matrix)
    print("Wood matrix:")
    print(grid.resource_matrix_wood)
    print("Stone matrix:")
    print(grid.resource_matrix_stone)
    print("House matrix:")
    print(grid.house_matrix)
    print()
    sim.timestep()