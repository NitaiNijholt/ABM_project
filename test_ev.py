from grid import Grid
from simulation import Simulation


# Params
height = 50
width = 50
num_agents = 1
n_timesteps = 5
num_resources = 8
house_cost = (2, 2)  # Define the cost of building a house
wood_rate = 2  # Define the exchange rate for wood to wealth
stone_rate = 3  # Define the exchange rate for stone to wealth

# Initialize grid
grid = Grid(width, height, house_cost)

# Initialize simulation
sim = Simulation(num_agents, grid, n_timesteps=n_timesteps, num_resources=num_resources, wood_rate=wood_rate, stone_rate=stone_rate)

# Set resources and wealth
for agent in grid.agents.values():
    agent.wood = 1
    agent.stone = 2
    agent.wealth = 20

for id, agent in grid.agents.items():
    print(f'Agent {id}')
    print(f'Agent wealth: {agent.wealth}')
    print(f'Agent resources: {agent.wood}, {agent.stone}')
    print(f'Earning rate for building a house: {agent.expected_income_building()}')
    print(f'Earning rate for buying resources: {agent.expected_income_buying()}')
    print(f'Earning rate for selling resources: {agent.expected_income_selling()}')
