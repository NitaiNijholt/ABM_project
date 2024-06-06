import numpy as np
from grid import Grid
from house import House

class Agent:
    def __init__(self, agent_id, position, grid, wealth=0, wood=0, stone=0):
        self.agent_id = agent_id
        self.position = position
        self.wealth = wealth
        self.wood = wood
        self.stone = stone
        self.grid = grid
        self.houses = []

    def random_move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        """
        neighbors = self.grid.get_neighbors(self.position)
        possible_moves = [neighbor for neighbor in neighbors if (self.grid.if_no_agent(neighbor) and self.grid.house_matrix[neighbor] == 0)]

        try:
            # Change position of agent if there is an empty neighboring cell
            new_position = possible_moves[np.random.randint(len(possible_moves))]
        except ValueError:  # Catch the case where possible_moves is empty
            print(f"Agent {self.agent_id} has no possible moves")
            return

        # Move agent on the grid
        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position

    def collect_resources(self):
        """
        Agent collects resources from the current position.
        """
        print(f"Agent {self.agent_id} at position {self.position} tries to collect resources.")
        print(f"Current inventory - Wood: {self.wood}, Stone: {self.stone}")

        resources_collected = False

        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            resources_collected = True

        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            resources_collected = True

        if resources_collected:
            print(f"Collected resources succesfully! New inventory - Wood: {self.wood}, Stone: {self.stone}")
        else:
            print(f"No resources collected at position {self.position}")



    def build_house(self, income_per_timestep=1):
        wood_cost, stone_cost = self.grid.house_cost
        if self.wood >= wood_cost and self.stone >= stone_cost and self.grid.house_matrix[self.position] == 0:
            self.wood -= wood_cost
            self.stone -= stone_cost
            self.grid.house_matrix[self.position] = 1
            house = House(self.agent_id, self.position, income_per_timestep=income_per_timestep)
            self.houses.append(house)
            self.grid.houses[self.position] = house
            print(f"Agent {self.agent_id} built a house at {self.position}")

    def collect_income(self):
        income_collected = sum(house.income_per_timestep for house in self.houses)
        self.wealth += income_collected
        print(f"Agent {self.agent_id} collected {income_collected} income from {len(self.houses)} houses. Total wealth: {self.wealth}")

    def step(self):
        """
        Agent performs a step: move, collect resources, and possibly build a house.
        """
        self.random_move()
        self.collect_resources()
        self.build_house()
        self.collect_income()
