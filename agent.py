import numpy as np
from grid import Grid

class Agent:
    def __init__(self, agent_id, position, grid, wealth=0, wood=0, stone=0):
        self.agent_id = agent_id
        self.position = position
        self.wealth = wealth
        self.wood = wood
        self.stone = stone
        self.grid = grid

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



    def build_house(self):
        wood_cost, stone_cost = self.grid.house_cost
        # Allow building a house if the position is occupied by the agent but no other house is there
        if self.wood >= wood_cost and self.stone >= stone_cost and self.grid.house_matrix[self.position] == 0:
            self.wood -= wood_cost
            self.stone -= stone_cost
            self.grid.house_matrix[self.position] = 1
            print(f"Agent {self.agent_id} built a house at {self.position}")
        else:
            print(f"Agent {self.agent_id} cannot build a house. Wood: {self.wood}, Stone: {self.stone}, Position has house: {self.grid.house_matrix[self.position] == 1}")

    def step(self):
        """
        Agent performs a step: move, collect resources, and possibly build a house.
        """
        self.random_move()
        self.collect_resources()
        self.build_house()
