import numpy as np

class Agent:
    def __init__(self, agent_id, position, grid, wealth=0):
        self.agent_id = agent_id
        self.position = position
        self.wealth = wealth
        self.grid = grid

    def random_move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        Should still be optimized
        """

        neighbors = self.grid.get_neighbors(self.position)
        empty_neighbors = []
        for neighbor in neighbors:
            if self.grid.if_empty(neighbor):
                empty_neighbors.append(neighbor)

        try:
            # Change position of agent if there is an empty neighboring cell
            new_position = empty_neighbors[np.random.randint(len(empty_neighbors))]
        
        except:

            # Return without moving if there is no empty neighboring cell
            return

        # Move agent on the grid
        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position