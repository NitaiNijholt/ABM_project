import numpy as np

class Grid:
    def __init__(self, width, height, house_cost=(2, 2)):
        self.width = width
        self.height = height
        self.house_cost = house_cost
        self.agents = {}
        self.agent_matrix = np.zeros((self.width, self.height), dtype=int)
        self.resource_matrix_wood = np.zeros((self.width, self.height), dtype=int)
        self.resource_matrix_stone = np.zeros((self.width, self.height), dtype=int)
        self.house_matrix = np.zeros((self.width, self.height), dtype=int)
        self.houses = {}


    def get_neighbors(self, position):
        """
        Return list containing coordinates of neighboring cells with periodic boundary conditions.
        """

        # Periodic boundary conditions in all directions
        top = ((position[0] - 1) % self.height, position[1])
        bottom = ((position[0] + 1) % self.height, position[1])
        left = (position[0], (position[1] - 1) % self.width)
        right = (position[0], (position[1] + 1) % self.width)
        return [top, bottom, left, right]

    def if_empty(self, position):
        """
        Check if a position is empty for agents and houses.
        """
        return self.agent_matrix[position] == 0 and self.house_matrix[position] == 0

    
    def if_no_agent(self, position):
        """
        Check if a position is empty for agents.
        """
        return self.agent_matrix[position] == 0
            

