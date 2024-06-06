import numpy as np

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = {}
        self.agent_matrix = np.zeros([self.width, self.height])

    def get_neighbors(self, position):
        """
        Return list containing coordinates of neighboring cell
        """

        top = (position[0] - 1, position[1])
        left = (position[0], position[1] - 1)
    
        # Modulus to account for periodic boundary conditons
        bottom = ((position[0] + 1) % self.height, position[1])
        right = (position[0], (position[1] + 1) % self.width)
        return top, bottom, left, right
    
    def if_empty(self, position):
        return self.agent_matrix[position] == 0
