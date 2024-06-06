import numpy as np

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = {}
        self.agent_matrix = np.zeros([self.width, self.height])
          # Adding resource matrices for wood and stone
        self.resource_matrix_wood = np.zeros([self.width, self.height])
        self.resource_matrix_stone = np.zeros([self.width, self.height])


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
        """
        Check if a position is empty for both agents and resources.
        """
        return (self.agent_matrix[position] == 0 and
                self.resource_matrix_wood[position] == 0 and
                self.resource_matrix_stone[position] == 0)
            

    
