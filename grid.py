import numpy as np

class Grid:
    def __init__(self, width, height, house_cost=(2, 2), income_per_timestep=1):
        """
        Initialize grid with given width and height.

        Parameters
        ----------
        width : int
            Width of the grid.
        height : int
            Height of the grid.
        house_cost : tuple
            Cost of building a house in terms of wood and stone.
        """
        self.width = width
        self.height = height
        self.house_cost = house_cost
        self.agents = {}
        self.agent_matrix = np.zeros((self.width, self.height), dtype=int)
        self.resource_matrix_wood = np.zeros((self.width, self.height), dtype=int)
        self.resource_matrix_stone = np.zeros((self.width, self.height), dtype=int)
        self.house_matrix = np.zeros((self.width, self.height), dtype=int)
        self.house_incomes = np.ones((self.width, self.height), dtype=int) * income_per_timestep
        # self.houses = {}

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

    def if_no_agents_houses(self, position):
        """
        Check if a position is empty for agents and houses.
        """
        return self.agent_matrix[position] + self.house_matrix[position] == 0