import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, width, height, house_cost=(2, 2), income_per_timestep=1, income_kernel=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), max_house_num=3):
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
        self.houses = {}
        self.income_kernel = income_kernel
        self.max_house_num = max_house_num

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

    def if_no_agents(self, position):
        """
        Check if a position is empty for agents.
        """
        return self.agent_matrix[position] == 0

    def if_no_houses(self, position):
        """
        Check if a position is empty for houses.
        """
        return self.house_matrix[position] == 0

    def update_house_incomes(self):
        """
        Update the income of a house at a given position.
        """
        self.house_incomes = convolve2d(self.house_matrix, self.income_kernel, mode='same', boundary='wrap') + 1

    def plot_houses(self):
        """
        Plot the houses on the grid.
        """
        plt.figure()
        plt.imshow(self.house_matrix, cmap='viridis', interpolation='nearest')
        plt.axis('off')
        plt.colorbar()
        plt.show()