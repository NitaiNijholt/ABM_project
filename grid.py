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
        income_per_timestep : int, optional
            Income per timestep for each house (default is 1).
        income_kernel : np.ndarray, optional
            Kernel used for calculating house incomes based on neighboring houses (default is a 3x3 array with ones on the edges and center).
        max_house_num : int, optional
            Maximum number of houses an agent can build (default is 3).
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
        Return a list containing coordinates of neighboring cells with periodic boundary conditions.

        Parameters
        ----------
        position : tuple
            Coordinates (x, y) of the current cell.

        Returns
        -------
        list
            List of coordinates of neighboring cells.
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

        Parameters
        ----------
        position : tuple
            Coordinates (x, y) of the cell to check.

        Returns
        -------
        bool
            True if the position is empty for agents, False otherwise.
        """
        return self.agent_matrix[position] == 0

    def if_no_houses(self, position):
        """
        Check if a position is empty for houses.

        Parameters
        ----------
        position : tuple
            Coordinates (x, y) of the cell to check.

        Returns
        -------
        bool
            True if the position is empty for houses, False otherwise.
        """
        return self.house_matrix[position] == 0

    def update_house_incomes(self):
        """
        Update the income of houses at all positions based on neighboring houses.

        This method uses a convolution to calculate the new incomes.
        """
        self.house_incomes = convolve2d(self.house_matrix, self.income_kernel, mode='same', boundary='wrap') + 1

    def plot_houses(self):
        """
        Plot the houses on the grid using a heatmap.
        """
        plt.figure()
        plt.imshow(self.house_matrix, cmap='viridis', interpolation='nearest')
        plt.axis('off')
        plt.colorbar()
        plt.show()

