import numpy as np
from grid import Grid
from house import House
import sys
from network import Network
from numba import jit


class Agent:
    def __init__(self, sim, agent_id, position, grid, market, creation_time, 
                 wealth=0, wood=0, stone=0, income_per_timestep=1, network=None):
        """
        Initialize agent with given position, wealth, wood, stone, grid, market, life expectancy, and creation time.

        Parameters
        ----------
        sim : Simulation
            Simulation object.
        agent_id : int
            Unique identifier of the agent.
        position : tuple
            Position of the agent on the grid.
        grid : Grid
            Grid object where the agent is located.
        market : Market
            Market object where the agent can trade resources.
        creation_time : int
            Time when the agent was created.
        wealth : int
            Initial wealth of the agent.
        wood : int
            Initial number of the wood resources of the agent.
        stone : int
            Initial number of the stone resources of the agent.
        lifetime_distribution : str
            Distribution of the lifetime.
            Options: 'gamma', 'lognormal'.
        lifetime_mean : float
            Mean of the lifetime distribution.
        lifetime_std : float
            Standard deviation of the lifetime distribution.
        """
        self.agent_id = agent_id
        self.position = position
        self.initial_wealth = wealth
        self.wealth = max(wealth, 0)

        self.fitness = wealth + stone * market.stone_rate + wood * market.wood_rate
        self.wood = wood
        self.stone = stone
        self.grid = grid
        self.market = market
        self.houses = []
        self.creation_time = creation_time
        self.sim = sim
        self.income_per_timestep = income_per_timestep

        self.currently_building_timesteps = 0
        self.required_building_time = 5
        self.current_action = 'Nothing'
        self.earning_rates = {}
        self.best_position = 0

        self.wealth_over_time = []
        self.houses_over_time = []
        self.gathered_at_timesteps = []
        self.bought_at_timesteps = []
        self.sold_at_timesteps = []
        self.taxes_paid_at_timesteps = []
        self.tax_paid_last_time = 0
        self.welfare_gotten_last_time = 0
        self.income = 0  # New attribute to track income

        self.punish_move = 1
        self.punish_gather = 1
        self.punish_build = 1
        self.punish_sell = 1
        self.punish_buy = 1
        self.punish_successful_move = 1

        self.successful_actions = 0


        self.input_size = 34
        self.primary_output_size = 9
        self.hidden_size = 40
        self.n_hidden_layers = 2

        total_parameters = (
            (self.input_size * self.hidden_size) + self.hidden_size +  # Input to first hidden layer
            ((self.n_hidden_layers - 1) * (self.hidden_size * self.hidden_size + self.hidden_size)) +  # Hidden layers
            (self.hidden_size * self.primary_output_size) +  # Last hidden layer to output layer
            self.primary_output_size  # Biases for the output layer
        )


        if network is None:
            self.network = np.random.rand(total_parameters)
        else:
            self.network = network
        
        self.model = self.build_model()
        
    def build_model(self):
        model = Network(self.input_size,
                        self.hidden_size,
                        self.primary_output_size,
                        self.n_hidden_layers,
                        self.network)
    
        return model
    
    def get_inputs(self):
        initial_inputs = np.array([
            np.log1p(self.market.wood_rate),
            np.log1p(self.market.stone_rate),
            np.log1p(self.wood),
            np.log1p(self.stone),
            np.log1p(self.sim.t - self.creation_time),
            np.log1p(self.wealth),
            np.log1p(self.income),
            np.log1p(self.tax_paid_last_time),
            np.log1p(self.welfare_gotten_last_time),
            float(self.currently_building_timesteps == 0)
        ])

        # Get the positions of neighbors
        positions = self.grid.get_neighbors(self.position)

        # Extract resource matrices for the positions and convert to arrays for efficient processing
        wood_resources = np.array([self.grid.resource_matrix_wood[pos] for pos in positions])
        stone_resources = np.array([self.grid.resource_matrix_stone[pos] for pos in positions])
        house = np.array([self.grid.house_matrix[pos] for pos in positions])
        agent = np.array([self.grid.agent_matrix[pos] for pos in positions])
        income = np.array([self.grid.house_incomes[pos] for pos in positions])

        # Log transform the numeric values
        neighbor_inputs = np.concatenate([
            np.log1p(wood_resources),
            np.log1p(stone_resources),
            np.log1p(house),
            np.log1p(income),
            agent.astype(float)
        ])
        
        pos = self.position
        current_cell_inputs = np.array([
            np.log1p(self.grid.resource_matrix_wood[pos]),
            np.log1p(self.grid.resource_matrix_stone[pos]),
            np.log1p(self.grid.house_matrix[pos]),
            np.log1p(self.grid.house_incomes[pos])
        ])

        # Combine all inputs into a single array
        inputs = np.concatenate([initial_inputs, neighbor_inputs, current_cell_inputs])


        return inputs


    def update_fitness(self):
        fitness = np.sum([
            self.wealth - self.initial_wealth, 
        ]) * ((self.successful_actions  / (self.sim.t - self.creation_time)))
        self.fitness = fitness
        self.sim.agent_dict[self.agent_id] = fitness

    def move(self, direction):
        """
        Agent moves to the specified direction if the target cell is empty.
        If all neighboring cells are occupied, the agent does not move.
        """
        x, y = self.position
        if direction == 'up':
            new_position = (x, (y - 1)%self.grid.height)
            self.current_action = 'move'
        elif direction == 'down':
            new_position = (x, (y + 1)%self.grid.height)
            self.current_action = 'move'
        elif direction == 'left':
            new_position = ((x - 1)%self.grid.width, y)
            self.current_action = 'move'
        elif direction == 'right':
            new_position = ((x + 1)%self.grid.width, y)
            self.current_action = 'move'
        elif direction == 'stay':
            new_position = self.position
            self.current_action = 'stay'
        else:
            raise ValueError("Invalid direction")


        # Ensure new position is within grid bounds
        if self.grid.agent_matrix[new_position] == 0:
            # Move the agent
            self.grid.agent_matrix[self.position] = 0
            self.grid.agent_matrix[new_position] = self.agent_id
            self.position = new_position
            self.sim.moving += 1

            self.fitness *= self.punish_successful_move
        
        else:
            self.current_action = 'failed move'
            self.sim.action_failure += 1
            self.sim.failed_moving += 1
            self.fitness *= self.punish_move

    def gather(self):
        """
        Agent gathers resources from the current position.
        """

        succeeded = False
        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            self.current_action = 'gather'
            self.gathered_at_timesteps.append(1)
            succeeded = True

        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            self.gathered_at_timesteps.append(1)
            self.current_action = 'gather'
            succeeded = True
        
        if not succeeded:
            self.fitness *= self.punish_gather
            self.current_action = 'failed gather'
            self.sim.action_failure += 1
            self.sim.failed_gathering += 1
        else:
             self.sim.gathering += 1
             self.successful_actions += 1
        
        
        self.gathered_at_timesteps.append(0)

    def build_house(self):
        """
        Agent completes the construction of a house
        """
        
        self.grid.house_matrix[self.position] += 1
        house = House(self.agent_id, self.position)
        self.houses.append(house)
        if self.position not in self.grid.houses:
            self.grid.houses[self.position] = [house]
        else:
            self.grid.houses[self.position].append(house)
        # print(f"Agent {self.agent_id} completed building a house at {self.position}")

    def build(self):
        """
        Agent builds a house at the current position if the agent has enough resources and the cell is empty.
        """

        # Only build if the agent has the resources to do so
        wood_cost, stone_cost = self.grid.house_cost
        if self.wood >= wood_cost and self.stone >= stone_cost and self.grid.house_matrix[self.position] < self.grid.max_house_num:
            self.wood -= wood_cost
            self.stone -= stone_cost
            self.currently_building_timesteps = 1
            self.current_action = 'start building'
            self.sim.build += 1
            self.successful_actions += 1
        #     self.fitness += 20
        else:
            self.fitness *= self.punish_build
            self.current_action = 'failed building'
            self.sim.failed_build += 1
            self.sim.action_failure += 1

    def collect_income(self):
        """
        Agent collects income from all houses.
        """
        income_collected = sum(self.grid.house_incomes[house.position] for house in self.houses) * self.income_per_timestep
        self.wealth += income_collected
        self.income = income_collected  # Track the income

    def step(self):
        if self.currently_building_timesteps > 0:
            self.currently_building_timesteps += 1
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()
                self.current_action = 'continue building'
        else:
            inputs = np.array(self.get_inputs()).astype(np.float32).reshape(1, -1)
            outputs = self.model.forward(inputs)

            action = np.argmax(outputs)

            actions = [
                self.build,
                self.gather,
                lambda: self.move('up'),
                lambda: self.move('down'),
                lambda: self.move('left'),
                lambda: self.move('right'),
                lambda: self.move('stay'),
                self.buy,
                self.sell
            ]


            actions[action]()
            # print(f"Agent {self.agent_id} does {self.current_action}, ({action})")
        
        self.collect_income()
        self.wealth_over_time.append(self.wealth)
        self.houses_over_time.append(len(self.houses))
        self.bought_at_timesteps.append(0)
        self.sold_at_timesteps.append(0)
        self.taxes_paid_at_timesteps.append(0)
    

    def die(self):
        # if self.fitness > 0:
        #     if self.sim.writer:
        #         consolidated_data = (
        #             self.wealth_over_time +
        #             self.houses_over_time +
        #             self.gathered_at_timesteps +
        #             self.bought_at_timesteps +
        #             self.sold_at_timesteps +
        #             self.taxes_paid_at_timesteps
        #         )
        #         self.sim.writer.writerow(np.concatenate(([self.fitness], np.array(self.network))))

        del self.grid.agents[self.agent_id]
        del self.sim.agent_dict[self.agent_id]
        self.grid.agent_matrix[self.position] = 0

        for house in self.houses:
            self.grid.house_matrix[house.position] -= 1
            


    def buy(self):
        """
        Agent buys resources from the market.
        """
        # Only buy if agent has enough wealth to affort it
        wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
        stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)

        if self.wealth >= (wood_to_buy * self.market.wood_rate + stone_to_buy * self.market.stone_rate) and (wood_to_buy > 0 or stone_to_buy > 0 ):
            self.current_action = 'buy'

            self.market.add_buyer(self, wood_to_buy, stone_to_buy)
            self.sim.buy += 1
            self.successful_actions += 1
        #     self.fitness += 10
        
        else:
            self.fitness *= self.punish_buy
            self.current_action = 'failed buy'
            self.sim.action_failure += 1
            self.sim.failed_buy += 1

    def sell(self):
        """
        Agent sells all resources to the market.
        """
        if self.wood > 0 and self.stone > 0:
            self.market.add_seller(self, self.wood, self.stone)
            self.current_action = 'sell'
            self.sim.sell += 1
            self.successful_actions += 1
        else:
            self.fitness *= self.punish_sell
            self.current_action = 'failed sell'
            self.sim.action_failure += 1
            self.sim.failed_sell += 1
