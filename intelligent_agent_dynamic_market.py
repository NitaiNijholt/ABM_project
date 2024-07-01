import numpy as np
from grid import Grid
from house import House
import sys
from network import Network
from numba import jit
from agent import Agent


class IntelligentAgent(Agent):
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

        super().__init__(sim, agent_id, position, grid, market, creation_time, 
                 wealth, wood, stone, income_per_timestep=income_per_timestep)


        self.fitness = wealth + stone * market.stone_rate + wood * market.wood_rate

        self.tax_paid_last_time = 0
        self.welfare_gotten_last_time = 0
        self.initial_wealth = wealth

        self.punish_move = 1
        self.punish_gather = 1
        self.punish_build = 1
        self.punish_sell = 1
        self.punish_buy = 1
        self.punish_successful_move = 1

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
        
        self.order_limit = 1
        self.order_cooldown_period = 5
        self.orders_placed = 0
        self.last_order_time = -self.order_cooldown_period
        
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
        ])
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
        else:
            raise ValueError("Invalid direction")


        # Ensure new position is within grid bounds
        if self.grid.agent_matrix[new_position] == 0 or new_position==self.position:
            # Move the agent
            self.grid.agent_matrix[self.position] = 0
            self.grid.agent_matrix[new_position] = self.agent_id
            self.position = new_position
            # self.sim.moving += 1

    def gather(self):
        """
        Agent gathers resources from the current position.
        """

        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            self.current_action = 'gather'
            self.gathered_at_timesteps.append(1)
            self.sim.gathering += 1

        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            self.gathered_at_timesteps.append(1)
            self.current_action = 'gather'
            self.sim.gathering += 1
        
        
        self.gathered_at_timesteps.append(0)


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
            self.current_action = 'start_building'
            self.sim.build += 1


    def step(self):
        if self.currently_building_timesteps > 0:
            self.currently_building_timesteps += 1
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()
                self.current_action = 'continue_building'
        else:
            inputs = np.array(self.get_inputs()).astype(np.float32).reshape(1, -1)
            outputs = self.model.forward(inputs)

            # Sort actions based on output values in descending order
            sorted_actions = np.argsort(outputs[0])[::-1]

            
            # Iterate through actions in the order of their desirability
            for action in sorted_actions:
                if action == 0:
                    wood_cost, stone_cost = self.grid.house_cost
                    if self.wood >= wood_cost and self.stone >= stone_cost and self.grid.house_matrix[self.position] < self.grid.max_house_num:
                        self.build()
                        break
                elif action == 1:
                    if self.grid.resource_matrix_wood[self.position] > 0 or self.grid.resource_matrix_stone[self.position] > 0:
                        self.gather()
                        break
                elif action == 7:
                    wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
                    stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)
                    if self.can_place_order() and self.wealth >= (wood_to_buy * self.market.wood_rate + stone_to_buy * self.market.stone_rate) and (wood_to_buy > 0 or stone_to_buy > 0):
                        if self.buy():
                            break
                elif action == 8:
                    if self.can_place_order() and self.wood > 0 or self.stone > 0:
                        self.sell()
                        break
                else:
                    x, y = self.position
                    if action == 2 and self.grid.agent_matrix[(x, (y - 1) % self.grid.height)] == 0:
                        self.move('up')
                        break
                    elif action == 3 and self.grid.agent_matrix[(x, (y + 1) % self.grid.height)] == 0:
                        self.move('down')
                        break
                    elif action == 4 and self.grid.agent_matrix[((x - 1) % self.grid.width, y)] == 0:
                        self.move('left')
                        break
                    elif action == 5 and self.grid.agent_matrix[((x + 1) % self.grid.width, y)] == 0:
                        self.move('right')
                        break
                    elif action == 6:
                        self.move('stay')
                        break
                

            # actions[action]()
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
        determined_price_wood = self.determine_price('buy', 'wood')
        determined_price_stone = self.determine_price('buy', 'stone')

        if determined_price_stone and determined_price_wood:
            wood_price = min(determined_price_wood, self.wealth * wood_to_buy / (wood_to_buy + stone_to_buy))
            stone_price = min(determined_price_stone, self.wealth * stone_to_buy / (wood_to_buy + stone_to_buy))

            if self.wealth >= (wood_to_buy * wood_price + stone_to_buy * stone_price) and (wood_to_buy > 0 or stone_to_buy > 0 ):
                self.current_action = 'buy'

                self.place_order(self.order_books, 'wood', 'buy', price = wood_price, quantity = wood_to_buy)
                self.place_order(self.order_books, 'stone', 'buy', price = stone_price, quantity = stone_to_buy)

                self.sim.buy += 1
                return True
        return False


    def sell(self):
        """
        Agent sells all resources to the market.
        """
        
        if self.wood > 0 or self.stone > 0:
            wood_price = self.determine_price('sell', 'wood')
            stone_price = self.determine_price('sell', 'stone')
            if wood_price and stone_price:
                self.current_action = 'sell'
                self.sim.sell += 1
                self.place_order(self.order_books, 'wood', 'sell', price = wood_price, quantity = self.wood)
                self.place_order(self.order_books, 'stone', 'sell', price = stone_price, quantity = self.stone)


    def can_place_order(self):
        """
        Check if the agent can place an order based on the order limit and cooldown period.
        """
        current_timestep = self.sim.t
        if self.orders_placed < self.order_limit:
            return True
        if current_timestep - self.last_order_time >= self.order_cooldown_period:
            self.orders_placed = 0
            return True
        return False

    def place_order(self, order_books, resource_type, order_type, price, quantity):
        """
        Place a buy or sell order in the order book (market).

        Parameters
        ----------
        order_books : dict
            A dictionary containing the order books for different resources.
            Example: {'wood': wood_order_book, 'stone': stone_order_book}
        resource_type : str
            The type of resource to trade ('wood' or 'stone').
        order_type : str
            The type of order to place ('buy' or 'sell').
        price : float
            The price at which to place the order.
        quantity : int
            The amount of the resource to trade.
        """

        if resource_type not in order_books:
            raise ValueError(f"Invalid resource type: {resource_type}")

        order_book = order_books[resource_type]

        if order_type == 'buy':
            for _ in range(quantity):
                order_book.place_bid(self.agent_id, price)
        elif order_type == 'sell':
            for _ in range(quantity):
                order_book.place_ask(self.agent_id, price)
        else:
            raise ValueError(f"Invalid order type: {order_type}")

        self.orders_placed += 1
        self.last_order_time = self.sim.t

        # # Log the order placement for debugging
        # print(f"Agent {self.agent_id} placed a {order_type} order for {quantity} units of {resource_type} at price {price}.")


        
    def determine_price(self, order_type, resource):
        age = self.sim.t - self.creation_time
        total_income = max(self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time), 0)
        if total_income == 0:
            return None
        earning_rate = total_income / self.required_building_time
        m_price = earning_rate / sum(self.grid.house_cost) #* 10**(np.random.normal(0, 0.01)), 1)

        ob_price_wood_bid, ob_price_wood_ask = self.sim.wood_order_book.check_price()
        ob_price_stone_bid, ob_price_stone_ask = self.sim.stone_order_book.check_price()

        if resource == 'wood':
            ob_price_bid, ob_price_ask = ob_price_wood_bid, ob_price_wood_ask
        elif resource == 'stone':
            ob_price_bid, ob_price_ask = ob_price_stone_bid, ob_price_stone_ask
        else:
            return None

        if order_type == 'buy':
            if ob_price_ask is not None:
                self.market.wood_rate = min(ob_price_ask, m_price) if resource == 'wood' else self.market.stone_rate
                self.market.stone_rate = min(ob_price_ask, m_price) if resource == 'stone' else self.market.wood_rate
            else:
                self.market.wood_rate = m_price if resource == 'wood' else self.market.stone_rate
                self.market.stone_rate = m_price if resource == 'stone' else self.market.wood_rate
        elif order_type == 'sell':
            if ob_price_bid is not None:
                self.market.wood_rate = max(ob_price_bid, m_price) if resource == 'wood' else self.market.stone_rate
                self.market.stone_rate = max(ob_price_bid, m_price) if resource == 'stone' else self.market.wood_rate
            else:
                self.market.wood_rate = m_price if resource == 'wood' else self.market.stone_rate
                self.market.stone_rate = m_price if resource == 'stone' else self.market.wood_rate

        return self.market.wood_rate if resource == 'wood' else self.market.stone_rate
