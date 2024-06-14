import numpy as np
from scipy.stats import gamma, lognorm
from grid import Grid
from house import House

# +
class Agent:
    def __init__(self, sim, agent_id, position, grid, market, creation_time, 
                 wealth=0, wood=0, stone=0, lifetime_distribution='gamma', lifetime_mean=80, lifetime_std=10, income_per_timestep=1):
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
        self.wealth = wealth
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

        self.wealth_over_time = []
        self.houses_over_time = []
        self.gathered_at_timesteps = []
        self.bought_at_timesteps = []
        self.sold_at_timesteps = []
        self.taxes_paid_at_timesteps = []

        assert lifetime_distribution in ['gamma', 'lognormal'], 'Invalid lifetime distribution'

        # Generate the actual lifetime
        self.actual_lifetime = self.generate_lifetime(lifetime_distribution, lifetime_mean, lifetime_std)

        # Guess the lifetime
        self.guessed_lifetime = self.generate_lifetime(lifetime_distribution, lifetime_mean, lifetime_std)

        # TODO: Do we need this?
        # Current objective of the agent. Now, 'Trade' is the only thing that does something
        self.objective = 'Trade'

        # Exponential factor indicating how determined the agent is in his objective. 0 is equivalent to no objective, and higher values lead to smaller and smaller probabilities that the agent does something which delays achieving the objective
        self.objective_alpha = 1

    def generate_lifetime(self, lifetime_distribution, lifetime_mean, lifetime_std):
        """
        Generate a lifetime based on the given distribution.
        """
        if lifetime_distribution == 'gamma':
            
            # mean = shape * scale, var = shape * scale^2
            scale = lifetime_std**2 / lifetime_mean
            shape = lifetime_mean / scale
            return int(gamma.rvs(a=shape, scale=scale, size=1)[0])
        elif lifetime_distribution == 'lognormal':
            # mean = exp(mu + sigma^2/2), var = (exp(sigma^2) - 1) * exp(2 * mu + sigma^2)
            sigma = np.sqrt(np.log(1 + (lifetime_std / lifetime_mean)**2))
            mu = np.log(lifetime_mean) - sigma**2 / 2
            return int(lognorm.rvs(s=sigma, scale=np.exp(mu), size=1)[0])
        
    def find_moves(self):
        """
        Finds all possible moves and returns them
        """
        neighbors = self.grid.get_neighbors(self.position)
        return [neighbor for neighbor in neighbors if self.grid.if_no_agents_houses(neighbor)]

    def get_direction_to_position(self, position, normalized=True):
        """
        Returns (normalized) direction from current position to another position on the grid
        """

        direction = [position[0] - self.position[0], position[1] - self.position[1]]
        if normalized:
            magnitude = np.sqrt(direction[0]**2 + direction[1]**2)
            return [direction[0] / magnitude, direction[1] / magnitude]
        return direction
    
    def get_random_move_in_direction(self, direction, possible_moves):
        """
        Calculated scores of a move based on the desired direction, and uses those scores as weights in a weighted random choice to find a move
        """
        
        move_scores = []
        for move in possible_moves:
            move_direction = [move[0] - self.position[0], move[1] - self.position[1]]
            score = np.dot(direction, move_direction)
            move_scores.append(score)
        
        # Adjust scores to be non-negative by adding the absolute value of the smallest score
        min_score = min(move_scores)
        move_scores = [1 + (score - min_score) ** self.objective_alpha for score in move_scores]
        
        # Use random.choices to select a move based on the scores as weights
        return possible_moves[np.random.choice(len(possible_moves), p=np.array(move_scores)/sum(move_scores))]

    def move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        """

        possible_moves = self.find_moves()
        # print(f"Agent {self.agent_id} at position {self.position} tries to move. Possible moves: {possible_moves}")
        if len(possible_moves) == 0:
            # print(f"Agent {self.agent_id} has no possible moves")#############################################################################
            return

        # Move to a random neighboring cell
        new_position = possible_moves[np.random.randint(len(possible_moves))]

        # Move agent on the grid
        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position

    def gather(self):
        """
        Agent gathers resources from the current position.
        """
        # print(f"Agent {self.agent_id} at position {self.position} tries to collect resources.")
        # print(f"Current inventory - Wood: {self.wood}, Stone: {self.stone}")###########################################################################

        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            self.gathered_at_timesteps.append(1)
            # print(f"Collected resources succesfully! New inventory - Wood: {self.wood}, Stone: {self.stone}")#########################################################

        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            self.gathered_at_timesteps.append(1)
            # print(f"Collected resources succesfully! New inventory - Wood: {self.wood}, Stone: {self.stone}")#########################################################

        # print(f"No resources collected at position {self.position}")###############################################################################
        self.gathered_at_timesteps.append(0)

    def build_house(self):
        """
        Agent completes the construction of a house
        """ 

        self.grid.house_matrix[self.position] = 1
        house = House(self.agent_id, self.position, income_per_timestep=self.income_per_timestep)
        self.houses.append(house)
        self.grid.houses[self.position] = house
        # print(f"Agent {self.agent_id} completed building a house at {self.position}")#################################################################

    def build(self):
        """
        Agent builds a house at the current position if the agent has enough resources and the cell is empty.
        """
        wood_cost, stone_cost = self.grid.house_cost
        self.wood -= wood_cost
        self.stone -= stone_cost
        # print(f"Agent {self.agent_id} starts building a house at {self.position}")##################################################################################
        self.currently_building_timesteps = 1
        
    def collect_income(self):
        """
        Agent collects income from all houses.
        """
        income_collected = sum(house.income_per_timestep for house in self.houses)
        self.wealth += income_collected
        # print(f"Agent {self.agent_id} collected {income_collected} income from {len(self.houses)} houses. Total wealth: {self.wealth}")#####################################

    def step(self):
        """
        Agent performs a step: move, collect resources, possibly build a house, and trade.
        """
        # print(f"Agent {self.agent_id} decides to {self.current_action}. Current wealth: {self.wealth}, Wood: {self.wood}, Stone: {self.stone}, Lifetime: {self.actual_lifetime - (self.sim.t - self.creation_time)}")

        # If the agent is currently building
        if self.currently_building_timesteps > 0:
            # print(f"Agent is currently building a house and cannot do other things. Currently building {self.currently_building_timesteps}/{self.required_building_time}")########################

            self.currently_building_timesteps += 1

            # If the agent just now completed building:
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()

        # Agent can do other things if he is not building
        else:
            # Agent decides to build, buy, sell, gather or move
            actions = [self.move, self.build, self.buy, self.sell, self.gather]
            self.earning_rates = {'move': self.earning_rate_random_moving(),
                                'build': self.earning_rate_building(),
                                'buy': self.earning_rate_buying(),
                                'sell': self.earning_rate_selling(),
                                'gather': self.earning_rate_gathering()}
            action = actions[np.argmax(list(self.earning_rates.values()))]
            # print(f"Agent {self.agent_id} decides to {action.__name__} based on earning rates: {self.earning_rates}")

            # Update the current_action based on the action taken
            self.current_action = action.__name__
            
            # Perform the action
            action()

        # Agent can always collect
        self.collect_income()

        # TODO: Update data collection based on the action taken, and also consider the tax
        # Data collection
        self.wealth_over_time.append(self.wealth)   # Wealth before tax
        self.houses_over_time.append(len(self.houses))
        self.bought_at_timesteps.append(0)
        self.sold_at_timesteps.append(0)
        self.taxes_paid_at_timesteps.append(0)

        # Agent dies when he reaches his life expectancy
        if self.sim.t >= self.creation_time + self.actual_lifetime:
            self.die()
    
    def die(self):
        if self.sim.writer:
            consolidated_data = (
                self.wealth_over_time +
                self.houses_over_time +
                self.gathered_at_timesteps +
                self.bought_at_timesteps +
                self.sold_at_timesteps +
                self.taxes_paid_at_timesteps
                )
            self.sim.writer.writerow(consolidated_data)
        self.sim.make_agent(max(self.grid.agents.keys()) + 1)
        del self.grid.agents[self.agent_id]
        self.grid.agent_matrix[self.position] = 0
        # print(f"Agent {self.agent_id} died at the age of {self.actual_lifetime}")#################################################

        
#     def trade(self, wood_to_trade=0, stone_to_trade=0):
#         """
#         Agent trades resources at the market.
        
#         Currently: If agent has any resources at all, go to market and trade
#         TODO: Walking to the market takes multiple timesteps if far away. Now agents teleport
#         TODO: Add utility function such that it determines how and when agent will trade
#         """

#         if wood_to_trade > 0:
#             if wood_to_trade <= self.wood:
#                 wealth_received = self.market.trade_wood_for_wealth(wood_to_trade)
#                 self.wood -= wood_to_trade
#                 self.wealth += wealth_received
#                 # print(f"Agent {self.agent_id} traded {wood_to_trade} wood for {wealth_received} wealth")######################################
#             # else:
#                 # print(f"Agent {self.agent_id} does not have enough wood to trade")################################################################

#         if stone_to_trade > 0:
#             if stone_to_trade <= self.stone:
#                 wealth_received = self.market.trade_stone_for_wealth(stone_to_trade)
#                 self.stone -= stone_to_trade
#                 self.wealth += wealth_received
#             #     print(f"Agent {self.agent_id} traded {stone_to_trade} stone for {wealth_received} wealth")
#             # else:
#             #     print(f"Agent {self.agent_id} does not have enough stone to trade")#########################################################################
        
#         self.objective = 'Nothing'

    # Random trading strategy for testing
    def trade(self):
        """
        Agent decides to trade resources in the market.
        """
        order_books = {'wood': self.sim.wood_order_book, 'stone': self.sim.stone_order_book}

        if self.wood > 2:
            self.place_order(order_books, 'wood', 'sell', price=90, quantity=1)
        elif self.wealth > 100:
            self.place_order(order_books, 'wood', 'buy', price=100, quantity=1)

        if self.stone > 2:
            self.place_order(order_books, 'stone', 'sell', price=100, quantity=1)
        elif self.wealth > 150:
            self.place_order(order_books, 'stone', 'buy', price=110, quantity=1)

    def buy(self):
        """
        Agent buys resources from the market.
        """
        wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
        stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)
        self.market.add_buyer(self, wood_to_buy, stone_to_buy)

    def sell(self):
        """
        Agent sells all resources to the market.
        """
        self.market.add_seller(self, self.wood, self.stone)

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

        # Log the order placement for debugging
        print(f"Agent {self.agent_id} placed a {order_type} order for {quantity} units of {resource_type} at price {price}.")


    def earning_rate_building(self):
        """
        Calculate the earning rate of building a house.
        """
        # If agent does not have enough resources, return 0
        if self.grid.house_cost[0] > self.wood or self.grid.house_cost[1] > self.stone:
            return 0

        age = self.sim.t - self.creation_time
        total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
        earning_rate = total_income / self.required_building_time
        return earning_rate

    def earning_rate_buying(self):
        """
        Calculate the earning rate of buying resources.
        """
        num_wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
        num_stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)
        cost = self.market.wood_rate * num_wood_to_buy + self.market.stone_rate * num_stone_to_buy

        # If agent has enough resources, return 0
        if num_wood_to_buy == 0 and num_stone_to_buy == 0:
            return 0

        # If agent doesn't have enough wealth, return 0
        if cost > self.wealth:
            return 0

        # If market doesn't have enough resources, return 0
        if num_wood_to_buy > self.market.wood or num_stone_to_buy > self.market.stone:
            return 0

        # Calculate and return the net earning rate
        age = self.sim.t - self.creation_time
        required_time = self.required_building_time + 1
        gross_income = self.income_per_timestep * (self.guessed_lifetime - age - required_time)
        return (gross_income - cost) / required_time

    def earning_rate_selling(self):
        """
        Calculate the earning rate of selling resources.
        """
        # If agent does not have any resources, return 0
        if self.wood == 0 and self.stone == 0:
            return 0
        earning_rate = self.market.wood_rate * self.wood + self.market.stone_rate * self.stone
        return earning_rate
    
    # TODO: Compare this with the earning rate of gathering resources
    def expected_value_gathering(self):
        """
        Calculate the expected value of gathering resources in the neighboring cells, including the current position.
        """
        # Get the neighboring cells including the current position
        positions_to_check = self.grid.get_neighbors(self.position) + [self.position]
        expected_value = 0

        # Check the resources in the neighboring cells and the current position
        for pos in positions_to_check:
            if self.grid.resource_matrix_wood[pos] > 0:
                expected_value += self.grid.resource_matrix_wood[pos] * self.market.wood_rate
            if self.grid.resource_matrix_stone[pos] > 0:
                expected_value += self.grid.resource_matrix_stone[pos] * self.market.stone_rate

        return expected_value

    def earning_rate_gathering(self):
        """
        Calculate the earning rate of gathering resources in the current position.
        """
        if self.grid.resource_matrix_wood[self.position] == 0:
            return 0

        # Simply use the price of the resources to estimate the earning rate
        earning_rate = self.market.wood_rate * min(1, self.grid.resource_matrix_wood[self.position]) + self.market.stone_rate * min(1, self.grid.resource_matrix_stone[self.position])  # Gather at most 1 unit of each resource per time step
        return earning_rate

    def earning_rate_random_moving(self):
        """
        Calculate the earning rate of moving to a neighboring cell randomly.
        """
        positions_to_check = self.grid.get_neighbors(self.position) # 4 Von Neumann neighborhood
        earning_rate = 0
        for pos in positions_to_check:
            wood, stone = self.grid.resource_matrix_wood[pos], self.grid.resource_matrix_stone[pos]

            # 1 time step to move to the cell. Time steps to gather resources is max(wood, stone). 
            required_time = max(wood, stone) + 1

            # Simply use the price of the resources to estimate the earning rate
            earning_rate += (self.market.wood_rate * wood + self.market.stone_rate * stone) / required_time
        
        # The probability of moving to a neighboring cell is 1/len(positions_to_check)
        return earning_rate / len(positions_to_check)