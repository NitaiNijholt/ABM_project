import numpy as np
from scipy.stats import gamma, lognorm
from grid import Grid
from house import House

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
        self.order_books = {'wood': self.sim.wood_order_book, 'stone': self.sim.stone_order_book}

        self.income = 0  # New attribute to track income

        assert lifetime_distribution in ['gamma', 'lognormal'], 'Invalid lifetime distribution'

        # Generate the actual lifetime
        self.actual_lifetime = self.generate_lifetime(lifetime_distribution, lifetime_mean, lifetime_std)

        # Guess the lifetime
        self.guessed_lifetime = self.generate_lifetime(lifetime_distribution, lifetime_mean, lifetime_std)

        # Current objective of the agent. Now, 'Trade' is the only thing that does something
        self.objective = 'Trade'

        # Exponential factor indicating how determined the agent is in his objective. 0 is equivalent to no objective, and higher values lead to smaller and smaller probabilities that the agent does something which delays achieving the objective
        self.objective_alpha = 1

    def generate_lifetime(self, lifetime_distribution, lifetime_mean, lifetime_std):
        """
        Generate a lifetime based on the given distribution.
        """
        if lifetime_distribution == 'gamma':
            scale = lifetime_std**2 / lifetime_mean
            shape = lifetime_mean / scale
            return int(gamma.rvs(a=shape, scale=scale, size=1)[0])
        elif lifetime_distribution == 'lognormal':
            sigma = np.sqrt(np.log(1 + (lifetime_std / lifetime_mean)**2))
            mu = np.log(lifetime_mean) - sigma**2 / 2
            return int(lognorm.rvs(s=sigma, scale=np.exp(mu), size=1)[0])

    def find_moves(self):
        """
        Finds all possible moves and returns them
        """
        neighbors = self.grid.get_neighbors(self.position)
        return [neighbor for neighbor in neighbors if self.grid.if_no_agents(neighbor)]

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
        
        min_score = min(move_scores)
        move_scores = [1 + (score - min_score) ** self.objective_alpha for score in move_scores]
        return possible_moves[np.random.choice(len(possible_moves), p=np.array(move_scores)/sum(move_scores))]

    def move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        """
        possible_moves = self.find_moves()
        if len(possible_moves) == 0:
            return

        new_position = possible_moves[np.random.randint(len(possible_moves))]

        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position

    def gather(self):
        """
        Agent gathers resources from the current position.
        """
        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            self.gathered_at_timesteps.append(1)
        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            self.gathered_at_timesteps.append(1)
        self.gathered_at_timesteps.append(0)

    def build_house(self):
        """
        Agent completes the construction of a house
        """
        if self.grid.house_matrix[self.position] >= self.grid.max_house_num:
            return
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
        wood_cost, stone_cost = self.grid.house_cost
        self.wood -= wood_cost
        self.stone -= stone_cost
        self.currently_building_timesteps = 1

    def collect_income(self):
        """
        Agent collects income from all houses.
        """
        income_collected = sum(self.grid.house_incomes[house.position] for house in self.houses)
        self.wealth += income_collected
        self.income = income_collected  # Track the income

    def step(self):
        # print(f"Agent {self.agent_id} at step {self.sim.t}: Wealth={self.wealth}, Wood={self.wood}, Stone={self.stone}")
        if self.currently_building_timesteps > 0:
            self.currently_building_timesteps += 1
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()
        else:
            actions = [self.move, self.build, self.buy, self.sell, self.gather]
            self.earning_rates = {
                'move': self.earning_rate_random_moving(),
                'build': self.earning_rate_building(),
                'buy': self.earning_rate_buying(),
                'sell': self.earning_rate_selling(),
                'gather': self.earning_rate_gathering()
            }
            # print(f"Agent {self.agent_id} action rates: {self.earning_rates}")
            action = actions[np.argmax(list(self.earning_rates.values()))]
            self.current_action = action.__name__
            # print(f"Agent {self.agent_id} at timestep {self.sim.t} performing action: {self.current_action}")
            action()

        self.collect_income()
        self.wealth_over_time.append(self.wealth)
        self.houses_over_time.append(len(self.houses))
        self.bought_at_timesteps.append(0)
        self.sold_at_timesteps.append(0)
        self.taxes_paid_at_timesteps.append(0)

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
        
        # Remove agent's orders from the order books
        self.sim.wood_order_book.remove_orders(self.agent_id)
        self.sim.stone_order_book.remove_orders(self.agent_id)

        for house in self.houses:
            self.grid.house_matrix[house.position] -= 1

    def buy(self):
        """
        Agent buys resources from the market.
        """      
        wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
        stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)
        wood_price = self.determine_price('buy', 'wood')
        stone_price = self.determine_price('buy', 'stone')
        self.place_order(self.order_books, 'wood', 'buy', price = wood_price, quantity = wood_to_buy)
        self.place_order(self.order_books, 'stone', 'buy', price = stone_price, quantity = stone_to_buy)

    def sell(self):
        """
        Agent sells all resources to the market.
        """
        
        wood_price = self.determine_price('sell', 'wood')
        stone_price = self.determine_price('sell', 'stone')
        self.place_order(self.order_books, 'wood', 'sell', price = wood_price, quantity = self.wood)
        self.place_order(self.order_books, 'stone', 'sell', price = stone_price, quantity = self.stone)

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

        # Log the order placement for debugging
        # print(f"Agent {self.agent_id} placed a {order_type} order for {quantity} units of {resource_type} at price {price}.")

    def update_prices(self, order_type):
        order_books = {'wood': self.sim.wood_order_book, 'stone': self.sim.stone_order_book}
        
        building_earning_rate = self.earning_rate_building()
        # max/min prices per resource for buying/selling 
        age = self.sim.t - self.creation_time
        total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
        earning_rate = total_income / self.required_building_time
        m_price = earning_rate / sum(self.grid.house_cost)
        ob_price_wood_bid, ob_price_wood_ask = self.sim.wood_order_book.check_price()
        ob_price_stone_bid, ob_price_stone_ask = self.sim.stone_order_book.check_price()
        assert order_type == 'sell' or order_type == 'buy', 'Takes in "buy" or "sell" only'

        # Sell order
        if order_type == 'sell':
            if ob_price_wood_bid != None:
                self.market.wood_rate = max(ob_price_wood_bid, m_price)
            else:
                self.market.wood_rate = m_price
                
            if ob_price_stone_bid != None:
                self.market_stone_rate = max(ob_price_stone_bid, m_price)
            else:
                self.market_stone_rate = m_price
        # if not sell then buy order
        else:
            if ob_price_wood_ask != None:
                self.market.wood_rate = min(ob_price_wood_ask, m_price)
            else:
                self.market.wood_rate = m_price
            if ob_price_stone_ask != None:
                self.market.stone_rate = min(ob_price_stone_ask, m_price)
            else:
                self.market.stone_rate = m_price
        
    def determine_price(self, order_type, resource):
        age = self.sim.t - self.creation_time
        total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
        earning_rate = total_income / self.required_building_time
        m_price = earning_rate / sum(self.grid.house_cost)
        ob_price_wood_bid, ob_price_wood_ask = self.sim.wood_order_book.check_price()
        ob_price_stone_bid, ob_price_stone_ask = self.sim.stone_order_book.check_price()
        if order_type == 'buy':
            if resource == 'wood':
                if ob_price_wood_ask != None:
                    self.market.wood_rate = min(ob_price_wood_ask, m_price)
                    return self.market.wood_rate
                else:
                    self.market.wood_rate = m_price
                    return self.market.wood_rate
            if resource == 'stone':
                if ob_price_stone_ask != None:
                    self.market.stone_rate = min(ob_price_stone_ask, m_price)
                    return self.market.stone_rate
                else:
                    self.market.stone_rate = m_price
                    return self.market.stone_rate
        elif order_type == 'sell':
            if resource == 'wood':
                if ob_price_wood_bid != None:
                    self.market.wood_rate = max(ob_price_wood_bid, m_price)
                    return self.market.wood_rate
                else:
                    self.market.wood_rate = m_price
                    return self.market.wood_rate

            if resource == 'stone':
                if ob_price_stone_ask != None:
                    self.market.stone_rate = max(ob_price_stone_bid, m_price)
                    return self.market.stone_rate
                else:
                    self.market.stone_rate = m_price
                    return self.market.stone_rate
        
    def earning_rate_building(self):
        """
        Calculate the earning rate of building a house.
        """
        if self.grid.house_cost[0] > self.wood or self.grid.house_cost[1] > self.stone or self.grid.house_matrix[self.position] >= self.grid.max_house_num:
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
        self.update_prices('buy')
        
        cost = self.market.wood_rate * num_wood_to_buy + self.market.stone_rate * num_stone_to_buy

        # If agent has enough resources, return 0
        if num_wood_to_buy == 0 and num_stone_to_buy == 0:
            return 0

        # If agent doesn't have enough wealth, return 0
        if cost > self.wealth:
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
            print()
            return 0
        self.update_prices('sell')
        earning_rate = self.market.wood_rate * self.wood + self.market.stone_rate * self.stone
        return earning_rate
    
    def earning_rate_gathering(self):
        """
        Calculate the earning rate of gathering resources in the current position.
        """
        if self.grid.resource_matrix_wood[self.position] == 0 and self.grid.resource_matrix_stone[self.position] == 0:
            return 0
        self.update_prices('sell')
        earning_rate = self.market.wood_rate * min(1, self.grid.resource_matrix_wood[self.position]) + self.market.stone_rate * min(1, self.grid.resource_matrix_stone[self.position])
        return earning_rate
    
    

    def earning_rate_random_moving(self):
        """
        Calculate the earning rate of moving to a neighboring cell randomly.
        """
        self.update_prices('sell')
        if (self.grid.house_cost[0] <= self.wood and self.grid.house_cost[1] <= self.stone) and self.grid.house_matrix[self.position] != 0:
            age = self.sim.t - self.creation_time
            total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
            earning_rate = total_income / (self.required_building_time + 1)
            return earning_rate
        
        positions = self.grid.get_neighbors(self.position) # 4 Von Neumann neighborhood
        positions_to_check = set()
        for close_position in positions:
            for position in self.grid.get_neighbors(close_position):
                positions_to_check.add(position)
        
        earning_rate = 0
        for pos in positions_to_check:
            wood, stone = self.grid.resource_matrix_wood[pos], self.grid.resource_matrix_stone[pos]
            required_time = max(wood, stone) + 1
            earning_rate += (self.market.wood_rate * wood + self.market.stone_rate * stone) / required_time
        
        return earning_rate / len(positions_to_check)
