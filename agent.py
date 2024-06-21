import numpy as np
from scipy.stats import gamma, lognorm
from grid import Grid
from house import House
import random

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
        self.best_position = 0

        self.wealth_over_time = []
        self.houses_over_time = []
        self.gathered_at_timesteps = []
        self.bought_at_timesteps = []
        self.sold_at_timesteps = []
        self.taxes_paid_at_timesteps = []

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

    def random_move(self):
        """
        Agent moves to a neighboring cell randomly.
        """
        possible_moves = self.find_moves()
        if len(possible_moves) == 0:
            return

        new_position = possible_moves[np.random.randint(len(possible_moves))]

        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position

    def move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        """
        new_position = self.best_position
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
        income_collected = sum(self.grid.house_incomes[house.position] for house in self.houses) * self.income_per_timestep
        self.wealth += income_collected
        self.income = income_collected  # Track the income

    def step(self):
        self.income = sum(self.grid.house_incomes[house.position] for house in self.houses) * self.income_per_timestep
        # print(f"Agent {self.agent_id} at step {self.sim.t}: Wealth={self.wealth}, Wood={self.wood}, Stone={self.stone}")
        if self.currently_building_timesteps > 0:
            self.currently_building_timesteps += 1
            self.current_action = 'building_house'
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()
        else:
            actions = [self.move, self.build, self.buy, self.sell, self.gather]
            self.best_position = self.find_target_position()
            self.earning_rates = {
                'move': self.earning_rate_moving(),
                'build': self.earning_rate_building(),
                'buy': self.earning_rate_buying(),
                'sell': self.earning_rate_selling(),
                'gather': self.earning_rate_gathering()
            }
            # print(f"Agent {self.agent_id} action rates: {self.earning_rates}")
            action = actions[np.argmax(list(self.earning_rates.values()))]
            # print(f"Agent {self.agent_id} at timestep {self.sim.t} performing action: {self.current_action}")
            action()
            self.current_action = action.__name__
            
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

        for house in self.houses:
            self.grid.house_matrix[house.position] -= 1

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

    def earning_rate_building(self):
        """
        Calculate the earning rate of building a house.
        """
        if self.grid.house_cost[0] > self.wood or self.grid.house_cost[1] > self.stone or self.grid.house_matrix[self.position] >= self.grid.max_house_num:
            return 0

        age = self.sim.t - self.creation_time
        total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
        total_income = self.posttax_extra_income(total_income)
        earning_rate = total_income / self.required_building_time
        return earning_rate

    def earning_rate_buying(self):
        """
        Calculate the earning rate of buying resources.
        """
        num_wood_to_buy = max(0, self.grid.house_cost[0] - self.wood)
        num_stone_to_buy = max(0, self.grid.house_cost[1] - self.stone)
        cost = self.market.wood_rate * num_wood_to_buy + self.market.stone_rate * num_stone_to_buy

        if num_wood_to_buy == 0 and num_stone_to_buy == 0:
            return 0

        if cost > self.wealth:
            return 0

        if num_wood_to_buy > self.market.wood or num_stone_to_buy > self.market.stone:
            return 0

        age = self.sim.t - self.creation_time
        required_time = self.required_building_time + 1
        gross_income = self.income_per_timestep * (self.guessed_lifetime - age - required_time)
        gross_income = self.posttax_extra_income(gross_income)
        return (gross_income - cost) / required_time

    def earning_rate_selling(self):
        """
        Calculate the earning rate of selling resources.
        """
        if self.wood == 0 and self.stone == 0:
            return 0
        earning_rate = self.market.wood_rate * self.wood + self.market.stone_rate * self.stone
        earning_rate = self.posttax_extra_income(earning_rate)
        return earning_rate

    def earning_rate_gathering(self):
        """
        Calculate the earning rate of gathering resources in the current position.
        """
        if self.grid.resource_matrix_wood[self.position] == 0 and self.grid.resource_matrix_stone[self.position] == 0:
            return 0

        earning_rate = self.market.wood_rate * max(1, self.grid.resource_matrix_wood[self.position]) + self.market.stone_rate * max(1, self.grid.resource_matrix_stone[self.position])
        earning_rate = self.posttax_extra_income(earning_rate)
        return earning_rate

    def earning_rate_random_moving(self):
        """
        Calculate the earning rate of moving to a neighboring cell randomly.
        """
        if (self.grid.house_cost[0] <= self.wood and self.grid.house_cost[1] <= self.stone) and self.grid.house_matrix[self.position] != 0:
            age = self.sim.t - self.creation_time
            total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time - 1)
            total_income = self.posttax_extra_income(total_income)
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
            required_time = min(wood, stone) + 1
            earning_rate += self.posttax_extra_income(self.market.wood_rate * wood + self.market.stone_rate * stone) / required_time
        return earning_rate / len(positions_to_check)

    def earning_rate_moving(self):
        """
        Calculate the earning rate of moving to a target neighboring cell.
        """
        position = self.best_position
        # If the agent has enough resources and the cell is available for building, the agent should go there and build a house
        if self.grid.house_cost[0] <= self.wood and self.grid.house_cost[1] <= self.stone and self.grid.house_matrix[position] < self.grid.max_house_num:
            age = self.sim.t - self.creation_time
            total_income = self.income_per_timestep * self.grid.house_incomes[position] * (self.guessed_lifetime - age - self.required_building_time - 1)
            total_income = self.posttax_extra_income(total_income)
            earning_rate = total_income / (self.required_building_time + 1)
            return earning_rate
        # Otherwise, the agent should move to the target cell and gather resources
        else:
            wood, stone = self.grid.resource_matrix_wood[position], self.grid.resource_matrix_stone[position]
            required_time = min(wood, stone) + 1
            earning_rate = self.posttax_extra_income(self.market.wood_rate * wood + self.market.stone_rate * stone) / required_time

        return earning_rate

    def find_target_position(self):
        """
        Find the target position to move to.
        """
        positions = self.grid.get_neighbors(self.position)
        positions = [pos for pos in positions if self.grid.if_no_agents(pos)]
        # print(f"Agent {self.agent_id} available positions: {positions}")

        # If there are no available positions, the agent should stay in the current position
        if not positions:
            print(f"Agent {self.agent_id} has no available positions to move to.")
            print(f"Returning current position: {self.position}")
            return self.position

        # The initial best position is randomly selected from the available positions
        best_position = random.choice(positions)

        # When the agent has enough resources to build a house, the agent should move to the most valuable cell that is available for building
        if self.wood >= self.grid.house_cost[0] and self.stone >= self.grid.house_cost[1]:
            for pos in positions:
                if self.grid.house_matrix[pos] < self.grid.max_house_num and self.grid.house_incomes[pos] > self.grid.house_incomes[best_position]:
                    best_position = pos
        # Otherwise, the agent should move to the neighbors to gather resources
        else:
            required_wood, required_stone = self.grid.house_cost[0] - self.wood, self.grid.house_cost[1] - self.stone
            best_position = max(positions, key=lambda pos: min(self.grid.resource_matrix_wood[pos], required_wood) + min(self.grid.resource_matrix_stone[pos], required_stone))
        return best_position

    def posttax_extra_income(self, extra_income):
        """
        Calculate the post-tax extra income given the pre-tax extra income.
        """
        tax_brackets = self.sim.tax_policy.tax_brackets
        # First, calculate the post-tax house income
        tax1 = 0
        previous_bound = 0
        for upper_bound, tax_rate in tax_brackets:
            if self.income > previous_bound:
                taxable_income = min(self.income - previous_bound, upper_bound - previous_bound)
                tax1 += taxable_income * tax_rate
                previous_bound = upper_bound
            else:
                break
        # posttax_house_income = self.income - tax1

        # Then, calculate the total post-tax income
        tax2 = 0
        previous_bound = 0
        for upper_bound, tax_rate in tax_brackets:
            if extra_income > previous_bound:
                taxable_income = min(extra_income - previous_bound, upper_bound - previous_bound)
                tax2 += taxable_income * tax_rate
                previous_bound = upper_bound
            else:
                break
        # total_posttax_income = self.income + extra_income - tax2

        return extra_income + tax1 - tax2