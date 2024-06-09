import numpy as np
from grid import Grid
from house import House

class Agent:
    def __init__(self, sim, agent_id, position, grid, market, life_expectancy, creation_time, wealth=0, wood=0, stone=0):
        self.agent_id = agent_id
        self.position = position
        self.wealth = wealth
        self.wood = wood
        self.stone = stone
        self.grid = grid
        self.market = market
        self.houses = []
        self.market_position = (0, 0)  # Define the market position
        self.life_expectancy = life_expectancy
        self.creation_time = creation_time
        self.sim = sim

        self.currently_building_timesteps = 0
        self.required_building_time = 5

        
    def find_random_move(self):
        """
        Finds a random move and returns the position
        """
        neighbors = self.grid.get_neighbors(self.position)
        possible_moves = [neighbor for neighbor in neighbors if (self.grid.if_no_agent(neighbor) and self.grid.house_matrix[neighbor] == 0)]
        return possible_moves[np.random.randint(len(possible_moves))]

    def move(self):
        """
        Agent moves to an empty neighboring cell. If all neighboring cells are occupied, the agent does not move.
        """

        try:
            # Change position of agent if there is an empty neighboring cell
            new_position = self.find_random_move()
        except ValueError:  # Catch the case where possible_moves is empty
            print(f"Agent {self.agent_id} has no possible moves")
            return

        # Move agent on the grid
        self.grid.agent_matrix[self.position] = 0
        self.grid.agent_matrix[new_position] = self.agent_id
        self.position = new_position

    def collect_resources(self):
        """
        Agent collects resources from the current position.
        """
        print(f"Agent {self.agent_id} at position {self.position} tries to collect resources.")
        print(f"Current inventory - Wood: {self.wood}, Stone: {self.stone}")

        resources_collected = False

        if self.grid.resource_matrix_wood[self.position] > 0:
            self.wood += 1
            self.grid.resource_matrix_wood[self.position] -= 1
            resources_collected = True

        if self.grid.resource_matrix_stone[self.position] > 0:
            self.stone += 1
            self.grid.resource_matrix_stone[self.position] -= 1
            resources_collected = True

        if resources_collected:
            print(f"Collected resources succesfully! New inventory - Wood: {self.wood}, Stone: {self.stone}")
            return True
        print(f"No resources collected at position {self.position}")
        return False

    def build_house(self, income_per_timestep=1):
        """
        Agent completes the construction of a house
        """ 

        self.grid.house_matrix[self.position] = 1
        house = House(self.agent_id, self.position, income_per_timestep=income_per_timestep)
        self.houses.append(house)
        self.grid.houses[self.position] = house
        print(f"Agent {self.agent_id} completed building a house at {self.position}")


    def start_building_house(self):
        """
        Agent builds a house at the current position if the agent has enough resources and the cell is empty.
        """

        wood_cost, stone_cost = self.grid.house_cost
        if self.wood >= wood_cost and self.stone >= stone_cost and self.grid.house_matrix[self.position] == 0:
            self.wood -= wood_cost
            self.stone -= stone_cost
            print(f"Agent {self.agent_id} starts building a house at {self.position}")
            self.currently_building_timesteps = 1
        
    def collect_income(self):
        """
        Agent collects income from all houses.
        """
        income_collected = sum(house.income_per_timestep for house in self.houses)
        self.wealth += income_collected
        print(f"Agent {self.agent_id} collected {income_collected} income from {len(self.houses)} houses. Total wealth: {self.wealth}")

    def step(self):
        """
        Agent performs a step: move, collect resources, possibly build a house, and trade.
        """

        # If the agent is currently building
        if self.currently_building_timesteps > 0:
            print(f"Agent is currently building a house and cannot do other things. Currently building {self.currently_building_timesteps}/{self.required_building_time}")

            self.currently_building_timesteps += 1

            # If the agent just now completed building:
            if self.currently_building_timesteps == self.required_building_time:
                self.currently_building_timesteps = 0
                self.build_house()

        # Agent can do other things if he is not building
        else:

            # Agent tries to collect resource, and nothing else if he succeeds
            if not self.collect_resources():
                self.move()
                self.start_building_house()
                # self.trade(wood_to_trade=1, stone_to_trade=1)
    
        # Agent can always collect
        self.collect_income()

        # Agent dies when he reaches his life expectancy
        if self.sim.t >= self.creation_time + self.life_expectancy:
            self.die()
    
    def die(self):
        del self.grid.agents[self.agent_id]
        self.grid.agent_matrix[self.position] = 0
        print(f"Agent {self.agent_id} died at the age of {self.life_expectancy}")
        
    def move_to_market(self):
        """
        Agent moves to the market position.
        """
        self.grid.agent_matrix[self.position] = 0
        self.position = self.market_position
        self.grid.agent_matrix[self.position] = self.agent_id
        
    def trade(self, wood_to_trade=0, stone_to_trade=0):
        """
        Agent trades resources at the market.
        
        Currently: If agent has any resources at all, go to market and trade
        TODO: Walking to the market takes multiple timesteps if far away. Now agents teleport
        TODO: Add utility function such that it determines how and when agent will trade
        """
        self.move_to_market()

        if wood_to_trade > 0:
            if wood_to_trade <= self.wood:
                wealth_received = self.market.trade_wood_for_wealth(wood_to_trade)
                self.wood -= wood_to_trade
                self.wealth += wealth_received
                print(f"Agent {self.agent_id} traded {wood_to_trade} wood for {wealth_received} wealth")
            else:
                print(f"Agent {self.agent_id} does not have enough wood to trade")

        if stone_to_trade > 0:
            if stone_to_trade <= self.stone:
                wealth_received = self.market.trade_stone_for_wealth(stone_to_trade)
                self.stone -= stone_to_trade
                self.wealth += wealth_received
                print(f"Agent {self.agent_id} traded {stone_to_trade} stone for {wealth_received} wealth")
            else:
                print(f"Agent {self.agent_id} does not have enough stone to trade")
