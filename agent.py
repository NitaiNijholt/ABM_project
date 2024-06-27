import numpy as np
from scipy.stats import gamma, lognorm
from grid import Grid
from house import House
from agent_static_market import Agent_static_market

class Agent(Agent_static_market):
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
        super().__init__(sim, agent_id, position, grid, market, creation_time, wealth, wood, stone, lifetime_distribution, lifetime_mean, lifetime_std, income_per_timestep)
        self.order_books = {'wood': self.sim.wood_order_book, 'stone': self.sim.stone_order_book}
        self.amount_orders = 0
        self.limit = 1
        

    def step(self):
        self.collect_income()
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
#             print(f"Agent {self.agent_id} action rates: {self.earning_rates}")
            action_index = np.argmax(list(self.earning_rates.values()))
            action = actions[action_index]
            
            if action.__name__ == 'buy' or action.__name__ == 'sell':
                if self.amount_orders >= self.limit:
                    # print('LIMIT HIT')
                    earning_rates_values = list(self.earning_rates.values())
                    earning_rates_values[action_index] = -10
                    # Compute the argmax again to find the second highest value
                    second_action_index = np.argmax(earning_rates_values)
                    action = actions[second_action_index]                 
            self.current_action = action.__name__
#             print(f"Agent {self.agent_id} at timestep {self.sim.t} performing action: {self.current_action}")
            list_sell = ['build', 'sell', 'gather']
            if action.__name__ in list_sell:
                self.amount_orders += 1
                self.update_prices('sell')
            elif action.__name__ == 'buy':
                self.amount_orders += 1
                self.update_prices('buy')
                
            action()
#             self.current_action = action.__name__
            
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
        if wood_price and stone_price and self.wealth >= wood_price * wood_to_buy + stone_price * stone_to_buy:
            self.place_order(self.order_books, 'wood', 'buy', price = wood_price, quantity = wood_to_buy)
            self.place_order(self.order_books, 'stone', 'buy', price = stone_price, quantity = stone_to_buy)

    def sell(self):
        """
        Agent sells all resources to the market.
        """
        
        wood_price = self.determine_price('sell', 'wood')
        stone_price = self.determine_price('sell', 'stone')
        if wood_price and stone_price:
            if self.wood > 0:
                self.place_order(self.order_books, 'wood', 'sell', price = wood_price, quantity = self.wood)
            if self.stone > 0:
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
        assert order_type in ('sell', 'buy'), 'Takes in "buy" or "sell" only'

        age = self.sim.t - self.creation_time
        total_income = self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time)
        earning_rate = total_income / self.required_building_time
        m_price = earning_rate / sum(self.grid.house_cost)

        ob_prices = {
            'wood': self.sim.wood_order_book.check_price(),
            'stone': self.sim.stone_order_book.check_price()
        }

        for resource in ob_prices:
            ob_price_bid, ob_price_ask = ob_prices[resource]
            if order_type == 'sell':
                if ob_price_bid is not None:
                    rate = max(ob_price_bid, m_price)
                else:
                    rate = m_price
            else:
                if ob_price_ask is not None:
                    rate = min(ob_price_ask, m_price)
                else:
                    rate = m_price

            if resource == 'wood':
                self.market.wood_rate = rate
            elif resource == 'stone':
                self.market.stone_rate = rate

        
    def determine_price(self, order_type, resource):
        age = self.sim.t - self.creation_time
        total_income = max(self.income_per_timestep * (self.guessed_lifetime - age - self.required_building_time), 0)
        if total_income == 0:
            return None
        earning_rate = total_income / self.required_building_time
        m_price = earning_rate / sum(self.grid.house_cost)

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