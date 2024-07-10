class OrderBooks:
    def __init__(self, agents_dict, resource_type, order_lifespan, agents):
        """
        Initialize the OrderBooks class to manage bids and asks for resources in a trading system.

        This class tracks orders (bids and asks) placed by agents, matches them based on price,
        and handles order expiry after a specified number of timesteps.

        Parameters
        ----------
        agents_dict : dict
            Dictionary containing agent details with their ID as the key.
        resource_type : str
            The type of resource this order book handles (e.g., 'wood' or 'stone').
        order_lifespan : int
            The number of timesteps an order remains active before expiring.
        agents : dict
            Reference to the agents dictionary from the Simulation class.
        """
        self.bids = []
        self.asks = []
        self.transactions = []
        self.agents_dict = agents_dict
        self.resource_type = resource_type
        self.order_lifespan = order_lifespan
        self.current_timestep = 0
        self.agents = agents  # Reference to the actual Agent objects

    def place_bid(self, agent_id, price):
        """
        Place a bid order in the order book.

        Parameters
        ----------
        agent_id : int
            ID of the agent placing the bid.
        price : float
            Price of the bid.
        """
        self.bids.append({'agent_id': agent_id, 'price': price, 'timestamp': self.current_timestep})
        self.match_orders('bid')

    def place_ask(self, agent_id, price):
        """
        Place an ask order in the order book.

        Parameters
        ----------
        agent_id : int
            ID of the agent placing the ask.
        price : float
            Price of the ask.
        """
        self.asks.append({'agent_id': agent_id, 'price': price, 'timestamp': self.current_timestep})
        self.match_orders('ask')

    def check_price(self):
        """
        Check the current best bid and ask prices.

        Returns
        -------
        tuple
            Best bid price and best ask price.
        """
        if self.bids:
            best_bid = max(self.bids, key=lambda x: x['price'])['price']
        else:
            best_bid = None

        if self.asks:
            best_ask = min(self.asks, key=lambda x: x['price'])['price']
        else:
            best_ask = None
        return best_bid, best_ask

    def match_orders(self, order_type):
        """
        Match orders in the order book based on price.

        This method matches the highest bid with the lowest ask if the bid price is greater than or equal to the ask price,
        and updates the agents' resources and wealth accordingly.

        Parameters
        ----------
        order_type : str
            The type of order to match ('bid' or 'ask').
        """
        # Sort bids in descending order by price and asks in ascending order by price
        self.bids.sort(key=lambda x: x['price'], reverse=True)
        self.asks.sort(key=lambda x: x['price'])

        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            if best_bid['price'] >= best_ask['price']:
                bid_agent = self.agents_dict[best_bid['agent_id']]
                ask_agent = self.agents_dict[best_ask['agent_id']]

                if order_type == 'bid':
                    price = best_ask['price']
                else:
                    price = best_bid['price']

                # Update the wealth, resources, income
                bid_agent['wealth'] -= price
                bid_agent[self.resource_type] += 1
                
                ask_agent['wealth'] += price
                ask_agent[self.resource_type] -= 1

                # Update the actual Agent objects
                self.agents[best_bid['agent_id']].wealth = bid_agent['wealth']
                self.agents[best_bid['agent_id']].__setattr__(self.resource_type, bid_agent[self.resource_type])
                self.agents[best_ask['agent_id']].wealth = ask_agent['wealth']
                self.agents[best_ask['agent_id']].__setattr__(self.resource_type, ask_agent[self.resource_type])
                self.agents[best_ask['agent_id']].income += price
                
                # Decrement the amount_orders for both agents
                self.agents[best_bid['agent_id']].amount_orders -= 1
                self.agents[best_ask['agent_id']].amount_orders -= 1

                # Remove the best bid and ask from the lists
                self.bids.pop(0)
                self.asks.pop(0)
                
                self.transactions.append({'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
            else:
                break

    def increment_timestep(self):
        """
        Increment the current timestep and expire old orders.
        """
        self.current_timestep += 1
        self.expire_orders()

    def remove_orders(self, agent_id):
        """
        Remove all orders of a given agent from the order books.

        Parameters
        ----------
        agent_id : int
            ID of the agent whose orders should be removed.
        """
        self.bids = [order for order in self.bids if order['agent_id'] != agent_id]
        self.asks = [order for order in self.asks if order['agent_id'] != agent_id]

    def expire_orders(self):
        """
        Expire orders that have been in the order book longer than the order lifespan.
        """
        self.bids = [order for order in self.bids if not self.expire_order(order, 'bid')]
        self.asks = [order for order in self.asks if not self.expire_order(order, 'ask')]

    def expire_order(self, order, order_type):
        """
        Determine if an order should expire based on its timestamp.

        Parameters
        ----------
        order : dict
            The order to check for expiration.
        order_type : str
            The type of order ('bid' or 'ask').

        Returns
        -------
        bool
            True if the order has expired, False otherwise.
        """
        if self.current_timestep - order['timestamp'] >= self.order_lifespan:
            agent_id = order['agent_id']
            agent = self.agents_dict[agent_id]
            # Decrement the amount_orders for the agent
            self.agents[agent_id].amount_orders -= 1
            return True
        return False


