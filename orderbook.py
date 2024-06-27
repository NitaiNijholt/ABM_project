class OrderBooks:
    def __init__(self, agents_dict, resource_type, order_lifespan, agents):
        """
        OrderBooks class to manage bids and asks for resources in a trading system.

        This class tracks orders (bids and asks) placed by agents, matches them based on price,
        and handles order expiry after a specified number of timesteps.

        Attributes:
            agents_dict (dict): Dictionary containing agent details with their ID as the key.
            resource_type (str): The type of resource this order book handles (e.g., 'wood' or 'stone').
            order_lifespan (int): The number of timesteps an order remains active before expiring.
            current_timestep (int): The current timestep in the simulation.
            agents (dict): Reference to the agents dictionary from the Simulation class.
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
        # Debug print
        # print(f"Placing bid: Agent {agent_id}, Price {price}, Resource {self.resource_type}")
        self.bids.append({'agent_id': agent_id, 'price': price, 'timestamp': self.current_timestep})
        self.match_orders('bid')

    def place_ask(self, agent_id, price):
        # Debug print
        # print(f"Placing ask: Agent {agent_id}, Price {price}, Resource {self.resource_type}")
        self.asks.append({'agent_id': agent_id, 'price': price, 'timestamp': self.current_timestep})
        self.match_orders('ask')

    def check_price(self):
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
                self.agents[best_bid['agent_id']].income -= price  
                self.agents[best_ask['agent_id']].wealth = ask_agent['wealth']
                self.agents[best_ask['agent_id']].__setattr__(self.resource_type, ask_agent[self.resource_type])
                self.agents[best_ask['agent_id']].income += price
                
                # Decrement the amount_orders for both agents
                self.agents[best_bid['agent_id']].amount_orders -= 1
                self.agents[best_ask['agent_id']].amount_orders -= 1

                # Remove the best bid and ask from the lists
                self.bids.pop(0)
                self.asks.pop(0)

                # Debug print
                # print('TRANSACTION HAPPENED:', {'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
                self.transactions.append({'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
            else:
                break


    def increment_timestep(self):
        self.current_timestep += 1
        self.expire_orders()

    def remove_orders(self, agent_id):
        """
        Remove all orders of a given agent from the order books.
        """
        self.bids = [order for order in self.bids if order['agent_id'] != agent_id]
        self.asks = [order for order in self.asks if order['agent_id'] != agent_id]

    def expire_orders(self):
        self.bids = [order for order in self.bids if not self.expire_order(order, 'bid')]
        self.asks = [order for order in self.asks if not self.expire_order(order, 'ask')]

    def expire_order(self, order, order_type):
        if self.current_timestep - order['timestamp'] >= self.order_lifespan:
            agent_id = order['agent_id']
            agent = self.agents_dict[agent_id]

            if order_type == 'bid':
                # Refund the bid amount
                agent['wealth'] += order['price']
            elif order_type == 'ask':
                if agent[self.resource_type] >= 1:
                    agent[self.resource_type] -= 1

            # Decrement the amount_orders for the agent
            self.agents[agent_id].amount_orders -= 1

            # Debug print
            # print(f"Order expired: Agent {agent_id}, Price {order['price']}, Resource {self.resource_type}, Type {order_type}")
            return True
        return False

