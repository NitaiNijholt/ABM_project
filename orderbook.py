import random

# Make buying / selling price based on EV of the building of the houses?

# +
import numpy as np

class OrderBooks:
    def __init__(self, agents_dict, resource_type, order_lifespan):
        """
        OrderBooks class to manage bids and asks for resources in a trading system.

        This class tracks orders (bids and asks) placed by agents, matches them based on price,
        and handles order expiry after a specified number of timesteps.

        Attributes:
            agents_dict (dict): Dictionary containing agent details with their ID as the key.
            resource_type (str): The type of resource this order book handles (e.g., 'wood' or 'stone').
            order_lifespan (int): The number of timesteps an order remains active before expiring.
            current_timestep (int): The current timestep in the simulation.
        """
        self.bids = []
        self.asks = []
        self.transactions = []
        self.agents_dict = agents_dict
        self.resource_type = resource_type
        self.order_lifespan = order_lifespan
        self.current_timestep = 0
    
    def place_bid(self, agent_id, price):
        print(f"Placing bid: Agent {agent_id}, Price {price}, Resource {self.resource_type}")
        self.bids.append({'agent_id': agent_id, 'price': price, 'timestamp': self.current_timestep})
        self.match_orders('bid')
        
    def place_ask(self, agent_id, price):
        print(f"Placing ask: Agent {agent_id}, Price {price}, Resource {self.resource_type}")
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
        while self.bids and self.asks:
            best_bid = max(self.bids, key=lambda x: x['price'])
            best_ask = min(self.asks, key=lambda x: x['price'])
            
            if best_bid['price'] >= best_ask['price']:
                print('BEST_BID, BEST_ASK', best_bid, best_ask)
                print('AGENT_DICT HERE WEEWEWEO', self.agents_dict)
                bid_agent = self.agents_dict[best_bid['agent_id']]
                ask_agent = self.agents_dict[best_ask['agent_id']]
                
                if order_type == 'bid':
                    price = best_ask['price']
                else:
                    price = best_bid['price']
                
                bid_agent['wealth'] -= price
                bid_agent[self.resource_type] += 1  
                
                ask_agent['wealth'] += price
                ask_agent[self.resource_type] -= 1
                
                self.bids.remove(best_bid)
                self.asks.remove(best_ask)
                print('TRANSACTION HAPPENED:', {'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
                self.transactions.append({'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
            else:
                break

    def increment_timestep(self):
        self.current_timestep += 1
        self.expire_orders()

    def expire_orders(self):
        self.bids = [order for order in self.bids if not self.expire_order(order, 'bid')]
        self.asks = [order for order in self.asks if not self.expire_order(order, 'ask')]

    def expire_order(self, order, order_type):
        if self.current_timestep - order['timestamp'] >= self.order_lifespan:
            agent = self.agents_dict[order['agent_id']]
            if order_type == 'bid':
                # Refund the bid amount
                agent['wealth'] += order['price']
            elif order_type == 'ask':
                if agent[self.resource_type] >= 1:
                    agent[self.resource_type] -= 1
            print(f"Order expired: Agent {order['agent_id']}, Price {order['price']}, Resource {self.resource_type}, Type {order_type}")
            return True
        return False

