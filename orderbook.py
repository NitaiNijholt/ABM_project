import random

# Make buying / selling price based on EV of the building of the houses?

class OrderBooks:
    def __init__(self, agents_dict):
        self.bids = []
        self.asks = []
        self.transactions = []
        self.agents_dict = agents_dict
    
    def place_bid(self, agent_id, price):
        self.bids.append({'agent_id': agent_id, 'price': price})
        self.match_orders('bid')  # Pass 'bid' to indicate the function is called from place_bid
        
    def place_ask(self, agent_id, price):
        self.asks.append({'agent_id': agent_id, 'price': price})
        self.match_orders('ask')  # Pass 'ask' to indicate the function is called from place_ask
        
    def check_price(self):
        # Either there are orders left in the bids/ask or none
        if self.bids:
            best_bid = max(self.bids, key=lambda x: x['price'])
        else:
            best_bid = None
        
        if self.asks:
            best_ask = min(self.asks, key=lambda x: x['price'])
        else:
            best_ask = None
        return best_bid, best_ask
        
    def match_orders(self, order_type):
        transactions = []
        # Check for matching orders immediately when a new order is placed
        
        while self.bids and self.asks:
            best_bid = max(self.bids, key=lambda x: x['price'])
            best_ask = min(self.asks, key=lambda x: x['price'])
            
            if best_bid['price'] >= best_ask['price']:
                bid_agent = self.agents_dict[best_bid['agent_id']]
                ask_agent = self.agents_dict[best_ask['agent_id']]
                
                # Use the existing order's price for the transaction
                if order_type == 'bid':
                    price = best_ask['price']
                else:
                    price = best_bid['price']
                
                bid_agent['wealth'] -= price
                bid_agent['wood'] += 1  
                
                ask_agent['wealth'] += price
                ask_agent['wood'] -= 1
                
                self.bids.remove(best_bid)
                self.asks.remove(best_ask)
                
                self.transactions.append({'buyer': best_bid['agent_id'], 'seller': best_ask['agent_id'], 'price': price})
            else:
                break