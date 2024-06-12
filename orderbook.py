#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random


# In[ ]:


class OrderBook:
    def __init__(self, agents_dict):
        self.bids = []
        self.asks = []
        self.agents_dict = agents_dict
    
    def place_bid(self, agent_id, price):
        self.bids.append({'agent_id': agent_id, 'price': price})
        
    def place_ask(self, agent_id, price):
        self.asks.append({'agent_id': agent_id, 'price': price})
        
    def match_orders(self):
        transactions = []
        while self.bids and self.asks:
            best_bid = max(self.bids, key=lambda x: x['price'])
            best_ask = min(self.asks, key=lambda x: x['price'])
            
            if best_bid['price'] >= best_ask['price']:
                bid_agent = self.agents_dict[best_bid['agent_id']]
                ask_agent = self.agents_dict[best_ask['agent_id']]
                
                price = best_ask['price']
                
                bid_agent['wealth'] -= price
                bid_agent['wood'] += 1  
                
                ask_agent['wealth'] += price
                ask_agent['wood'] -= 1
                
                self.bids.remove(best_bid)
                self.asks.remove(best_ask)
                
                transactions.append({'buyer': bid_agent, 'seller': ask_agent, 'price': price})
            else:
                break
        
        return transactions
    
    def set_aggregate_price(self, prev_price):
        self.bids.sort(key=lambda x: x['price'], reverse=True)
        self.asks.sort(key=lambda x: x['price'])
        
        if not self.bids or not self.asks:
            return prev_price
        
        best_bid = self.bids[0]['price']
        best_ask = self.asks[0]['price']
        
        if best_bid >= best_ask:
            transactions = self.match_orders()
            avg_price = sum(t['price'] for t in transactions) / len(transactions)
            new_price = avg_price + self.delta * len(transactions)
        else:
            new_price = prev_price
        
        self.bids = []
        self.asks = []
        
        return new_price

