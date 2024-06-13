class Market:
    def __init__(self, initial_wood_rate, initial_stone_rate, sensitivity=0.01):
        self.wood_rate = initial_wood_rate
        self.stone_rate = initial_stone_rate
        self.sensitivity = sensitivity

        self.wood_to_buy = 0
        self.wood_to_sell = 0
        self.stone_to_buy = 0
        self.stone_to_sell = 0

        self.wood = 0
        self.stone = 0

    def add_seller(self, seller, wood_to_sell, stone_to_sell):
        """
        This function should be called by an agent when they decide to sell resources.

        seller (Agent): The agent that wants to sell
        wood_to_sell (int): The amount of wood the agent wants to sell
        stone_to_sell (int): The amount of stone the agent wants to sell
        """
        # Record 
        self.wood_to_sell += wood_to_sell
        self.stone_to_sell += stone_to_sell

        # Process the order
        seller.wealth += wood_to_sell * self.wood_rate + stone_to_sell * self.stone_rate
        seller.wood -= wood_to_sell
        self.wood += wood_to_sell
        seller.stone -= stone_to_sell
        self.stone += stone_to_sell

    def add_buyer(self, buyer, wood_to_buy, stone_to_buy):
        """
        This function should be called by an agent when they decide to buy resources.

        buyer (Agent): The agent that wants to buy
        wood_to_buy (int): The amount of wood the agent wants to buy
        stone_to_buy (int): The amount of stone the agent wants to buy
        """
        # Check if the market has enough resources
        if self.wood < wood_to_buy or self.stone < stone_to_buy:
            print("Market does not have enough resources.")
            return

        # Record
        self.wood_to_buy += wood_to_buy
        self.stone_to_buy += stone_to_buy

        # Process the order
        buyer.wealth -= wood_to_buy * self.wood_rate + stone_to_buy * self.stone_rate
        buyer.wood += wood_to_buy
        self.wood -= wood_to_buy
        buyer.stone += stone_to_buy
        self.stone -= stone_to_buy

    def update_price(self):
        """
        This function should be called at the end of simulation.timestep to update the price of resources.
        """
        self.wood_rate *= 1 + self.sensitivity * (self.wood_to_buy - self.wood_to_sell)
        self.stone_rate *= 1 + self.sensitivity * (self.stone_to_buy - self.stone_to_sell)

        # Reset the counters
        self.wood_to_buy = 0
        self.wood_to_sell = 0

        # # Show the market status
        # print(f"\nMarket:")
        # print(f"Wood to buy: {self.wood_to_buy}")
        # print(f"Wood to sell: {self.wood_to_sell}")
        # print(f"Updated wood rate: {self.wood_rate}")
        # print(f"Stone to buy: {self.stone_to_buy}")
        # print(f"Stone to sell: {self.stone_to_sell}")
        # print(f"Updated stone rate: {self.stone_rate}")