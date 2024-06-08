#!/usr/bin/env python
# coding: utf-8
# %%

# %%


class Market:
    def __init__(self, wood_rate, stone_rate):
        self.wood_rate = wood_rate
        self.stone_rate = stone_rate

    def trade_wood_for_wealth(self, wood_amount):
        wealth_amount = wood_amount * self.wood_rate
        return wealth_amount

    def trade_stone_for_wealth(self, stone_amount):
        wealth_amount = stone_amount * self.stone_rate
        return wealth_amount

