import numpy as np

class House:
    def __init__(self, owner, position, income_per_timestep):
        self.owner = owner
        self.position = position
        self.income_per_timestep = income_per_timestep