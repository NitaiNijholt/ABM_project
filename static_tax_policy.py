from grid import Grid
from simulation import Simulation
import numpy as np

class StaticTaxPolicy:
    def __init__(self, grid):
        self.agents = list(grid.agents.values())
        # self.agents = []
        # for agent in grid.agents.values():
        #     self.agents.append(agent)
        self.tax_brackets = self.calculate_tax_brackets()

    def calculate_tax_brackets(self):
        # Collect all wealth values from agents and sort them
        wealth_values = sorted(agent.wealth for agent in self.agents)

        # Determine wealth brackets based on quantiles
        quartiles = np.percentile(wealth_values, [25, 50, 75, 100])
        tax_rates = [0.1, 0.2, 0.3, 0.4]

        # Create a list of (upper_bound, tax_rate) tuples
        tax_brackets = [(quartiles[i], tax_rates[i]) for i in range(len(quartiles))]
        return tax_brackets
    
    def calculate_tax(self, agent):
        # Determine the tax rate based on the agent's wealth
        for i, (upper_bound, tax_rate) in enumerate(self.tax_brackets):
            if agent.wealth <= upper_bound:
                return agent.wealth * tax_rate

    def apply_taxes(self):
        # Apply tax to each agent and adjust their wealth
        for agent in self.agents:
            tax = self.calculate_tax(agent)
            agent.wealth -= tax
            print(f"Agent {agent.agent_id} with wealth {agent.wealth + tax} pays tax {tax}. Remaining wealth: {agent.wealth}")


grid = Grid(4, 4, (2, 2))
sim = Simulation(0, grid)
# agents = [Agent(agent_id=i, position=(i, i), grid=None, wealth=i*1000) for i in range(1, 9)]
for i in range(1, 9):
    sim.make_agent(i)
    grid.agents[i].wealth = i * 1000
    grid.agents[i].position = (i, i)
static_tax_policy = StaticTaxPolicy(grid)
static_tax_policy.apply_taxes()





