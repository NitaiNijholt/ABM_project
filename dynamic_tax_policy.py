from grid import Grid
from simulation import Simulation
import numpy as np

class DynamicTaxPolicy:
    def __init__(self, grid, std_dev_threshold):
        self.agents = list(grid.agents.values())
        # self.agents = []
        # for agent in grid.agents.values():
        #     self.agents.append(agent)
        self.std_dev_threshold = std_dev_threshold
        self.base_tax_rates = [0.1, 0.2, 0.3, 0.4]  # Base tax rates
        self.tax_brackets = []

    def update_tax_brackets(self):
        wealth_values = [agent.wealth for agent in self.agents]
        current_std_dev = np.std(wealth_values)

        # Initialize the adjustment factors to zero (no change)
        adjustment_factors = [0, 0, 0, 0]

        # Check if the standard deviation of wealth is above the threshold
        if current_std_dev > self.std_dev_threshold:
            # Adjust tax rates based on quartiles: more for higher wealth, less for lower wealth
            adjustment_factors = [-0.02, -0.01, 0.01, 0.02]
        
        # Calculate adjusted tax rates
        adjusted_tax_rates = [max(0, min(rate + adjustment_factors[i], 1))
                              for i, rate in enumerate(self.base_tax_rates)]
        
        # Define wealth quartiles and assign adjusted tax rates
        quartiles = np.percentile(wealth_values, [25, 50, 75, 100])
        self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

    def collect_and_distribute_taxes(self):
        # Collect taxes
        total_tax_collected = 0
        for agent in self.agents:
            tax = self.calculate_tax(agent)
            agent.wealth -= tax
            total_tax_collected += tax
            print(f"Agent {agent.agent_id} pays tax {tax}. Remaining wealth: {agent.wealth}")


        # Distribute taxes evenly
        redistribution_amount = total_tax_collected / len(self.agents) if self.agents else 0
        for agent in self.agents:
            agent.wealth += redistribution_amount
            print(f"Agent {agent.agent_id} with adjusted wealth {agent.wealth}")

    def calculate_tax(self, agent):
        # Calculate tax based on wealth and corresponding bracket
        for upper_bound, tax_rate in self.tax_brackets:
            if agent.wealth <= upper_bound:
                return agent.wealth * tax_rate

    def simulate_time_step(self):
        self.update_tax_brackets()
        self.collect_and_distribute_taxes()


# Initialize agents

grid = Grid(4, 4, (2, 2))
sim = Simulation(0, grid)
target_std_dev_threshold = 1000 

# agents = [Agent(agent_id=i, position=(i, i), grid=None, wealth=i*1000 + 10000) for i in range(1, 5)]

for i in range(1, 9):
    sim.make_agent(i)
    grid.agents[i].wealth = i * 1000
    grid.agents[i].position = (i, i)

dynamic_tax_policy = DynamicTaxPolicy(grid, target_std_dev_threshold)
for _ in range(10):  # Simulate 10 time steps
    dynamic_tax_policy.simulate_time_step()


