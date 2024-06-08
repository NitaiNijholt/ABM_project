from agent import Agent
import numpy as np

class DynamicTaxPolicy:
    def __init__(self, agents, target_wealth):
        self.agents = agents
        self.target_wealth = target_wealth
        self.base_tax_rates = [0.1, 0.2, 0.3, 0.4]  # Base tax rates
        self.tax_brackets = []

    def update_tax_brackets(self):
        current_average_wealth = sum(agent.wealth for agent in self.agents) / len(self.agents)
        if current_average_wealth < self.target_wealth:
            adjustment_factor = -0.02
        else:
            adjustment_factor = 0.02
        
        adjusted_tax_rates = [max(0, rate + adjustment_factor) for rate in self.base_tax_rates]
        
        wealth_values = sorted(agent.wealth for agent in self.agents)
        quartiles = np.percentile(wealth_values, [25, 50, 75, 100])
        self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

    def collect_and_distribute_taxes(self):
        # Collect taxes
        total_tax_collected = 0
        for agent in self.agents:
            tax = self.calculate_tax(agent)
            agent.wealth -= tax
            total_tax_collected += tax

        # Distribute taxes evenly
        if len(self.agents) > 0:
            redistribution_amount = total_tax_collected / len(self.agents)
            for agent in self.agents:
                agent.wealth += redistribution_amount

        # Apply taxes again after redistribution
        for agent in self.agents:
            tax = self.calculate_tax(agent)
            agent.wealth -= tax
            print(f"Agent {agent.agent_id} with adjusted wealth pays tax {tax}. Remaining wealth: {agent.wealth}")

    def calculate_tax(self, agent):
        for upper_bound, tax_rate in self.tax_brackets:
            if agent.wealth <= upper_bound:
                return agent.wealth * tax_rate
        return 0

    def simulate_time_step(self):
        self.update_tax_brackets()
        self.collect_and_distribute_taxes()

# Initialize agents
agents = [Agent(agent_id=i, position=(i, i), grid=None, wealth=i*1000 + 10000) for i in range(1, 5)]
target_wealth = 15000  # Set the target average wealth
dynamic_policy = DynamicTaxPolicy(agents, target_wealth)

for _ in range(10):  # Simulate 10 time steps
    dynamic_policy.simulate_time_step()

