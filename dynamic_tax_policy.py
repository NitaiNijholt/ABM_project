from grid import Grid
from simulation import Simulation
import numpy as np

class DynamicTaxPolicy:
    def __init__(self, grid, std_dev_threshold):
        self.grid = grid
        # self.agents = list(grid.agents.values())
        self.std_dev_threshold = std_dev_threshold
        self.house_incomes = [self.calculate_house_income(agent) for agent in self.grid.agents.values()]
        self.base_tax_rates = [0.1, 0.2, 0.3, 0.4]  # Base tax rates
        self.tax_brackets = []
        print("House incomes for all agents:", self.house_incomes)
    
    def calculate_house_income(self, agent):
        # Calculate the total income from the houses owned by the agent
        return sum([house.income_per_timestep for house in agent.houses])

    def update_tax_brackets(self):
        income_values = self.house_incomes
        current_std_dev = np.std(income_values)

        # Initialize the adjustment factors to zero (no change)
        adjustment_factors = [0, 0, 0, 0]

        # Check if the standard deviation of income is above the threshold
        if current_std_dev > self.std_dev_threshold:
            # Adjust tax rates based on quartiles: more for higher wealth, less for lower wealth
            adjustment_factors = [-0.02, -0.01, 0.01, 0.02]
        
        # Calculate adjusted tax rates
        adjusted_tax_rates = [max(0, rate + adj) for rate, adj in zip(self.base_tax_rates, adjustment_factors)]
        
        # Define income quartiles and assign adjusted tax rates
        quartiles = np.percentile(income_values, [25, 50, 75, 100])
        print("Calculated quartiles:", quartiles)  # Print the quartiles for reference
        self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

    def calculate_tax(self, agent):
        income = self.calculate_house_income(agent)
        tax = 0
        previous_bound = 0
        for upper_bound, tax_rate in self.tax_brackets:
            if income > previous_bound:
                taxable_income = min(income - previous_bound, upper_bound - previous_bound)
                tax += taxable_income * tax_rate
                previous_bound = upper_bound
            else:
                break
        return tax


    def collect_and_distribute_taxes(self):
        # Collect taxes
        total_tax_collected = 0
        for agent_idx, agent in enumerate(self.grid.agents.values()):
            tax = self.calculate_tax(agent)
            agent.wealth -= tax
            total_tax_collected += tax
            print(f"Agent {agent.agent_id} pays tax {tax} with remaining wealth {agent.wealth}.")


        # Distribute taxes evenly
        redistribution_amount = total_tax_collected / len(self.grid.agents) if self.grid.agents else 0
        for agent_idx, agent in enumerate(self.grid.agents.values()):
            agent.wealth += redistribution_amount
            print(f"Agent {agent.agent_id} receives {total_tax_collected / len(self.grid.agents)} from tax revenue, new wealth: {agent.wealth}.")


    def gini_coefficient(self):
        # Calculate the Gini coefficient
        incomes = np.array(self.house_incomes)
        n = len(incomes)
        income_matrix = np.abs(np.subtract.outer(incomes, incomes))
        gini = income_matrix.sum() / (2 * n * np.sum(incomes))
        print(f"Calculated Gini Coefficient: {gini}") 
        return gini

    def calculate_equality(self):
        # Calculate the equality measure using the Gini index
        n = len(self.grid.agents)
        gini_index = self.gini_coefficient()
        eq_value = 1 - (n / (n - 1)) * gini_index
        print(f"Calculated Equality Measure: {eq_value}")
        return eq_value

    def calculate_productivity(self):
        # Sum of all house incomes which represents the total productivity
        total_productivity = np.sum(self.house_incomes)
        print(f"Total Productivity: {total_productivity}")
        return total_productivity

    def calculate_social_welfare(self):
        # Calculate the social welfare by equality * productivity
        social_welfare = self.calculate_equality() * self.calculate_productivity()
        print(f"Calculated Social Welfare: {social_welfare}")
        return social_welfare







    def simulate_time_step(self):
        self.update_tax_brackets()
        self.collect_and_distribute_taxes()
        self.gini_coefficient()
        self.calculate_equality()
        self.calculate_productivity()
        self.calculate_social_welfare()



# Initialize grid and agents
grid = Grid(4, 4, (2, 2))
sim = Simulation(0, grid)
target_std_dev_threshold = 1

# Create agents and build houses
for i in range(1, 9):
    sim.make_agent(i)
    for j in range(i):
        grid.agents[i].build_house() 

# Initialize dynamic tax policy and apply taxes
dynamic_tax_policy = DynamicTaxPolicy(grid, target_std_dev_threshold)

# Simulate 10 time steps and update tax policy each time
for _ in range(10): 
    dynamic_tax_policy.simulate_time_step()
    print(f"Simulation step {_+1} completed.")


# class DynamicTaxPolicy:
#     def __init__(self, grid, std_dev_threshold):
#         self.agents = list(grid.agents.values())
#         # self.agents = []
#         # for agent in grid.agents.values():
#         #     self.agents.append(agent)
#         self.std_dev_threshold = std_dev_threshold
#         self.base_tax_rates = [0.1, 0.2, 0.3, 0.4]  # Base tax rates
#         self.tax_brackets = []

#     def update_tax_brackets(self):
#         wealth_values = [agent.wealth for agent in self.agents]
#         current_std_dev = np.std(wealth_values)

#         # Initialize the adjustment factors to zero (no change)
#         adjustment_factors = [0, 0, 0, 0]

#         # Check if the standard deviation of wealth is above the threshold
#         if current_std_dev > self.std_dev_threshold:
#             # Adjust tax rates based on quartiles: more for higher wealth, less for lower wealth
#             adjustment_factors = [-0.02, -0.01, 0.01, 0.02]
        
#         # Calculate adjusted tax rates
#         adjusted_tax_rates = [max(0, min(rate + adjustment_factors[i], 1))
#                               for i, rate in enumerate(self.base_tax_rates)]
        
#         # Define wealth quartiles and assign adjusted tax rates
#         quartiles = np.percentile(wealth_values, [25, 50, 75, 100])
#         self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

#     def collect_and_distribute_taxes(self):
#         # Collect taxes
#         total_tax_collected = 0
#         for agent in self.agents:
#             tax = self.calculate_tax(agent)
#             agent.wealth -= tax
#             total_tax_collected += tax
#             print(f"Agent {agent.agent_id} pays tax {tax}. Remaining wealth: {agent.wealth}")


#         # Distribute taxes evenly
#         redistribution_amount = total_tax_collected / len(self.agents) if self.agents else 0
#         for agent in self.agents:
#             agent.wealth += redistribution_amount
#             print(f"Agent {agent.agent_id} with adjusted wealth {agent.wealth}")

#     def calculate_tax(self, agent):
#         # Calculate tax based on wealth and corresponding bracket
#         for upper_bound, tax_rate in self.tax_brackets:
#             if agent.wealth <= upper_bound:
#                 return agent.wealth * tax_rate

#     def simulate_time_step(self):
#         self.update_tax_brackets()
#         self.collect_and_distribute_taxes()


# # Initialize agents

# grid = Grid(4, 4, (2, 2))
# sim = Simulation(0, grid)
# target_std_dev_threshold = 1000 

# # agents = [Agent(agent_id=i, position=(i, i), grid=None, wealth=i*1000 + 10000) for i in range(1, 5)]

# for i in range(1, 9):
#     sim.make_agent(i)
#     grid.agents[i].wealth = i * 1000
#     grid.agents[i].position = (i, i)

# dynamic_tax_policy = DynamicTaxPolicy(grid, target_std_dev_threshold)
# for _ in range(10):  # Simulate 10 time steps
#     dynamic_tax_policy.simulate_time_step()


