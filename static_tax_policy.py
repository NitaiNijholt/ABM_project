# from grid import Grid
# from simulation import Simulation
import numpy as np


class StaticTaxPolicy:
    def __init__(self, grid, sim, discount_rate=0.95):
        self.grid = grid
        # self.pretax_incomes = [i*1000 for i in range(1, 9)]   # test data
        self.discount_rate = discount_rate
        self.pretax_incomes = []
        self.posttax_incomes = []
        self.tax_brackets = []
        self.previous_welfare = 0
        self.total_discounted_welfare_change = 0
        self.sim = sim
        # print("Initial pretax incomes:", self.pretax_incomes)

    def calculate_tax_brackets(self):
        self.pretax_incomes = [agent.income for agent in self.grid.agents.values()]
        self.posttax_incomes = self.pretax_incomes.copy()
        # Determine income brackets based on quantiles
        percentiles = [5.12, 65.67, 96.93, 100]
        quartiles = np.percentile(self.pretax_incomes, percentiles)
        # print("Calculated quartiles:", quartiles)  # Print the quartiles for reference
        tax_rates = [0, 0.1903, 0.3693, 0.4950]

        # Create a list of (upper_bound, tax_rate) tuples
        tax_brackets = [(quartiles[i], tax_rates[i]) for i in range(len(quartiles))]
        # print("Updated tax brackets based on pretax incomes:", tax_brackets)
        return tax_brackets

    def calculate_tax(self, agent_id):
        # Calculate tax based on cumulative brackets for progressive taxation
        tax_brackets = self.calculate_tax_brackets()
        income = self.pretax_incomes[agent_id]
#         print(f"Agent {agent_id+1} with pretax income {income}.")
        tax = 0
        previous_bound = 0
        for i, (upper_bound, tax_rate) in enumerate(tax_brackets):
            if income > previous_bound:
                taxable_income = min(income - previous_bound, upper_bound - previous_bound)
                tax += taxable_income * tax_rate
                previous_bound = upper_bound
            else:
                break

        return tax

    def apply_taxes(self):
        # Apply tax to each agent and adjust their wealth
        total_tax_collected = 0
        taxes = []
        for agent_id, agent in enumerate(self.grid.agents.values()):
            tax = self.calculate_tax(agent_id)
            total_tax_collected += tax
            agent.wealth -= tax
            taxes.append(tax)
            

        # Redistribution of tax revenue
        redistribution_amount = total_tax_collected / len(self.grid.agents) if self.grid.agents else 0
        self.sim.average_tax_values[self.sim.t] = redistribution_amount
        
        for agent_id, agent in enumerate(self.grid.agents.values()):
            agent.wealth += redistribution_amount
            # print(f"Agent {agent.agent_id} receives {redistribution_amount} from tax revenue, new wealth: {agent.wealth}.")
            self.posttax_incomes[agent_id] = self.pretax_incomes[agent_id] - taxes[agent_id] + redistribution_amount
#         print(f"tax: {tax}")
#         print(f"Posttax house incomes after redistribution: {self.posttax_incomes}")

        current_welfare = self.calculate_social_welfare()
        welfare_change = current_welfare - self.previous_welfare
        discounted_change = (self.discount_rate ** self.sim.t) * welfare_change
        self.total_discounted_welfare_change += discounted_change
        self.previous_welfare = current_welfare

        self.sim.total_discounted_welfare_change[self.sim.t] = self.total_discounted_welfare_change

    def gini_coefficient(self):
        # Calculate the Gini coefficient
        wealths = list(agent.wealth for agent in self.grid.agents.values())
        n = len(wealths)
        wealth_matrix = np.abs(np.subtract.outer(wealths, wealths))
        gini = wealth_matrix.sum() / (2 * n * np.sum(wealths))
#         print(f"Calculated Gini Coefficient: {gini}") 
        return gini
    

    def calculate_equality(self):
        # Calculate the equality measure using the Gini index
        n = len(self.grid.agents)
        gini_index = self.gini_coefficient()
        eq_value = 1 - (n / (n - 1)) * gini_index
        self.sim.equality[self.sim.t] = eq_value
#         print(f"Calculated Equality Measure: {eq_value}")
        return eq_value

    def calculate_productivity(self, use_posttax=True):
        # Sum of all incomes which represents the total productivity
        total_productivity = np.sum(agent.wealth for agent in self.grid.agents.values())
        self.sim.productivity[self.sim.t] = total_productivity
        # print(f"Total Productivity: {total_productivity}")
        return total_productivity

    def calculate_social_welfare(self):
        # Calculate the social welfare by equality * productivity
        social_welfare = self.calculate_equality() * self.calculate_productivity()
        self.sim.social_welfare[self.sim.t] = social_welfare
        # print(f"Calculated Social Welfare: {social_welfare}")
        return social_welfare

    def simulate_time_step(self, time_step):
        self.pretax_incomes = [agent.income for agent in self.grid.agents.values()]

        self.calculate_tax_brackets()
        self.apply_taxes(time_step)

