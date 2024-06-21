import numpy as np


class DynamicTaxPolicy:
    def __init__(self, grid, sim, std_dev_threshold=1, discount_rate=0.95):
        self.grid = grid
        self.std_dev_threshold = std_dev_threshold
        self.discount_rate = discount_rate
        self.pretax_incomes = []
        self.posttax_incomes = []
        self.base_tax_rates = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45]
        self.tax_brackets = []
        self.previous_welfare = 0
        self.total_discounted_welfare_change = 0
        self.sim = sim


    def update_tax_brackets(self):
        self.pretax_incomes = [agent.income for agent in self.grid.agents.values()]
        self.posttax_incomes = self.pretax_incomes.copy()
        
        income_values = self.pretax_incomes   # Use pretax incomes for setting tax brackets
        current_std_dev = np.std(income_values)

        # Initialize the adjustment factors to zero (no change)
        adjustment_factors = [0, 0, 0, 0, 0, 0]

        # Check if the standard deviation of income is above the threshold
        if current_std_dev > self.std_dev_threshold:
            # Adjust tax rates based on quartiles: more for higher wealth, less for lower wealth
            adjustment_factors = [-0.1, -0.08, -0.05, 0.05, 0.1, 0.2]
        
        # Calculate adjusted tax rates
        adjusted_tax_rates = [max(0, rate + adj) for rate, adj in zip(self.base_tax_rates, adjustment_factors)]
        
        # Define income quartiles and assign adjusted tax rates
        percentiles = [10, 30, 50, 70, 90, 100]
        quartiles = np.percentile(income_values, percentiles)
        # print("Calculated quartiles:", quartiles)  # Print the quartiles for reference
        self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

    def calculate_tax(self, agent_idx):
        income = self.pretax_incomes[agent_idx - 1]
        print(f"Agent {agent_idx} has pretax income {income}.")
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

    def apply_taxes(self):
        # Collect taxes
        self.update_tax_brackets()
        total_tax_collected = 0
        taxes = []
        for agent_idx, agent in enumerate(self.grid.agents.values()):
            tax = self.calculate_tax(agent_idx)
            agent.wealth -= tax
            taxes.append(tax)
            total_tax_collected += tax
            # print(f"Agent {agent.agent_id} pays tax {tax} with remaining wealth {agent.wealth}.")


        # Distribute taxes evenly
        redistribution_amount = total_tax_collected / len(self.grid.agents) if self.grid.agents else 0
        self.sim.average_tax_values[self.sim.t] = redistribution_amount
        
        for agent_idx, agent in enumerate(self.grid.agents.values()):
            agent.wealth += redistribution_amount
            # print(f"Agent {agent.agent_id} receives {redistribution_amount} from tax revenue, new wealth: {agent.wealth}.")

        # Update posttax incomes after taxes are collected and distributed
        for idx, agent in enumerate(self.grid.agents.values()):
            self.posttax_incomes[idx] = self.pretax_incomes[idx] - taxes[idx] + redistribution_amount
            # print(f"Agent {agent.agent_id} has posttax income {self.posttax_incomes[idx]}.")
        
        current_welfare = self.calculate_social_welfare()
        welfare_change = current_welfare - self.previous_welfare
        discounted_change = (self.discount_rate ** self.sim.t) * welfare_change
        self.total_discounted_welfare_change += discounted_change
        self.previous_welfare = current_welfare

        self.sim.total_discounted_welfare_change[self.sim.t] = self.total_discounted_welfare_change
        # print(f"Total discounted welfare change at step {self.sim.t}: {self.total_discounted_welfare_change}")

    def gini_coefficient(self):
        # Calculate the Gini coefficient
        wealths = list(agent.wealth for agent in self.grid.agents.values())
        n = len(wealths)
        wealth_matrix = np.abs(np.subtract.outer(wealths, wealths))
        gini = wealth_matrix.sum() / (2 * n * np.sum(wealths))
        # print(f"Calculated Gini Coefficient: {gini}") 
        return gini

    # def gini_coefficient(self, use_posttax=True):
    #     # Calculate the Gini coefficient
    #     incomes = np.array(self.posttax_incomes if use_posttax else self.pretax_incomes)
    #     n = len(incomes)
    #     income_matrix = np.abs(np.subtract.outer(incomes, incomes))
    #     gini = income_matrix.sum() / (2 * n * np.sum(incomes))
    #     print(f"Calculated Gini Coefficient: {gini}") 
    #     return gini
    
    def calculate_equality(self):
        # Calculate the equality measure using the Gini index
        n = len(self.grid.agents)
        gini_index = self.gini_coefficient()
        eq_value = 1 - (n / (n - 1)) * gini_index
        self.sim.equality[self.sim.t] = eq_value
        # print(f"Calculated Equality Measure: {eq_value}")
        return eq_value

    # def calculate_productivity(self, use_posttax=True):
    #     # Sum of all incomes which represents the total productivity
    #     total_productivity = np.sum(self.posttax_incomes if use_posttax else self.pretax_incomes)
    #     self.sim.productivity[self.sim.t] = total_productivity
    #     print(f"Total Productivity: {total_productivity}")
    #     return total_productivity

    def calculate_productivity(self):
        # Sum of all incomes which represents the total productivity
        # print("Wealths:", self.grid.agents.values.wealth)
        # print("Wealths:", self.grid.agents.wealth)
        total_productivity = np.sum(agent.wealth for agent in self.grid.agents.values())
        self.sim.productivity[self.sim.t] = total_productivity
        # print(f"Total Productivity: {total_productivity}")
        return total_productivity

    def calculate_social_welfare(self):
        # Calculate the social welfare by equality * productivity
        productivity = self.calculate_productivity()
        social_welfare = self.calculate_equality() * productivity
        self.sim.social_welfare[self.sim.t] = social_welfare
        # print(f"Calculated Social Welfare: {social_welfare}")
        return social_welfare

    def simulate_time_step(self, time_step):
        # Update pretax_incomes
        self.pretax_incomes = [agent.income for agent in self.grid.agents.values()]
        # self.pretax_incomes = [i*1000 for i in range(1, 5)]  # For testing purposes
        # print("Pretax incomes for all agents:", self.pretax_incomes)

        # # Add random fluctuation to pretax_incomes (only for testing welfare change)
        # self.pretax_incomes = [income + income * np.random.uniform(-0.1, 0.1) for income in self.pretax_incomes]
        # print(f"Adjusted pretax incomes at step {time_step}: {self.pretax_incomes}") 

        self.update_tax_brackets()
        self.apply_taxes(time_step)

        # current_welfare = self.calculate_social_welfare()
        # welfare_change = current_welfare - self.previous_welfare
        # discounted_change = (self.discount_rate ** time_step) * welfare_change
        # self.total_discounted_welfare_change += discounted_change
        # self.previous_welfare = current_welfare

        # self.total_discounted_welfare_change_history[time_step] = self.total_discounted_welfare_change
        # self.sim.total_discounted_welfare_change[self.sim.t] = self.total_discounted_welfare_change
        # print(f"Welfare change at step {time_step}: {welfare_change}")
        # print(f"Discounted welfare change at step {time_step}: {discounted_change}")
        # print(f"Total discounted welfare change up to step {time_step}: {self.total_discounted_welfare_change}")


# # Initialize grid and agents
# grid = Grid(4, 4, (2, 2))
# sim = Simulation(0, grid)
# target_std_dev_threshold = 1000

# # Create agents and build houses
# for i in range(1, 5):
#     sim.make_agent(i)

# # Initialize dynamic tax policy and apply taxes
# dynamic_tax_policy = DynamicTaxPolicy(grid, target_std_dev_threshold, discount_rate=0.95)

# # Simulate 4 time steps and update tax policy each time
# for t in range(4): 

#     # Run the simulation step with the updated incomes        
#     dynamic_tax_policy.simulate_time_step(t)
#     print(f"Simulation step {t+1} completed.")




"""
Version 2
"""


# class DynamicTaxPolicy:
#     def __init__(self, grid, std_dev_threshold, discount_rate):
#         self.grid = grid
#         # self.agents = list(grid.agents.values())
#         self.std_dev_threshold = std_dev_threshold
#         self.discount_rate = discount_rate
#         self.pretax_house_incomes = [self.calculate_house_income(agent) for agent in self.grid.agents.values()]
#         self.posttax_house_incomes = self.pretax_house_incomes.copy()
#         self.base_tax_rates = [0.1, 0.2, 0.3, 0.4]  # Base tax rates
#         self.tax_brackets = []
#         self.previous_welfare = 0
#         self.total_discounted_welfare_change = 0
#         print("House pretax incomes for all agents:", self.pretax_house_incomes)
    
#     def calculate_house_income(self, agent):
#         # Calculate the total income from the houses owned by the agent
#         return sum([house.income_per_timestep for house in agent.houses])

#     def update_tax_brackets(self):
#         income_values = self.pretax_house_incomes   # Use pretax incomes for setting tax brackets
#         current_std_dev = np.std(income_values)

#         # Initialize the adjustment factors to zero (no change)
#         adjustment_factors = [0, 0, 0, 0]

#         # Check if the standard deviation of income is above the threshold
#         if current_std_dev > self.std_dev_threshold:
#             # Adjust tax rates based on quartiles: more for higher wealth, less for lower wealth
#             adjustment_factors = [-0.02, -0.01, 0.01, 0.02]
        
#         # Calculate adjusted tax rates
#         adjusted_tax_rates = [max(0, rate + adj) for rate, adj in zip(self.base_tax_rates, adjustment_factors)]
        
#         # Define income quartiles and assign adjusted tax rates
#         quartiles = np.percentile(income_values, [25, 50, 75, 100])
#         print("Calculated quartiles:", quartiles)  # Print the quartiles for reference
#         self.tax_brackets = [(quartiles[i], adjusted_tax_rates[i]) for i in range(len(quartiles))]

#     def calculate_tax(self, agent):
#         income = self.pretax_house_incomes[agent.agent_id - 1]
#         print(f"Agent {agent.agent_id} has pretax income {income}.")
#         tax = 0
#         previous_bound = 0
#         for upper_bound, tax_rate in self.tax_brackets:
#             if income > previous_bound:
#                 taxable_income = min(income - previous_bound, upper_bound - previous_bound)
#                 tax += taxable_income * tax_rate
#                 previous_bound = upper_bound
#             else:
#                 break
#         return tax


#     def collect_and_distribute_taxes(self):
#         # Collect taxes
#         total_tax_collected = 0
#         taxes = []
#         for agent_idx, agent in enumerate(self.grid.agents.values()):
#             tax = self.calculate_tax(agent)
#             agent.wealth -= tax
#             taxes.append(tax)
#             total_tax_collected += tax
#             print(f"Agent {agent.agent_id} pays tax {tax} with remaining wealth {agent.wealth}.")


#         # Distribute taxes evenly
#         redistribution_amount = total_tax_collected / len(self.grid.agents) if self.grid.agents else 0
#         for agent_idx, agent in enumerate(self.grid.agents.values()):
#             agent.wealth += redistribution_amount
#             print(f"Agent {agent.agent_id} receives {redistribution_amount} from tax revenue, new wealth: {agent.wealth}.")

#         # Update posttax incomes after taxes are collected and distributed
#         for idx, agent in enumerate(self.grid.agents.values()):
#             self.posttax_house_incomes[idx] = self.pretax_house_incomes[idx] - taxes[idx] + redistribution_amount
#             print(f"Agent {agent.agent_id} has posttax income {self.posttax_house_incomes[idx]}.")

#     def gini_coefficient(self, use_posttax=True):
#         # Calculate the Gini coefficient
#         incomes = np.array(self.posttax_house_incomes if use_posttax else self.pretax_house_incomes)
#         n = len(incomes)
#         income_matrix = np.abs(np.subtract.outer(incomes, incomes))
#         gini = income_matrix.sum() / (2 * n * np.sum(incomes))
#         print(f"Calculated Gini Coefficient: {gini}") 
#         return gini

#     def calculate_equality(self):
#         # Calculate the equality measure using the Gini index
#         n = len(self.grid.agents)
#         gini_index = self.gini_coefficient()
#         eq_value = 1 - (n / (n - 1)) * gini_index
#         print(f"Calculated Equality Measure: {eq_value}")
#         return eq_value

#     def calculate_productivity(self, use_posttax=True):
#         # Sum of all house incomes which represents the total productivity
#         total_productivity = np.sum(self.posttax_house_incomes if use_posttax else self.pretax_house_incomes)
#         print(f"Total Productivity: {total_productivity}")
#         return total_productivity

#     def calculate_social_welfare(self):
#         # Calculate the social welfare by equality * productivity
#         social_welfare = self.calculate_equality() * self.calculate_productivity()
#         print(f"Calculated Social Welfare: {social_welfare}")
#         return social_welfare


#     def simulate_time_step(self, time_step):
#         self.update_tax_brackets()
#         self.collect_and_distribute_taxes()

#         self.pretax_house_incomes = [self.calculate_house_income(agent) for agent in self.grid.agents.values()]
#         # print(f"Pretax incomes at step {time_step}: {self.pretax_house_incomes}") 

#         self.update_tax_brackets()
#         self.collect_and_distribute_taxes()

#         current_welfare = self.calculate_social_welfare()
#         welfare_change = current_welfare - self.previous_welfare
#         discounted_change = (self.discount_rate ** time_step) * welfare_change
#         self.total_discounted_welfare_change += discounted_change
#         self.previous_welfare = current_welfare

#         print(f"Welfare change at step {time_step}: {welfare_change}")
#         print(f"Discounted welfare change at step {time_step}: {discounted_change}")
#         print(f"Total discounted welfare change up to step {time_step}: {self.total_discounted_welfare_change}")


# # Initialize grid and agents
# grid = Grid(4, 4, (2, 2))
# sim = Simulation(0, grid)
# target_std_dev_threshold = 1

# # Create agents and build houses
# for i in range(1, 9):
#     sim.make_agent(i)
#     for j in range(i):
#         grid.agents[i].build_house() 

# # Initialize dynamic tax policy and apply taxes
# dynamic_tax_policy = DynamicTaxPolicy(grid, target_std_dev_threshold, discount_rate=0.95)

# # Simulate 10 time steps and update tax policy each time
# for t in range(10): 
#     # Update each house's income_per_timestep before simulating the tax policy
#     for agent in grid.agents.values():
#         for house in agent.houses:
#             # Assume income can randomly fluctuate by up to 10% of its current value
#             fluctuation = house.income_per_timestep * np.random.uniform(-0.1, 0.1)
#             house.income_per_timestep += fluctuation

#     # Run the simulation step with the updated incomes        
#     dynamic_tax_policy.simulate_time_step(t)
#     print(f"Simulation step {t+1} completed.")


"""
Version 1
"""

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


