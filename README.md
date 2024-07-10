# ABM Project

This repository contains the files for the Agent Based Modeling project, for the Agent Based Modeling course, University of Amsterdam 2024.
## Table of Contents

- [ABM Project](#abm-project)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Features](#features)
  - [License](#license)
  - [Contact Information](#contact-information)
  - [References](#references)
  
 

## Abstract

This project uses an Agent-Based Model (ABM) to investigate emergence of specialized agent behavior when assets that provide income over time and wealth heterogeneity are introduced. As secondary research question, we simulate the effects of static and redistributive dynamic tax policies on agent behaviors and wealth distribution. Agents, possessing assets like houses that generate income, can gather resources and engage in buying, selling, and building houses, which provide increased income when clustered. By contrasting Expected Value Maximizing agents with those using Neural Networks evolved through Neuroevolution, the study examines emergent behaviors in response to different tax policies, reflecting initial wealth disparities in the Netherlands.

Findings show that agents exhibit emergent behavior by forming distinct economic classes: a wealthy building class, a low-activity poor class, and either a middle class (for Expected Value Maximizing agents) or a trader class (for Neuroevolution agents). Sensitivity analysis reveals that parameters related to the number, cost, and income of assets significantly influence the persistence of these classes.

The study notes significant differences in Gini coefficients for wealth distribution across tax policies for expectation-maximizing agents, though these differences have a low effect size. Neuroevolution agents showed lower Gini coefficients and productivity, and neither agent’s income distribution aligned with the Netherlands’ data under the Dutch tax policy.

The main contribution of this study is the emergence of specialized behavior among different agent intelligence types and tax policies, with initial wealth as the primary form of heterogeneity. As wealth is easier to measure than skill, our model is easier to parameterize while still showing the emergence of building behaviors found in other research employing skill heterogeneity. Future research should explore the impact of more aggressive redistributive tax policies on inequality and examine how class persistence changes through the parameter space by examining pairwise cluster feature differences of classes instead of the total Euclidean distance between the cluster feature vectors.



## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

Ensure you have the following installed:

- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Required Python libraries (listed in `requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - networkx
  - more as specified in `requirements.txt`

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/NitaiNijholt/ABM_project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ABM_project
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Features

This project includes the following features:

- **agent.py**: Defines the `Agent` class, responsible for individual agent behaviors and interactions in the dynamic market using an EV-based decision model.
- **agent_static_market.py**: Defines the `Agent_static_market` class, which models agent behaviors in a static market environment using an EV-based decision model.
- **intelligent_agent_dynamic_market.py**: Defines the `IntelligentAgent` class, modeling agents that make decisions based on network influences in a dynamic market.
- **grid.py**: Defines the `Grid` class, which manages the spatial environment where agents operate.
- **house.py**: Defines the `House` class, representing housing units within the simulation.
- **dynamic_tax_policy.py**: Implements the `DynamicTaxPolicy` class, simulating a tax policy that adjusts dynamically based on the market conditions.
- **static_tax_policy.py**: Implements the `StaticTaxPolicy` class, simulating a fixed tax policy regardless of market changes.
- **market.py**: Defines the `Market` class, simulating a static market environment where agents interact.
- **orderbook.py**: Defines the `OrderBooks` class, which simulates the order books in a dynamic market environment.
- **network.py**: Defines the `Network` class, used for modeling intelligent agents' decision-making processes based on network connections and influences.
- **simulation.py**: Contains the `Simulation` class for running simulations of EV-based agents in the modeled environment.
- **simulation_evolve.py**: Contains the `Simulation` class for running simulations with network-based agents, incorporating evolutionary computing techniques for agent adaptation.
- **analysis_aggregate_feature_importances_and_ANOVA.py**: Contains scripts for analyzing aggregate feature importance and conducting ANOVA tests.
- **analysis_ks_test_income_disitributions.py**: Analyzes income distributions using the KS test.
- **analysis_multiruns_ks_test.py**: Validates multiple simulation runs using the KS test.
- **analysis_validation_multiruns_ks_test.py**: Further validation of multiple simulation runs using the KS test.
- **fit_income_data_external.ipynb**: Jupyter notebook for fitting external income data.
- **global_sensitivty_indices_plotter.py**: Plots global sensitivity indices.
- **multi_run_param_simulator.py**: Simulates multiple runs with varying parameters and conducts Agglomorative clustering on the action space.
- **multi_run_param_simulator_sensitivity.py**: Adds sensitivity analysis to multi-run simulations and conducts Agglomorative clustering on the action space.
- **plot_sensitivity_indices.py**: Plots sensitivity indices for various metrics.
- **sensitivity_analysis_oat.ipynb**: Jupyter notebook for One-At-A-Time (OAT) sensitivity analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or suggestions, please contact:

Project Link: [https://github.com/NitaiNijholt/ABM_project](https://github.com/NitaiNijholt/ABM_project)

## References

[1] Volker Grimm et al. “The ODD protocol: A review and first update”. In: Ecological Modelling 221.23 (2010), pp. 2760–2768. issn: 0304-3800. doi: [https://doi.org/10.1016/j.ecolmodel.2010.08.019](https://doi.org/10.1016/j.ecolmodel.2010.08.019). url: [http://www.sciencedirect.com/science/article/pii/S030438001000414X](http://www.sciencedirect.com/science/article/pii/S030438001000414X).
