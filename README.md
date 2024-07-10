# ABM Project

This repository contains the files for the Agent Based Modeling project, for the Agent Based Modeling course, University of Amsterdam 2024.

## Table of Contents

- [ABM Project](#abm-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Features](#features)
  - [License](#license)
  - [Contact Information](#contact-information)
  - [References](#references)

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

Ensure you have the following installed:

- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Required Python libraries (listed in `requirements.txt`):
  - numpy
  - scipy
  - pandas
  - matplotlib
  - statsmodels
  - sklearn
  - SALib
  - psutil
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

- **agent.py**: Defines the `Agent` class (using 'dynamic market' and 'EV based decision').
- **agent_static_market.py**: Defines the `Agent_static_market` class (using the 'static market' and 'EV-based decision').
- **intelligent_agent_dynamic_market.py**: Defines the `IntelligentAgent` class (using the 'dynamic market' and 'network-based decision').
- **grid.py**: Defines the `Grid` class.
- **house.py**: Defines the `House` class.
- **dynamic_tax_policy.py**: Defines the `DynamicTaxPolicy` class.
- **static_tax_policy.py**: Defines the `StaticTaxPolicy` class.
- **market.py**: Defines the `Market` (static market) class.
- **orderbook.py**: Defines the `OrderBooks` (dynamic market) class.
- **network.py**: Defines the `Network` class for intelligent agents' decision making and evolution.
- **simulation.py**: Defines the `Simulation` class for simulating the EV-based agents.
- **simulation_evolve.py**: Defines the `Simulation` class for simulating the network-based agents and performing evolutional computing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or suggestions, please contact:

Nitai Nijholt
University of Amsterdam
Email: 

Project Link: [https://github.com/NitaiNijholt/ABM_project](https://github.com/NitaiNijholt/ABM_project)

## References

[1] Volker Grimm et al. “The ODD protocol: A review and first update”. In: Ecological Modelling 221.23 (2010), pp. 2760–2768. issn: 0304-3800. doi: [https://doi.org/10.1016/j.ecolmodel.2010.08.019](https://doi.org/10.1016/j.ecolmodel.2010.08.019). url: [http://www.sciencedirect.com/science/article/pii/S030438001000414X](http://www.sciencedirect.com/science/article/pii/S030438001000414X).