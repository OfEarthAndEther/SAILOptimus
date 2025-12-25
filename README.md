# SAILOptimus: AI-Enabled Logistics Optimizer

Cost-optimal vessel scheduling and port-plant linkage optimization for steel supply chains. AI-driven decision support system integrating maritime logistics, inventory management, and production planning through mixed-integer linear programming with machine learning-based demand forecasting.

## SIH 2025

- **Ministry:** Ministry of Steel (MoS)
- **Theme:** Transportation & Logistics
- **PS Code:** 25209
- **Problem we're tackling:** Al-Enabled Logistics Optimizer for Cost-Optimal Vessel Scheduling and Port-Plant Linkage in Steel Supply Chain

## Team Details

- **Team Name:** RailOptimus
- **Team ID:** 102098
- **Institute Name:** Netaji Subhas University of Technology

**Team Leader:** [@poorva-tehlan](https://github.com/poorva-tehlan)

**Team Members:**

- **MEMBER_1**  - Poorva Tehlan - 2024UCA1891 - [@poorva-tehlan](https://github.com/poorva-tehlan)
- **MEMBER_2** - Bhavishya Maheshwari - 2024UCA1884 - [@BhavishyaMaheshwari](https://github.com/BhavishyaMaheshwari)
- **MEMBER_3** - Gauransh Gupta - 2024UCA1944 - [@GauranshGupta](https://github.com/GauranshGupta)
- **MEMBER_4** - Arnav Gupta - 2024UCA1885 - [@Arngithub407](https://github.com/Arngithub407)
- **MEMBER_5** - Joseph Jisso Aliyath - 2024UCA1957 - [@JosephJisso](https://github.com/JosephJisso)
- **MEMBER_6** - Srishti - 2024UCA1923 - [@OfEarthAndEther](https://github.com/OfEarthAndEther)

## Project Links

- **SIH Presentation:** [Presentation Link](https://drive.google.com/file/d/1mGwUPXI5NUwx-1gBuNRk9LEPiZozcTrq/view?usp=sharing)
- **Video Demonstration:** [Video Link](https://youtu.be/Z64fNkR3K2w)

## Overview

SAILOptimus optimizes complex supply chain decisions across three interdependent domains:

- **Vessel Scheduling**: Cost-optimal routing, capacity allocation, and frequency optimization
- **Port Operations**: Berth scheduling and anchorage management with congestion mitigation
- **Plant-Port Linkage**: Integrated inventory and production scheduling across geographically distributed mills

The system provides real-time optimization recommendations balancing:
- Total logistics cost (freight, handling, inventory)
- Supply reliability and service levels
- Inventory carrying costs
- Production efficiency and capacity utilization

**Key Results**: 15–28% cost reduction vs. baseline heuristics while maintaining 99%+ service reliability across dynamic scenarios.

## Features

- **Integrated Supply Chain Optimization**: Simultaneous optimization of vessel routing, port operations, and plant scheduling
- **AI-Powered Demand Forecasting**: LSTM-based demand prediction with seasonal decomposition
- **Congestion-Aware Scheduling**: Real-time port congestion detection using AIS data
- **Scenario Simulation**: Monte Carlo simulation of demand volatility, port disruptions, and vessel availability
- **Multi-Objective Optimization**: Weighted cost, service level, and sustainability metrics
- **Real-Time Decision Support**: Interactive dashboards for logistics planners and supply chain managers
- **Scalable Architecture**: Support for multi-port, multi-plant networks with 50+ vessel fleet

## Requirements

- **Python 3.8+** or **MATLAB R2020a+**
- MILP Solver: Gurobi (recommended) or CPLEX/CBC
- 8 GB RAM minimum; 16+ GB for large-scale simulations
- Historical AIS data and demand/inventory datasets

## Quick Start

### Python
```
from sailoptimus import SupplyChainOptimizer, ScenarioAnalyzer
```

Load network topology (ports, plants, vessels)
```network = SupplyChainOptimizer.load_network('config/network.json')```

Configure optimization
```
config = {
'objective': 'minimize_total_cost',
'constraints': ['service_level >= 0.99', 'vessel_utilization <= 0.95'],
'time_horizon': 180, # days
'planning_periods': 30 # daily reoptimization
}
```

Run optimization
```
optimizer = SupplyChainOptimizer(network, config)
solution = optimizer.solve(demand_forecast=forecast_df, port_status=port_data)

print(f"Total Cost: ${solution.cost:.2M} | Service Level: {solution.service_level:.2%}")
solution.export_recommendations('output/recommendations.xlsx')
```

### MATLAB
```
% Load and configure network
network = sailoptimus_load_network('config/network.mat');
config = sailoptimus_default_config();
config.objective = 'cost_minimization';
config.service_level_target = 0.99;

% Run optimization with demand forecast
[solution, metrics] = sailoptimus_optimize(network, demand_forecast, config);

% Visualize schedule
sailoptimus_visualize_gantt(solution.vessel_schedule);
sailoptimus_plot_inventory(solution.plant_inventory);
```

## Core Algorithms

### Supply Chain Model

**Decision Variables:**
- Vessel deployment schedule and routing (route selection, frequency)
- Port berth allocation and vessel sequencing
- Plant production schedules and inventory targets
- Order fulfillment timing across distribution centers

**Constraints:**
- Vessel capacity and voyage timing
- Port berth availability and turnaround time
- Plant production capacity and lead times
- Inventory bounds and service level targets
- Demand satisfaction and backorder limits

**Objective:**
Minimize: Σ(freight costs + port handling + inventory carrying + backorder penalty)
Subject to: Demand satisfaction, capacity, operational, service level constraints

## Citation

If you use SAILOptimus in research or production, please cite:
```
@misc{sailoptimus2025,
title={SAILOptimus: AI-Enabled Logistics Optimizer for Cost-Optimal Vessel Scheduling and Port-Plant Linkage in Steel Supply Chain},
author={Team OfEarthAndEther},
note={Smart India Hackathon 2025 Solution},
url={https://github.com/OfEarthAndEther/SAILOptimus},
year={2025}
}
```


Or reference the methodology paper:
- [Full Research Paper (PDF)](https://project-archive.inf.ed.ac.uk/ug4/20233777/ug4_proj.pdf)

## License

MIT License. See LICENSE file.

## Support

- **Issues & Bugs**: [GitHub Issues](https://github.com/OfEarthAndEther/SAILOptimus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OfEarthAndEther/SAILOptimus/discussions)
- **Documentation**: [ReadTheDocs](https://sailoptimus.readthedocs.io/)

## Contributors

Developed by **Team RailOptimus** for Smart India Hackathon 2025.
