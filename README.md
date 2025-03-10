# **Sustainable Driving Operations of Urban Plug-in Hybrid Buses Considering Restricted Emission Mapping Zones**  
This repository contains the source code and data required to replicate the study *"Sustainable Driving Operations of Urban Plug-in Hybrid Buses Considering Restricted Emission Mapping Zones."*

## **Repository Structure**  

The repository consists of the following main directories:

- **`results`**  
  This directory contains the experimental results presented in the paper, along with various Python 3 scripts for generating graphs and visualizations.

- **`bus_lines`**  
  This directory includes the segmented bus lines from Barcelona used in the study.

- **`Coevolutive_GA_sync`**  
  This directory contains the necessary scripts to run the synchronous version of CCMOCell.

- **`Coevolutive_GA_async`**  
  This directory contains the necessary scripts to run the asynchronous version of CCMOCell.

## **Running the Algorithms**  

To execute the experiment, use the following commands:

1. **Run the synchronous version:**
   ```bash
   python3 ParallelSyncHBCoEvolvedProblem.py
   
1. **Run the asynchronous version:**
   ```bash
   python3 ParallelASyncHBCoEvolvedProblem.py
