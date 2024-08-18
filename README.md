# VaDeSCEHR
This repository contains code for a method for clustering longitudinal time-to-event data as extracted from electronic health records, published as a preprint on https://www.medrxiv.org/content/10.1101/2024.01.11.24301148v2

Deep representation learning for clustering longitudinal survival data from electronic health records

# Dependencies install
All the dependencies are in packages.txt based on python3.9

python3.9 -m pip install -r packages.txt

# Data download and simulation data
The UKB EHR data must be downloaded from UK Biobank (https://www.ukbiobank.ac.uk/) by user's self

The 5 fold simulation data used in the paper can be found at https://github.com/JiajunQiu/TransVarSur/tree/main/data/simulation

# Example test
The following code can be used to re-produce the result on simulation for a single fold:

python3.9 main.py -s simulation -n ./data/simulation/benchmark_data_fold1.pkl -p outputs/output_simulation_fold1 -r False -b True -k True

And you can find the example of the outputs in https://github.com/JiajunQiu/VaDeSCEHR/tree/main/outputs/output_simulation_fold1
