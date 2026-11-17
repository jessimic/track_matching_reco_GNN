#!/bin/bash

#SBATCH --account=dune
#SBATCH --qos=shared
#SBATCH --constraint=gpu

#SBATCH --job-name=validate_grappa
#SBATCH --output=batch_outputs/output-validate_grappa_balance_200.txt 
#SBATCH --error=batch_outputs/output-validate_grappa_balance_200.txt

#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=4g
#SBATCH --time=2:00:00

srun -n 1 shifter --image deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim bash -c "python3 /global/homes/j/jessiem/track_matching_reco_GNN/gnn_val.py /global/homes/j/jessiem/track_matching_reco_GNN/configs/validation_grappa_NDLAr_balance_200.cfg"
