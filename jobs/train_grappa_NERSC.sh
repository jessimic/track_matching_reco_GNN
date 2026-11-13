#!/bin/bash

#SBATCH --account=dune
#SBATCH --qos=shared
#SBATCH --constraint=gpu

#SBATCH --job-name=train_uresnet
#SBATCH --output=batch_outputs/output-train_grappa.txt 
#SBATCH --error=batch_outputs/output-train_grappa.txt

#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=16g
#SBATCH --time=1:00:00

srun -n 1 shifter --image deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim bash -c "python3 /global/homes/j/jessiem/track_matching_reco_GNN/gnn.py"
