#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH -p shared
#SBATCH --mem=4000
#SBATCH -t 00-06:00
#SBATCH -o singularity_%x_%j.out # Standard out goes to this file
#SBATCH -e singularity_%x_%j.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2485353974@vtext.com


export TMPDIR=/n/holyscratch01/iaifi_lab/Users/jmicallef/

#singularity pull /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine.sif docker://deeplearnphysics/larcv2:ub22.04-cuda12.1-pytorch2.2.1-larndsim
singularity pull /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif docker://deeplearnphysics/larcv2:ub20.04-cuda11.6-pytorch1.13-larndsim

# Check for errors
if [ $? -ne 0 ]; then
    echo "Singularity pull failed" >&2
else
    echo "Singularity pull succeeded"
fi
