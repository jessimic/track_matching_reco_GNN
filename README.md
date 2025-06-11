# Track Matching inside SPINE using GrapPA GNN

Matching tracks. Project started by Nathanial at https://github.com/ndsantia/mlrecodune/tree/main

## Container
Use singularity container to run SPINE, per their instructions (https://github.com/DeepLearnPhysics/spine). One is stored on the Harvard cluster at `/n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif`. Example of how to pull your own singularity (if on a different cluster) at `jobs/pull_singularity.sh`.

## Running GNN
--> Make sure your config file is set up inside the config folder and your `gnn.py` is pointing to it
Run interactive job with GPU
$ `salloc --job-name=grappa_gnn  --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=8G   --time=04:00:00   --gres=gpu:1`
$ `singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`
OR
$ `srun singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`
srun calls a GPU, could alternatively call an interactive job first
OR use juptyer notebook (see below)

## Running Jupyter Notebook with Singularity
$ `srun --job-name=jupyter_lab   --output=jupyter_%j.log   --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=8G   --time=04:00:00   --gres=gpu:1   singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif jupyter lab --no-browser --ip=0.0.0.0 --port=8890`
(You will see it hang--this is OKAY!)
$ srun: job 40890758 queued and waiting for resources
$ srun: job 40890758 has been allocated resources
Wherever you submitted it, vim jupyter_*.log  to see which node you are on. You will need to log into another terminal on Harvard cluter to see this, since your other one is hanging. Go to your local machine, make sure the port and the job node match!
On local machine:
$ `ssh -N -f -L 8890:NAME_OF_HOST_CLUSTTER_FROM_LOG.edu:8890 jmicallef@login.rc.fas.harvard.edu`
Use the link inside the `jupter_\*.log` and put that into your computer's web browser.


## Code Overview:
- explore_simulation_MINERvA_2x2.ipynb: Plot and explore event displays
- create_minerva_GNN_input.py: Takes in MINERvA DST files and creates LArCV output files, used for the GNN input
- gnn.py: Calls the GNN
- configs: stores the configuration files for training, validating, and writing output file for validation
- validate_CNN_track_matching_commented.ipynb: Takes the log outputs (from training and validating) to make plots vs. iteration (or epoch). Also writes and then reads output from given validation iteration weights, to look at event displays and statistics for the specified iteration.
