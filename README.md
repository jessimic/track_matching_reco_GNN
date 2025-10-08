# Track Matching inside SPINE using GrapPA GNN

Matching tracks. Project started by Nathanial at https://github.com/ndsantia/mlrecodune/tree/main

## Container
Use singularity container to run SPINE, per their instructions (https://github.com/DeepLearnPhysics/spine). One is stored on the Harvard cluster at `/n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif`. Example of how to pull your own singularity (if on a different cluster) at `jobs/pull_singularity.sh`.

## Running GNN for Training and Inference
--> Make sure your config file is set up inside the config folder and your `gnn.py` is pointing to it

#### Run interactive job with GPU:
$ `salloc --job-name=grappa_gnn  --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=8G   --time=04:00:00   --gres=gpu:1`

$ `singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`

#### Or Call at once with srun:

$ `srun singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`

OR use juptyer notebook (see below)

## Running Jupyter Notebook with Singularity
#### Run interactive job *without* GPU:

$ `salloc --job-name=jupyter_lab  --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=2G   --time=04:00:00 `

Print out:

$ `salloc: Granted job allocation 20007889`

$ `salloc: Waiting for resource configuration`

$ `salloc: Nodes **holy8a26602** are ready for job`

$ `singularity shell /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif`

$  `jupyter lab --no-browser --ip=0.0.0.0 --port=8890`

On local machine: $ `ssh -N -f -L 8890:holy8a26602.login.rc.fas.harvard.edu:8890 jmicallef@login.rc.fas.harvard.edu`

Change to match node name allocated AND port number that you chose

#### OR call at once with srun (can be slower):

$ `srun --job-name=jupyter_lab   --output=jupyter_%j.log   --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=2G   --time=02:00:00   singularity exec /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif jupyter lab --no-browser --ip=0.0.0.0 --port=8890`

(You will see it hang--this is OKAY!)
Print out will be:

$ srun: job 40890758 queued and waiting for resources

$ srun: job 40890758 has been allocated resources

Wherever you submitted it, vim jupyter_*.log  to see which node you are on. You will need to log into another terminal on Harvard cluter to see this, since your other one is hanging. Go to your local machine, make sure the port and the job node match!

On local machine: $ `ssh -N -f -L 8890:NAME_OF_HOST_CLUSTTER_FROM_LOG.edu:8890 jmicallef@login.rc.fas.harvard.edu`

Use the link inside the `jupter_\*.log` and put that into your computer's web browser.

### To run with GPU...
- Add `--gres=gpu:1` to job request
- Add `--nv` before `.sif` path in singularity call

## Code Overview:
- explore_simulation_MINERvA_2x2.ipynb: Plot and explore event displays
- create_minerva_GNN_input.py: Takes in MINERvA DST files and creates LArCV output files, used for the GNN input
- gnn.py: Calls the GNN
- configs: stores the configuration files for training, validating, and writing output file for validation
- validate_CNN_track_matching_commented.ipynb: Takes the log outputs (from training and validating) to make plots vs. iteration (or epoch). Also writes and then reads output from given validation iteration weights, to look at event displays and statistics for the specified iteration.
