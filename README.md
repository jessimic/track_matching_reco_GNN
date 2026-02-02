# Track Matching inside SPINE using GrapPA GNN

Matching tracks. Project started by Nathanial at https://github.com/ndsantia/mlrecodune/tree/main

## Code Overview:
- explore_simulation_MINERvA_2x2.ipynb: Plot and explore event displays including 2x2
- explore_simulation_MINERvA_only.ipynb: Plot and explore MINERvA displays only
- create_LArCV_minerva.py: rewritten create_minerva_GNN_input.py, paired down to essentials (by Jessie).
- create_minerva_GNN_input.py: OLDER. Takes in MINERvA DST files and creates LArCV output files, used for the GNN input
- gnn.py: Calls the GNN (written by Nate).
- loop_validation.sh: example how to call the validation to run over all the saved training weight iterations.
- configs: stores the configuration files for training, validating, and writing output file for validation
- GNN_training_testing_curves.ipynb: Takes the log outputs (from training and validating) to make plots vs. iteration (or epoch). Also writes output file for all the entries once you choose the best validation iteration.
- Plotting_validation_Mx2.ipynh: Takes the hdf5 file written and looks at the overall confusion matrix along with single event plots showing "how correct" the network is per event.

## Container
Use singularity or apptainer container to run SPINE and all above scripts, per their instructions (https://github.com/DeepLearnPhysics/spine). One is stored on the Harvard cluster at `/n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif`. For NERSC, you'll use shifter (see below). Example of how to pull your own singularity (if on a different cluster) at `jobs/pull_singularity.sh`.

### Run on NERSC, request a GPU node, run shifter to get the container
$ `ssh $USER@permutter.ners.gov`

$ `salloc --nodes=1   --qos shared_interactive --ntasks=1  --time=01:00:00 --constraint gpu --gpus 1`

$ `shifter --image=deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim`

Instead of singularity or apptainer, NERSC uses "Shifter". More info at: https://docs.nersc.gov/services/jupyter/how-to-guides/#shifter. For NERSC, you can see more information about QOS limits and charges at https://docs.nersc.gov/jobs/policy/#qos-limits-and-charges. 

## Using Jupyter on NERSC
Available at https://jupyter.nersc.gov/.


## Running GNN for Training and Inference
--> Make sure your config file is set up inside the config folder and your `gnn.py` is pointing to it

## Harvard Cluster Directions

#### Run interactive job with GPU:
$ `salloc --job-name=grappa_gnn  --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=8G   --time=04:00:00   --gres=gpu:1`

$ `singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`

#### Or Call at once with srun:

$ `srun singularity exec --nv /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif gnn.py`

OR use juptyer notebook (see below)

#### Run interactive Jupyter notebook job *without* GPU:

$ `salloc --job-name=jupyter_lab  --nodes=1   --ntasks=1   --cpus-per-task=1   --mem=2G   --time=04:00:00 `

Print out:

$ `salloc: Granted job allocation 20007889`

$ `salloc: Waiting for resource configuration`

$ `salloc: Nodes **holy8a26602** are ready for job`

$ `singularity shell /n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine_larcv_ME_cuda_pytorch.sif`

$  `jupyter lab --no-browser --ip=0.0.0.0 --port=8890`

On local machine: $ `ssh -N -f -L 8890:holy8a26602.login.rc.fas.harvard.edu:8890 jmicallef@login.rc.fas.harvard.edu`

Change to match node name allocated AND port number that you chose

#### OR call Jupyter notebook at once with srun (can be slower):

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

