import os,sys
#sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')
#sys.path.insert(0, '/sdf/home/j/jessicam/spine/')
# Necessary imports
import yaml
from spine.driver import Driver
import argparse

parser = argparse.ArgumentParser(
        description="Run GNN using given config file")
parser.add_argument("input", default=None, help="Full path to config file")
args = parser.parse_args()

# Load configuration file
#with open('/global/u1/j/jessiem/track_matching_reco_GNN/configs/validation_grappa_writer.cfg', 'r') as f:
#with open('/global/u1/j/jessiem/track_matching_reco_GNN/configs/validation_grappa_write_infer.cfg', 'r') as f: #BROEKN

if args.input is None:
    config_file = '/global/homes/j/jessiem/track_matching_reco_GNN/configs/train_grappa_NDLAr_bs2.cfg'
else:
    config_file = args.input

with open(config_file, 'r') as f:
#with open('configs/validation_grappa_writer_ndlar.cfg', 'r') as f:

   cfg = yaml.safe_load(f)

# Initialize driver class
driver = Driver(cfg)

# Execute model following the configuration regimen
driver.run()
