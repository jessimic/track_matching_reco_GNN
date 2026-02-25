import os,sys
#sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')
#sys.path.insert(0, '/sdf/home/j/jessicam/spine/')
# Necessary imports
import yaml
from spine.driver import Driver
from spine.config import load_config_file

# Load configuration file
cfg_path = 'configs/spine_full_inference/full_chain_240819_modified.yaml'
cfg = load_config_file(cfg_path)

# Initialize driver class
driver = Driver(cfg)

# Execute model following the configuration regimen
driver.run()
