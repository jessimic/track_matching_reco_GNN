import os,sys
#sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')
sys.path.insert(0, '/sdf/home/j/jessicam/spine/')
# Necessary imports
import yaml
from spine.driver import Driver

# Load configuration file
with open('configs/train_grappa_track.cfg', 'r') as f:
#with open('configs/validation_grappa_writer.cfg', 'r') as f:
    cfg = yaml.safe_load(f)

# Initialize driver class
driver = Driver(cfg)

# Execute model following the configuration regimen
driver.run()
