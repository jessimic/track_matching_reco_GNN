import os,sys
sys.path.insert(0, '/n/holystore01/LABS/iaifi_lab/Users/jmicallef/spine/')

# Necessary imports
import yaml
from spine.driver import Driver

# Load configuration file
#with open('configs/train_grappa_track.cfg', 'r') as f:
with open('configs/validation_grappa_writer.cfg', 'r') as f:
    cfg = yaml.safe_load(f)

# Initialize driver class
driver = Driver(cfg)

# Execute model following the configuration regimen
#driver.run()
