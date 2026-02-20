import yaml
from spine.driver import Driver

# Load configuration file of the ML chain
cfg_path = '.configs/validation_grappa_writer.cfg'
cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)

driver = Driver(cfg)
