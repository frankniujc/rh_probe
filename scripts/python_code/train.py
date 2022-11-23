import sys
sys.path.append('.')

import argparse, json

from rh_probe.experiments import Experiment, ParsedExperiment

parser = argparse.ArgumentParser(description='Train RH probe model.')
parser.add_argument('configs', type=str, nargs='+')
args = parser.parse_args()

for config_path in args.configs:
    with open(config_path) as open_file:
        config = json.load(open_file)
    exp = ParsedExperiment(config)
    exp.train()
