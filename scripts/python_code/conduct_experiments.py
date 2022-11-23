import sys
sys.path.append('.')

import argparse

from rh_probe.experiments import FullExperiment

parser = argparse.ArgumentParser(description='Train RH probe model.')
parser.add_argument('-c', '--config-path', type=str, default='configs/full_experiments/rh_probe.json')
args = parser.parse_args()

exp = FullExperiment(args.config_path)
exp.conduct_experiment()