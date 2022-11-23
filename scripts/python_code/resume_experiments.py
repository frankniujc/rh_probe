import sys
sys.path.append('.')

import argparse

import torch
from rh_probe.experiments import FullExperiment

parser = argparse.ArgumentParser(description='Train RH probe model.')
parser.add_argument('path', type=str)
args = parser.parse_args()

exp = torch.load(args.path)