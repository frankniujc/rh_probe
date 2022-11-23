import sys
sys.path.append('.')

import argparse, json
import torch
from pathlib import Path

from rh_probe.experiments import ParsedExperiment

parser = argparse.ArgumentParser(description='Attack an RH probe model.')
parser.add_argument('path', type=str)
args = parser.parse_args()

exp = torch.load(args.path)
og = exp.evaluate(exp.corpus.test)
attack_pos = exp.attack_eval('pos', exp.corpus.test)
attack_rh = exp.attack_eval('rh', exp.corpus.test)
attack_emb = exp.attack_eval('emb', exp.corpus.test)
result = {
    'original': og.to_dict(),
    'attack_rh': attack_rh.to_dict(),
    'attack_emb': attack_emb.to_dict(),
    'attack_pos': attack_pos.to_dict(),
}

attack_name = 'attack' + Path(args.path).stem + '.json'

with open(attack_name, 'w') as open_file:
    json.dump(result, open_file, indent=2)