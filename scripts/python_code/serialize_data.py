import sys
sys.path.append('.')

import json
from argparse import Namespace
from pathlib import Path

from rh_probe.data import KimPTBCorpus

if __name__ == '__main__':

    config_path = 'configs/experiments.json'

    with open(config_path) as open_file:
        config_json = json.load(open_file)
    config = Namespace(**config_json)
    required_data = Namespace(
        rh=config.rh,
        pos=config.pos,
        embeddings=config.embeddings)
    corpus = KimPTBCorpus(required_data, config.corpus_path)

    path = Path(config.serialization_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus.serialize(config.serialization_path)

    print(corpus.train.max_output_length)