import sys
sys.path.append('.')

import json
from argparse import Namespace
from pathlib import Path

from rh_probe.data import KimPTBCorpus
from rh_probe.data.word2vec import gensim2embedding

from gensim.models import KeyedVectors

if __name__ == '__main__':

    config_path = 'configs/rh_pos_bert.json'

    with open(config_path) as open_file:
        config_json = json.load(open_file)
    config = Namespace(**config_json)
    required_data = Namespace(
        rh=config.rh,
        pos=config.pos,
        embeddings=config.embeddings)
    corpus = KimPTBCorpus(required_data, config.corpus_path)
    keyed_vectors = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    gensim2embedding(corpus, keyed_vectors, 'word2vec/googlenews_word2vec.pt')