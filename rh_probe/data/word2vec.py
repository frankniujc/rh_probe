from itertools import chain
import torch
from tqdm import tqdm

class Word2VecEmbedding:
    def __init__(self):
        self.token2tensor = {}

    def load(self, path):
        self.token2tensor = torch.load(path)
        self.unk_tensor = self.compute_avg()

    def compute_avg(self):
        # https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        unk_tensor = torch.sum(torch.stack([x for x in self.token2tensor.values()]), dim=0)
        unk_tensor /= len(self.token2tensor)
        return unk_tensor

    def __getitem__(self, key):
        return self.token2tensor.get(key, self.unk_tensor)

    def vectorize(self, sent, pad_to_length=None):
        tensors = [self[x] for x in sent]
        sent_tensor = torch.stack(tensors)
        pad_length = pad_to_length - len(sent)
        padded = torch.nn.functional.pad(sent_tensor, (0, 0, 0, pad_length)).float()
        return padded

def gensim2embedding(corpus, keyed_vectors, output_path):
    ptb_vocab = corpus.vocab()
    emb = {}

    for token in ptb_vocab:
        if token in keyed_vectors:
            emb[token] = torch.tensor(keyed_vectors[token])

    torch.save(emb, output_path)