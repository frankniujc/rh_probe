import pickle
from pathlib import Path
from itertools import chain
from tqdm import tqdm

from .tree import parse_and_binarise, RHTree

import torch

class Dictionary:
    def __init__(self, tags):
        # `tags` must not contain any duplicates
        self.tag2id = {tag:i for i,tag in enumerate(tags)}
        self.id2tag = {i:tag for i,tag in enumerate(tags)}

    def to_id(self, lst):
        return [self.tag2id[x] for x in lst]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.id2tag[key]
        elif isinstance(key, str):
            return self.tag2id

    def __len__(self):
        return len(self.tag2id)

class KimPTBCorpus:

    ptb_tree_tags = Dictionary(['[PAD]', '[BOS]', '[EOS]', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] + ['(', ')', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'PRT|ADVP', 'QP', 'RRC', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X'])

    def __init__(self, required_data, path, serialization_path=None):
        self.path = Path(path)
        self.required_data = required_data
        assert self.path.exists() and self.path.is_dir() and self.path.name.startswith('cpcfg_process')
        if serialization_path is None or not Path(serialization_path).exists():
            self.split()
        else:
            serialization_path = Path(serialization_path)
            print(f'Serialization specified and detected at `{serialization_path}`. Loading serialized corpus.')
            self.load_serialization(serialization_path)

    def vectorize(self, config, tokenizer=None):
        self.train.vectorize(config, tokenizer)
        self.valid.vectorize(config, tokenizer)
        self.test.vectorize(config, tokenizer)

    def vocab(self):
        return set(chain(*self.train.data['sent']))

    def split(self):
        self.train = KimPTBSplitDataset(self.required_data, self.path / 'ptb-train.txt', self)
        self.valid = KimPTBSplitDataset(self.required_data, self.path / 'ptb-valid.txt', self)
        self.test = KimPTBSplitDataset(self.required_data, self.path / 'ptb-test.txt', self)

    def serialize(self, path):
        path = Path(path)
        data = {
            'train': self.train.data,
            'valid': self.valid.data,
            'test': self.test.data,
        }
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def load_serialization(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        self.train = SerializedKimPTBSplitDataset(self.required_data, self.path / 'ptb-train.txt', self, data['train'])
        self.valid = SerializedKimPTBSplitDataset(self.required_data, self.path / 'ptb-valid.txt', self, data['valid'])
        self.test = SerializedKimPTBSplitDataset(self.required_data, self.path / 'ptb-test.txt', self, data['test'])

    @property
    def output_size(self):
        return len(self.ptb_tree_tags)

    def output_tags(self):
        tags = set(chain(*(self.train.data['tree_sequence']+self.valid.data['tree_sequence']+self.test.data['tree_sequence'])))
        return tags

    def get_bos_tensor(self, length):
        bos = self.ptb_tree_tags.tag2id['[BOS]']
        return torch.full((length,), bos)

    @property
    def bos_index(self):
        return self.ptb_tree_tags.tag2id['[EOS]']

    @property
    def eos_index(self):
        return self.ptb_tree_tags.tag2id['[EOS]']

    @property
    def pad_index(self):
        return self.ptb_tree_tags.tag2id['[PAD]']

    @property
    def brackets(self):
        return self.ptb_tree_tags.tag2id['('], self.ptb_tree_tags.tag2id[')']


class KimPTBSplitDataset(torch.utils.data.Dataset):

    line_count = {
        'ptb-train.txt': 39832,
        'ptb-valid.txt': 1700,
        'ptb-test.txt': 2416,
    }

    def __init__(self, required_data, txt_file_path, corpus):
        self.required_data = required_data
        self.txt_file_path = txt_file_path
        self.data = {
            'sent': [],
            'tree_sequence': [],
            'pos': [],
            'rh': [],
            'h_r': [],
        }
        self.corpus = corpus
        self.read_data(txt_file_path)

    def read_data(self, txt_file_path):
        with open(txt_file_path) as open_file:
            for line in tqdm(open_file,
                desc=f'Reading {self.txt_file_path.name}',
                total=self.line_count[self.txt_file_path.name]):
                if line.strip():
                    tree = parse_and_binarise(line)
                    self.data['tree_sequence'].append(['[BOS]'] + tree.tree_sequence() + ['[EOS]'])
                    self.data['sent'].append([x for x, _ in tree.pos()])

                    if self.required_data.pos:
                        self.data['pos'].append([x for _, x in tree.pos()])

                    if self.required_data.rh:
                        self.data['rh'].append(tree.rh_distances())
                        self.data['h_r'].append(tree.height())
        self.max_input_length = max(len(x) for x in self.data['sent'])
        self.max_output_length = max(len(x) for x in self.data['tree_sequence'])

    def build_dict(self):
        token_set = set(chain(*self.data['sent']))
        token2id = {token:i+1 for i, token in enumerate(token_set)}
        token2id['[UNK]'] = 0
        id2token = {i:token for token, i in token2id.items()}
        return token2id, id2token

    def vectorize(self, config, tokenizer):
        self.vectorize_output()
        if self.required_data.pos:
            self.vectorize_pos()
        if 'bert' in self.required_data.embeddings:
            self.bert_tokenize(tokenizer)
        else:
            self.word2vec_model = tokenizer

    def vectorize_pos(self):
        self.data['pos_id'] = [self.corpus.ptb_tree_tags.to_id(pos) for pos in self.data['pos']]

    def vectorize_output(self):
        self.data['tree_sequence_id'] = [self.corpus.ptb_tree_tags.to_id(pos) for pos in self.data['tree_sequence']]

    def bert_tokenize(self, bert_tokenizer):
        bert_encodings = bert_tokenizer(self.data['sent'],
            padding=True,
            return_length=True,
            return_offsets_mapping=True,
            is_split_into_words=True)
        self.bert_encodings = bert_encodings

        # Adjust for BERT offset
        if self.required_data.pos:
            self.data['adjusted_pos_id'] = self.adjust_bert_offset(self.data['pos_id'], 0)
        if self.required_data.rh:
            self.data['adjusted_rh'] = self.adjust_bert_offset(self.data['rh'], 0)

    def adjust_bert_offset(self, tags, pad_value):
        new_data = []
        for tag_lst, offsets in zip(tags, self.bert_encodings.offset_mapping):
            i = -1
            adjusted_tag_lst = []
            for s,e in offsets:
                if e == 0:
                    adjusted_tag_lst.append(pad_value)
                else:
                    if s == 0:
                        i+=1
                    adjusted_tag_lst.append(tag_lst[i])
            new_data.append(adjusted_tag_lst)
        return new_data

    def __len__(self):
        return len(self.data['sent'])

    def __getitem__(self, idx):
        # Input
        use_bert = 'bert' in self.required_data.embeddings
        use_word2vec = 'word2vec' in self.required_data.embeddings
        if use_bert:
            item = {key: torch.tensor(val[idx]) for key, val in self.bert_encodings.items()}
        if use_word2vec:
            item = {
                'word2vec': self.word2vec_model.vectorize(self.data['sent'][idx], pad_to_length=self.max_input_length),
                'length': len(self.data['sent'][idx]),
            }
        if not use_bert and not use_word2vec:
            item = {
                'length': len(self.data['sent'][idx]),
            }  

        if self.required_data.pos:
            tensor = torch.tensor(self.data['adjusted_'*use_bert + 'pos_id'][idx])
            if not use_bert:
                pad_length = self.max_input_length - len(tensor)
                tensor = torch.nn.functional.pad(tensor, (0, pad_length))
            item['pos'] = tensor

        if self.required_data.rh:
            tensor = torch.tensor(self.data['adjusted_'*use_bert + 'rh'][idx])
            if not use_bert:
                pad_length = self.max_input_length - len(tensor)
                tensor = torch.nn.functional.pad(tensor, (0, pad_length))
            item['rh'] = tensor

            item['h_r'] = torch.tensor(self.data['h_r'][idx])

        # Output
        tensor = torch.tensor(self.data['tree_sequence_id'][idx])
        item['tree_sequence_length'] = torch.tensor(len(tensor))
        pad_length = self.max_output_length - len(tensor)
        tensor = torch.nn.functional.pad(tensor, (0, pad_length))
        item['tree_sequence'] = tensor

        return item

class SerializedKimPTBSplitDataset(KimPTBSplitDataset):
    def __init__(self, required_data, txt_file_path, corpus, data):
        self.required_data = required_data
        self.txt_file_path = txt_file_path
        self.data = data
        self.corpus = corpus
        self.max_input_length = max(len(x) for x in self.data['sent'])
        self.max_output_length = max(len(x) for x in self.data['tree_sequence'])