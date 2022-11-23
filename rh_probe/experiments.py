import sys, json
import itertools
from copy import deepcopy
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict

from .seq2seq import EncoderDecoder
from .data import KimPTBCorpus
from .data.word2vec import Word2VecEmbedding
from .metrics import MetricCollection, metric_classes

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

class Record:
    def __init__(self, info):
        self.epochs = []
        self.info = info
        self.best = None

    @property
    def is_the_best_epoch(self):
        return self.best['epoch'] == len(self.epochs)-1

    def output(self, path):
        record = {
            'best': self.get_best_json(),
            'info': self.info
        }

        for i, epoch in enumerate(self.epochs, start=1):
            record[i] = {
                'valid': epoch['valid'].to_dict(),
                'test': epoch['test'].to_dict(),
            }

        with open(path, 'w') as open_file:
            json.dump(record, open_file, indent=4)

    def get_best_json(self):
        best = {
            'epoch': self.best['epoch'],
            'valid': self.best['valid'].to_dict(),
            'test': self.best['test'].to_dict(),
        }
        return best

    def record(self, valid, test):
        epoch = {
            'valid': valid,
            'test': test,
        }

        if self.best is None:
            self.best = epoch
            self.best['epoch'] = len(self.epochs)

        if valid.score > self.best['valid'].score:
            self.best = epoch
            self.best['epoch'] = len(self.epochs)

        self.epochs.append(epoch)

class Experiment:
    def __init__(self, config_path):
        self.read_config(config_path)
        self.use_bert = 'bert' in self.config.embeddings
        self.use_word2vec = 'word2vec' in self.config.embeddings
        self.device = torch.device(self.config.device)
        print(json.dumps(self.config_json, indent=2))

        self.setup_record()
        if self.use_bert:
            self.setup_bert()
        if self.use_word2vec:
            self.setup_word2vec()
        self.setup_data()
        self.setup_model()

    def resume(self, to_epoch, checkpoint_saving_path):
        print('RESUMING FROM')
        epoch = self.load(checkpoint_saving_path)
        self.model.eval()
        self.evaluate(self.corpus.valid)
        self.evaluate(self.corpus.test)
        import pdb; pdb.set_trace()

    def load(self, path=None):
        if path is None:
            path = Path(self.config.report_output_dir) / (self.exp_name + '.pt')
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        if 'ptb_tag_embedding.weight' in model_state_dict:
            model_state_dict.pop('ptb_tag_embedding.weight')

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def read_config(self, config_path):
        self.config_path = config_path
        with open(config_path) as open_file:
            self.config_json = json.load(open_file)
        self.config = Namespace(**self.config_json)

    def setup_record(self):
        self.record = Record(self.config_json)

    def setup_bert(self):
        from transformers import BertModel, BertConfig, BertTokenizerFast

        local_bert_config_path = Path('configs/huggingface_bert/') / (self.config.embeddings + '.json')

        self.bert_model = BertModel(BertConfig.from_json_file(local_bert_config_path)).to(self.device)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.bert_tokenizer = BertTokenizerFast.from_pretrained(self.config.embeddings)

    def setup_word2vec(self):
        self.word2vec_model = Word2VecEmbedding()
        self.word2vec_model.load(self.config.embeddings)

    def setup_data(self):
        required_data = Namespace(
            rh=self.config.rh,
            pos=self.config.pos,
            embeddings=self.config.embeddings)
        self.corpus = KimPTBCorpus(required_data, self.config.corpus_path, serialization_path=self.config.serialization_path)
        if self.use_bert:
            self.corpus.vectorize(self.config, tokenizer=self.bert_tokenizer)
        if self.use_word2vec:
            self.corpus.vectorize(self.config, tokenizer=self.word2vec_model)
        if not self.use_bert and not self.use_word2vec:
            self.corpus.vectorize(self.config)

    def setup_model(self):
        self.epoch = 0

        if self.use_bert:
            bert_model = self.bert_model
        else:
            bert_model = None

        self.model = EncoderDecoder(
            corpus=self.corpus,
            **self.config_json).to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        optimizers = {
            'adam': torch.optim.Adam,
        }
        self.optimizer = optimizers[self.config.optimizer](self.model.parameters(), lr=self.config.learning_rate)

    def prepare_input(self, batch, teacher_enforcing=False):
        input_data = OrderedDict()
        if self.use_bert:
            bert_output = self.bert_model(batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                token_type_ids=batch['token_type_ids'].to(self.device))
            input_data['embeddings'] = bert_output[0].detach()
            input_data['lengths'] = batch['attention_mask'].sum(dim=1)
        if self.use_word2vec:
            input_data['embeddings'] = batch['word2vec'].to(self.device)
            input_data['lengths'] = batch['length']
        if not self.use_bert and not self.use_word2vec:
            input_data['lengths'] = batch['length']

        if self.config.pos:
            input_data['pos'] = batch['pos'].to(self.device)

        if self.config.rh:
            input_data['rh'] = batch['rh'].to(self.device)

        gold_output = {
            'tree_sequence': batch['tree_sequence'].to(self.device),
            'tree_sequence_length': batch['tree_sequence_length'],
        }

        return input_data, gold_output

    def train(self):
        dataloader = DataLoader(self.corpus.train, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(self.config.epochs):
            self.train_epoch(dataloader)
            valid_metrics = self.evaluate(self.corpus.valid)
            test_metrics = self.evaluate(self.corpus.test)
            self.record.record(valid_metrics, test_metrics)
            if self.record.is_the_best_epoch:
                print('best performing epoch found, saving...')
                self.save_checkpoint()

        self.record.output(Path(self.config.report_output_dir) / self.report_name)
        self.save_checkpoint('_final')

    def resume(self):
        dataloader = DataLoader(self.corpus.train, batch_size=self.config.batch_size, shuffle=True)
        for epoch in range(self.epoch, self.config.epochs):
            self.train_epoch(dataloader)
            valid_metrics = self.evaluate(self.corpus.valid)
            test_metrics = self.evaluate(self.corpus.test)
            self.record.record(valid_metrics, test_metrics)
            if self.record.is_the_best_epoch:
                print('best performing epoch found, saving...')
                self.save_checkpoint()

        self.record.output(Path(self.config.report_output_dir) / self.report_name)
        self.save_checkpoint('_final')

    def save_checkpoint(self, extra=''):
        checkpoint_saving_path = Path(self.config.report_output_dir) / (self.exp_name + extra + '_full_model.pt')
        checkpoint_saving_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        torch.save(self, checkpoint_saving_path)

        print(f'checkpoint saved at {checkpoint_saving_path}')

    @property
    def report_name(self):
        return self.exp_name + '.json'

    @property
    def exp_name(self):
        name = []
        if self.config.rh:
            name.append('rh')
        if self.config.pos:
            name.append('pos')
        if self.use_bert:
            name.append(self.config.embeddings.split('/')[-1])            
        if self.use_word2vec:
            if 'googlenews' in self.config.embeddings:
                name.append('word2vec')
            else:
                name.append('word2vec')
        return '_'.join(name)

    def train_epoch(self, dataloader):
        self.epoch += 1
        self.model.train()
        total_size = 0.
        total_loss = 0.

        for i, batch in enumerate(tqdm(dataloader,
            desc=f'training [{self.epoch}]',
            file=sys.stderr)):

            input_data, gold_output = self.prepare_input(batch)
            teacher_enforced_output = self.model(input_data, gold_output)
            target = gold_output['tree_sequence'][:,1:gold_output['tree_sequence_length'].max()]

            loss = self.loss_function(teacher_enforced_output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_size += batch['tree_sequence'].shape[0]
            total_loss += loss.item()
        return total_loss / total_size

    def create_metrics(self):
        metrics = MetricCollection()

        for name, metric_class in metric_classes.items():
            metrics.register(name, metric_class(self.corpus))

        return metrics

    def evaluate_greedy(self, data_split):
        print('performing GREEDY evaluation')
        self.model.eval()
        metrics = self.create_metrics()
        dataloader = DataLoader(data_split, batch_size=self.config.eval_batch_size, shuffle=True)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader,
                desc=f'evaluating [{self.epoch}]',
                file=sys.stderr)):
                input_data, gold_output = self.prepare_input(batch)
                _, model_predictions = self.model(input_data, greedy=True)

                target = gold_output['tree_sequence'][:,1:gold_output['tree_sequence_length'].max()]

                metrics.record(model_predictions, target)

        print(metrics.quick_report)
        return metrics

    def evaluate(self, data_split):
        self.model.eval()
        metrics = self.create_metrics()
        dataloader = DataLoader(data_split, batch_size=self.config.eval_batch_size, shuffle=True)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader,
                desc=f'evaluating [{self.epoch}]',
                file=sys.stderr)):
                input_data, gold_output = self.prepare_input(batch)
                top_probs, top_tokens = self.model(input_data)

                model_predictions = top_tokens[:,0,:]
                target = gold_output['tree_sequence'][:,1:gold_output['tree_sequence_length'].max()]

                metrics.record(model_predictions, target)

        print(metrics.quick_report)
        return metrics

class ParsedExperiment(Experiment):
    def read_config(self, config):
        self.config_path = None
        self.config_json = config
        self.config = Namespace(**self.config_json)

    def attack_eval(self, attack_input, data_split):
        self.model.eval()
        metrics = self.create_metrics()
        dataloader = DataLoader(data_split, batch_size=self.config.eval_batch_size, shuffle=True)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader,
                desc=f'evaluating [{self.epoch}]',
                file=sys.stderr)):
                input_data, gold_output = self.prepare_attacked_input(batch, attack_input)
                top_probs, top_tokens = self.model(input_data)

                model_predictions = top_tokens[:,0,:]
                target = gold_output['tree_sequence'][:,1:gold_output['tree_sequence_length'].max()]

                metrics.record(model_predictions, target)

        print(f'Attacked {attack_input}!', metrics.quick_report)
        return metrics

    def prepare_attacked_input(self, batch, attack_input):
        input_data = OrderedDict()
        attack_scale = 0.1

        if self.use_bert:
            bert_output = self.bert_model(batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                token_type_ids=batch['token_type_ids'].to(self.device))
            if attack_input == 'emb':
                input_data['embeddings'] = torch.rand(bert_output[0].shape).to(self.device)
            else:
                input_data['embeddings'] = bert_output[0].detach()
            input_data['lengths'] = batch['attention_mask'].sum(dim=1)
        if self.use_word2vec:
            if attack_input == 'emb':
                input_data['embeddings'] = torch.rand(batch['word2vec'].shape).to(self.device)
            else:
                input_data['embeddings'] = batch['word2vec'].to(self.device)
            input_data['lengths'] = batch['length']
        if self.config.pos:
            if attack_input == 'pos':
                # input_data['pos'] = torch.rand(batch['pos'].shape).to(self.device)
                input_data['pos'] = 'attack!', batch['pos'].to(self.device)
            else:
                input_data['pos'] = batch['pos'].to(self.device)

        if self.config.rh:
            if attack_input == 'rh':
                input_data['rh'] = torch.rand(batch['rh'].shape).to(self.device)
            else:
                input_data['rh'] = batch['rh'].to(self.device)

        gold_output = {
            'tree_sequence': batch['tree_sequence'].to(self.device),
            'tree_sequence_length': batch['tree_sequence_length'],
        }

        return input_data, gold_output

class FullExperiment:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path) as open_file:
            self.config_json = json.load(open_file)
        self.config = Namespace(**self.config_json)

    def conduct_experiment(self):
        variables = OrderedDict(self.config.variables)
        for variable in itertools.product(*variables.values()):
            variable_config = {k:v for k, v in zip(variables, variable)}
            config = deepcopy(self.config_json)
            config.update(variable_config)
            config['variable'] = variable_config

            exp = ParsedExperiment(config)
            exp.train()
            del exp

    def resume_experiment(self, to_epoch, load_dir):
        variables = OrderedDict(self.config.variables)
        for variable in itertools.product(*variables.values()):
            variable_config = {k:v for k, v in zip(variables, variable)}
            config = deepcopy(self.config_json)
            config.update(variable_config)
            config['variable'] = variable_config

            exp = ParsedExperiment(config)
            checkpoint_saving_path = Path(load_dir) / (exp.exp_name + '.pt')
            exp.resume(to_epoch, checkpoint_saving_path)
            del exp