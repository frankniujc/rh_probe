import abc
from collections import OrderedDict, defaultdict
import string

from Levenshtein import distance as levenshtein_distance_str

import torch
import numpy as np

def pad_inputs(func):
    def inner_function(self, prediction, target):
        max_length = max(prediction.shape[-1], target.shape[-1])
        prediction = torch.nn.functional.pad(prediction, (0, max_length-prediction.shape[1]), value=self.corpus.pad_index)
        target = torch.nn.functional.pad(target, (0, max_length-target.shape[1]), value=self.corpus.pad_index)
        return func(self, prediction, target)
    return inner_function

class Metric:
    @abc.abstractmethod
    def to_dict(self):
        return NotImplemented

    @abc.abstractmethod
    def record(self, prediction, target):
        return NotImplemented

    @property
    def score(self):
        return self.to_dict()['score']

    @property
    def quick_report(self):
        return f'{self.__class__.__name__}: {self.score}'

    @property
    def report(self):
        dict_obj = self.to_dict()
        report = f"{self.__class__.__name__}: {dict_obj.pop('score')}"

        for key, val in dict_obj.items():
            report += f'\n\t{key}: {val}'

        return report

class TreeIntegrity(Metric):
    # Whether the decoder generated a tree
    def __init__(self, corpus):
        self.total = 0
        self.match = 0
        self.corpus = corpus

    def record(self, prediction, target):
        self.total += prediction.shape[0]
        succeed = self._compute(prediction)
        self.match += succeed.sum().item()

    def _compute(self, prediction):
        with torch.no_grad():
            lb, rb = self.corpus.brackets

            bracket_counter = torch.zeros(prediction.shape[0], dtype=int).to(prediction.device)
            succeed = torch.ones(prediction.shape[0], dtype=bool).to(prediction.device)
            prev_is_lb = torch.zeros(prediction.shape[0], dtype=bool).to(prediction.device)
            has_end = torch.zeros(prediction.shape[0], dtype=bool).to(prediction.device)
            reached_zero = torch.zeros(prediction.shape[0], dtype=int).to(prediction.device)

            for t in range(prediction.shape[1]):
                tokens = prediction[:, t]
                bracket_counter += (tokens == lb)
                bracket_counter -= (tokens == rb) * 1

                has_end |= tokens == self.corpus.eos_index

                no_violations = bracket_counter >= 0
                reached_zero += (bracket_counter == 0) & ~has_end

                is_tag = (tokens != lb) & (tokens != rb) & (tokens != self.corpus.eos_index) & (tokens != self.corpus.pad_index)
                tag_after_lb = ~torch.logical_xor(prev_is_lb, is_tag)
                no_violations &= tag_after_lb
                succeed &= no_violations

                prev_is_lb = tokens == lb

        succeed &= (reached_zero == 1)

        return succeed

    def to_dict(self):
        return {
            'score': self.match / self.total,
            'match': self.match,
            'total': self.total,
        }

class ExactMatchAccuracy(Metric):
    # Exact Match
    def __init__(self, corpus):
        self.total = 0
        self.match = 0
        self.corpus = corpus

    @pad_inputs
    def record(self, prediction, target):
        total = ((target != self.corpus.pad_index) & (target != self.corpus.eos_index) & (target != self.corpus.bos_index)) | ((prediction != self.corpus.pad_index) & (prediction != self.corpus.eos_index) & (prediction != self.corpus.bos_index))
        self.total += total.sum().item()
        match = (prediction == target) & total
        self.match += match.sum().item()

    def to_dict(self):
        return {
            'score': self.match / self.total,
            'match': self.match,
            'total': self.total,
        }

class SentenceExactMatchAccuracy(ExactMatchAccuracy):
    @pad_inputs
    def record(self, prediction, target):
        self.total += prediction.shape[0]
        ignored = target == self.corpus.pad_index
        match = (prediction == target) | ignored
        self.match += torch.all(match, dim=1).sum().item()

def seq_to_string(seq):
    all_char = string.printable
    s = ''.join(all_char[x.item()] for x in seq)
    return s

def _levenshtein_distance(seq1, seq2):
    str1, str2 = seq_to_string(seq1), seq_to_string(seq2)
    return levenshtein_distance_str(str1, str2)

def _levenshtein_distance_python(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = torch.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1]).item()

class LevenshteinDistance(Metric):
    def __init__(self, corpus):
        self.distances = []
        self.normalized_distances = [] # normalized by target length
        self.corpus = corpus

    def remove_padding(self, seq):
        not_pad = (seq != self.corpus.eos_index) & (seq != self.corpus.pad_index)
        length = not_pad.sum()
        return seq[:length]

    def record(self, predictions, targets):
        distances, norm = [], []
        for prediction, target in zip(predictions, targets):
            prediction = self.remove_padding(prediction)
            target = self.remove_padding(target)

            ld = _levenshtein_distance(prediction, target)
            self.distances.append(ld)
            self.normalized_distances.append(ld/len(target))

    def to_dict(self):
        return {
            'score': sum(self.normalized_distances) / len(self.distances),
            'not_normalized': sum(self.distances) / len(self.distances),
        }

class MetricCollection(Metric):
    def __init__(self):
        self.metrics = OrderedDict()

    def register(self, name, metric):
        self.metrics[name] = metric

    def record(self, prediction, target):
        for metric in self.metrics.values():
            metric.record(prediction, target)

    def to_dict(self):
        dict_obj = {}
        for metric_name, val in self.metrics.items():
            dict_obj[metric_name] = val.to_dict()
        return dict_obj

    @property
    def quick_report(self):
        return '\n'.join(m.quick_report for m in self.metrics.values())

    @property
    def report(self):
        return '\n\n'.join(m.report for m in self.metrics.values())

    @property
    def score(self):
        return tuple(m.score for m in self.metrics.values())

metric_classes = {
    'tree_integrity': TreeIntegrity,
    'em_acc': ExactMatchAccuracy,
    'sent_em_acc': SentenceExactMatchAccuracy,
    'levenshtein': LevenshteinDistance,
}