import warnings
from argparse import Namespace

import torch

rnn_classes = {'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU, 'rnn': torch.nn.RNN}
rnn_cell_classes = {'lstm': torch.nn.LSTMCell, 'gru': torch.nn.GRUCell, 'rnn': torch.nn.RNNCell}

class ModelBase(torch.nn.Module):

    required_config = []
    optional_config = []

    def __init__(self, config):
        super().__init__()

        assert all(x in config for x in self.required_config), f'The following configurations are missing for {self.__class__}: {self._missing_configs(config)}'

        self.config = Namespace(**{k:config[k] for k in self.required_config+self.optional_config})

    def _missing_configs(self, config):
        return [k for k in self.required_config if k not in config]

class Encoder(ModelBase):

    required_config = ['input_size', 'hidden_size', 'num_layers', 'bidirectional', 'dropout', 'rnn_class']

    def __init__(self, **encoder_config):
        super().__init__(encoder_config)

        self.rnn = rnn_classes[self.config.rnn_class](
            self.config.input_size,
            self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
            batch_first=True,
        )

    def forward(self, input_x, lengths):
        input_x = torch.nn.utils.rnn.pack_padded_sequence(
            input_x, lengths, enforce_sorted=False, batch_first=True)
        encoded, _ = self.rnn(input_x)
        encoded = torch.nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True)[0]

        return encoded[torch.arange(len(lengths)),lengths-1,:]

class Decoder(ModelBase):

    required_config = ['decoder_hidden_size', 'output_size', 'rnn_class', 'ptb_tag_embedding_size']

    def __init__(self, **decoder_config):
        super().__init__(decoder_config)

        self.embedding = torch.nn.Embedding(self.config.output_size, self.config.ptb_tag_embedding_size)

        self.rnn_cell = rnn_cell_classes[self.config.rnn_class](
            self.embedding.embedding_dim,
            self.config.decoder_hidden_size,
        )

        self.output = torch.nn.Linear(self.config.decoder_hidden_size, self.config.output_size)

    def forward(self, last_output, hidden):
        embedded = self.embedding(last_output)
        output_hidden = self.rnn_cell(embedded, hidden)
        return self.output(output_hidden), output_hidden

class EncoderDecoder(ModelBase):

    required_config = ['ptb_tag_embedding_size', 'max_sequence_length', 'beam_size']

    def __init__(self, corpus=None, **config):
        super().__init__(config)

        self.corpus = corpus

        # self.ptb_tag_embedding = torch.nn.Embedding(self.corpus.output_size, self.config.ptb_tag_embedding_size)

        self.encoder = Encoder(
            input_size=self.encoder_input_size,
            **config)

        self.decoder = Decoder(
            decoder_hidden_size=self.encoder.config.hidden_size * 2,
            output_size=self.corpus.output_size,
            # ptb_tag_embedding=self.ptb_tag_embedding,
            **config)

    def forward(self, input_data, gold_output=None, debug_gold=None, greedy=False):
        input_x, lengths = self.concat_input(input_data)
        encoded = self.encoder(input_x, lengths)
        if gold_output is not None:
            return self.teacher_enforcing(encoded, gold_output)
        elif greedy:
            return self.greedy(encoded)
        else:
            return self.beam_search(encoded)

    def greedy(self, encoded):
        batch_size = len(encoded)

        next_tokens = self.corpus.get_bos_tensor(batch_size).to(encoded.device)
        hidden = encoded

        log_probs, tokens = [], []

        for t in range(1, self.config.max_sequence_length):
            output_logits, hidden = self.decoder(next_tokens, hidden)
            output_log_prob = torch.nn.functional.log_softmax(output_logits, dim=1)
            next_tokens = output_log_prob.max(dim=1)[1]

            if t > 1:
                has_finished = tokens[-1] == self.corpus.eos_index
                next_tokens[has_finished] = self.corpus.eos_index

            log_probs.append(output_log_prob)
            tokens.append(next_tokens)
            if t > 1 and all(has_finished == self.corpus.eos_index):
                return torch.stack(log_probs, dim=1), torch.stack(tokens, dim=1)

        return torch.stack(log_probs, dim=1), torch.stack(tokens, dim=1)

    def beam_search(self, encoded):
        batch_size = len(encoded)

        bos_tensor = self.corpus.get_bos_tensor(batch_size).to(encoded.device)
        output_logits, hidden = self.decoder(bos_tensor, encoded)
        output_log_prob = torch.nn.functional.log_softmax(output_logits, dim=1)

        values, indices = output_log_prob.topk(self.config.beam_size, dim=1)

        log_prob_history = values.unsqueeze(-1)
        token_history = indices.unsqueeze(-1)

        last_output = indices.flatten()
        hidden = hidden.repeat(1, self.config.beam_size).view(batch_size*self.config.beam_size, -1)

        for t in range(1, self.config.max_sequence_length):
            output_logits_t, hidden = self.decoder(last_output, hidden)

            output_log_pb = torch.nn.functional.log_softmax(output_logits_t, dim=1)
            output_log_pb = output_log_pb.view(batch_size, self.config.beam_size, -1)
            combined_log_pb = log_prob_history[:,:,-1].unsqueeze(-1) + output_log_pb

            is_eos_index = (torch.arange(self.decoder.config.output_size) == self.corpus.eos_index).to(encoded.device)
            has_finished = (token_history[:,:,-1]==self.corpus.eos_index).unsqueeze(-1)
            ignored_terminated_log_pb = combined_log_pb.masked_fill(is_eos_index & has_finished, 0.)
            ignored_terminated_log_pb = ignored_terminated_log_pb.masked_fill((~is_eos_index) & has_finished, -float('inf'))

            values, indices = ignored_terminated_log_pb.view(batch_size, -1).topk(self.config.beam_size, dim=1)
            tokens = indices % self.decoder.config.output_size
            which_beam = indices.div(self.decoder.config.output_size, rounding_mode='floor')

            # update hidden
            hidden_reshaped = hidden.view(batch_size, self.config.beam_size, -1)
            new_hidden = hidden_reshaped.gather(1, which_beam.unsqueeze(-1).expand_as(hidden_reshaped))
            hidden = new_hidden.view(batch_size * self.config.beam_size, -1)

            # update last_output
            last_output = tokens.flatten()

            log_prob_history = torch.cat([log_prob_history.gather(1, which_beam.unsqueeze(-1).expand_as(log_prob_history)), values.unsqueeze(-1)], dim=-1)
            if token_history is None:
                token_history = tokens.unsqueeze(-1)
            else:
                token_history = torch.cat([token_history, tokens.unsqueeze(-1)], dim=-1)

            if all(token_history[:,0,-1] == self.corpus.eos_index):
                return log_prob_history, token_history
        return log_prob_history, token_history

    def teacher_enforcing(self, encoded, gold_output):
        output_logits_lst = []
        hidden = encoded
        for t in range(gold_output['tree_sequence_length'].max().item()-1):
            last_output = gold_output['tree_sequence'][:,t]
            output_logits_t, hidden = self.decoder(last_output, hidden)
            output_logits_lst.append(output_logits_t)
        return torch.stack(output_logits_lst, dim=2)

    @property
    def encoder_input_size(self):
        encoder_input_size = 0
        if self.corpus.required_data.pos:
            encoder_input_size += self.config.ptb_tag_embedding_size
        if self.corpus.required_data.rh:
            encoder_input_size += 1
        if 'bert' in self.corpus.required_data.embeddings:
            bert_sizes = {
                'bert-base-cased': 768,
                'prajjwal1/bert-tiny': 256,
            }
            encoder_input_size += bert_sizes[self.corpus.required_data.embeddings]
        if 'word2vec' in self.corpus.required_data.embeddings:
            encoder_input_size += 300

        return encoder_input_size

    def concat_input(self, input_data):
        data = []
        if self.corpus.required_data.pos:
            if input_data['pos'][0] == 'attack!':
                original_data = self.decoder.embedding(input_data['pos'][1])
                random_data = torch.rand(original_data.shape).to(input_data['pos'][1].device)
                data.append(random_data)
            else:
                data.append(self.decoder.embedding(input_data['pos']))
        if self.corpus.required_data.rh:
            data.append(input_data['rh'].unsqueeze(-1))
        if 'embeddings' in input_data:
            data.append(input_data['embeddings'])
        return torch.concat(data, dim=2), input_data['lengths']
