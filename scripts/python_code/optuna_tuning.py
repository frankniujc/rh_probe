import sys
sys.path.append('.')

import optuna
from torch.utils.data import DataLoader
from rh_probe.experiments import ParsedExperiment

def objective(trial):
    num_layers = trial.suggest_int("num_layers", 1, 5)
    encoder_hidden_size = trial.suggest_int("encoder_hidden_size", 50, 1000, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    ptb_tag_embedding_size = trial.suggest_int("ptb_tag_embedding_size", 50, 1000, step=50)

    config = {
        "rh": False,
        "pos": False,
        "embeddings": "bert-base-cased",

        "corpus_path": "data/cpcfg_process",
        "serialization_path": "data/corpus_cache/rh_pos_bert/kim_ptb.pkl",

        "device": "cuda",

        "batch_size": 64,
        "learning_rate": learning_rate,
        "eval_batch_size": 64,
        "epochs": 30,
        "ptb_tag_embedding_size": ptb_tag_embedding_size,
        "optimizer": "adam",
        "max_sequence_length": 1000,
        "beam_size": 5,

        "encoder": {
            "hidden_size": encoder_hidden_size,
            "num_layers": num_layers,
            "bidirectional": True,
            "dropout": 0.0,
            "rnn_class": "gru"
        },

        "decoder": {
            "rnn_class": "gru"
        }
    }

    exp = ParsedExperiment(config)
    dataloader = DataLoader(exp.corpus.train, batch_size=exp.config.batch_size, shuffle=True)
    exp.train_epoch(dataloader)
    metric = exp.evaluate(exp.corpus.valid)
    return metric.score[0]

if __name__ == '__main__':
    optuna.logging.enable_default_handler()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)