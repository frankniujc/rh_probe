# Using Roark-Hollingshead Distance to Probe BERT's Syntactic Competence

Code, data, and results of the [BlackboxNLP 2022](https://blackboxnlp.github.io/) paper [_Using Roark-Hollingshead Distance to Probe BERT's Syntactic Competence_](https://www.cs.toronto.edu/~niu/papers/niu2022rh.pdf)

**Abstract**: Probing BERT's general ability to reason about syntax is no simple endeavour, primarily because of the uncertainty surrounding how large language models represent syntactic structure. Many prior accounts of BERT's agility as a syntactic tool (Clark et al., 2013; Lau et al., 2014; Marvin and Linzen, 2018; Chowdhury and Zamparelli, 2018; Warstadt et al., 2019, 2020; Hu et al., 2020) have therefore confined themselves to studying very specific linguistic phenomena, and there has still been no definitive answer as to whether BERT "knows" syntax.

The advent of perturbed masking (Wu et al., 2020) would then seem to be significant, because this is a parameter-free probing method that directly samples syntactic trees from BERT's embeddings. These sampled trees outperform a right-branching baseline, thus providing preliminary evidence that BERT's syntactic competence bests a simple baseline. This baseline is underwhelming, however, and our reappraisal below suggests that this result, too, is inconclusive.

We propose *RH Probe*, an encoder-decoder probing architecture that operates on two probing tasks. We find strong empirical evidence confirming the existence of important syntactic information in BERT, but this information alone appears not to be enough to reproduce syntax in its entirety. Our probe makes crucial use of a conjecture made by Roark and Hollingshead (2008) that a particular lexical annotation that we shall call RH distance is a sufficient encoding of unlabelled binary syntactic trees, and we prove this conjecture.

## Preparation

### Install Dependencies
Create a virtual environment and install the required dependencies.
```bash
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```
Install [PyTorch](https://pytorch.org/get-started/locally/).
Please follow PyTorch's official installation guide.
This paper is implemented using PyTorch version `1.10.2+cu113`.

### Data Preparation
Follow [Kim et al.'s (2019)](https://github.com/harvardnlp/compound-pcfg) instructions to preprocess PTB data.  When finished, put the processed corpus `ptb-train.txt`, `ptb-valid.txt` and `ptb-test.txt` in `data/cpcfg_process`.

### Prepare word2vec Preparation
Download the word2vec (`GoogleNews-vectors-negative300.bin.gz`) model from [the official site](https://code.google.com/archive/p/word2vec/) to `word2vec/`.  Unzip the model and extract the useful entries.
```sh
cd word2vec
gunzip GoogleNews-vectors-negative300.bin.gz
cd ..
python scripts/python_code/load_word2vec.py
```

## Train Probes
You can train the probes by running the following script:
```sh
./scripts/conduct_exp.sh
```