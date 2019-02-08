import time

import sklearn
import numpy as np

from .bpe_utils import get_stats, merge_vocab
from .tree import build_bpe_tree, apply_bpe_tree


class BytePairEncoder(sklearn.base.TransformerMixin):
    def __init__(self, n_merges, n_jobs=None, chunksize=None, log_level=None,
                 vocab_threshold=None):
        self.n_merges = n_merges
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.log_level = log_level
        self.vocab_threshold = vocab_threshold
        self._space_escape = '▁'
        self._unkown_token = 0
        self._unkown_character = '<unk>'

    def fit(self, X):
        vocab = list(self._process_X(X))
        initial_vocab = set(vocab)
        removed_indices = set()
        t_started = time.time()
        for i in range(self.n_merges):
            if self.log_level is not None and i % self.log_level == 0:
                print(f'{i+1} iterations complete in {time.time() - t_started}')
            pairs, pair_index = get_stats(vocab, removed_indices)
            best = max(pairs, key=pairs.get)
            if self.vocab_threshold is not None \
                    and pairs[best] < self.vocab_threshold:
                print(f'Stopping after {i} iterations. Best pair occurs '
                      f'{pairs[best]} < {self.vocab_threshold} times')
                break
            vocab = merge_vocab(best, vocab, pair_index[best], removed_indices)

        self._vocab_stats = collections.Counter(vocab)
        vocab = set(vocab)
        vocab.update(initial_vocab)
        # reserve 0 for unkowns
        self.vocab = {k: i for i, k in enumerate(vocab, start=1)}
        self.vocab[self._unkown_character] = 0
        self._reverse_vocab = {i: k for k, i in self.vocab.items()}
        self._bpe_tree = build_bpe_tree(self.vocab)

    def transform(self, X):
        X = self._process_X(X)
        tokens = apply_bpe_tree(X, self._bpe_tree)
        return np.array([self._unkown_token if t is None else t for t in tokens])

    def inverse_transform(self, X):
        return [self._reverse_vocab[t] for t in X]

    def _process_X(self, X):
        return self._space_escape.join(X.split())
