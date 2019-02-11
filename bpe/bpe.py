import collections
import time

import sklearn
import numpy as np

from .utils import get_stats, merge_vocab
from .tree import build_bpe_tree, apply_bpe_tree


class BytePairEncoder(sklearn.base.TransformerMixin):    
    _unkown_character = '<unk>'
    _space_escape = '▁'

    def __init__(self, target_vocab_size, vocab_threshold=None, log_level=None):
        self.target_vocab_size = target_vocab_size
        self.log_level = log_level
        self.vocab_threshold = vocab_threshold

        # these will all be set during .fit()
        self.vocab = None
        self._vocab_stats = None
        self._reverse_vocab = None
        self._bpe_tree = None

    def fit(self, X):
        # get the initial vocabular consisting of all unique characters
        initial_vocab = set(X)
        initial_vocab.add(self._space_escape)

        words = self._split_X(X)
        vocab = [(list(word), freq) for word, freq in collections.Counter(words).items()]

        t_started = time.time()
        i = 0
        while self._compute_num_subwords(vocab) < self.target_vocab_size:
            if self.log_level is not None (i + 1) % self.log_level == 0:
                print(f'{i+1} iterations complete in {time.time() - t_started}')
            pair_stats, pair_index = get_stats(vocab)
            best = max(pair_stats, key=pair_stats.get)
            if self.vocab_threshold is not None \
                    and pair_stats[best] < self.vocab_threshold:
                print(f'Stopping after {i} iterations. Best pair occurs '
                      f'{pair_stats[best]} < {self.vocab_threshold} times')
                break
            vocab = merge_vocab(best, vocab, pair_index[best])
            i += 1

        # build the final vocabulary
        vocab_stats = collections.Counter()
        _ = [vocab_stats.update({subword: freq})
                                for word, freq in vocab
                                for subword in word]
        final_vocab = set(vocab_stats)
        final_vocab.update(initial_vocab)
        final_vocab = {k: i for i, k in enumerate(final_vocab, start=1)}
        final_vocab[self._unkown_character] = 0
        self.vocab = final_vocab

        # these are needed for .transform() and .inverse_transform()
        self._reverse_vocab = {i: k for k, i in self.vocab.items()}
        self._bpe_tree = build_bpe_tree(self.vocab)

        # keep this for curiosity/debugging
        self._vocab_stats = vocab_stats

    def transform(self, X):
        X = self._split_X(X)
        return np.concatenate([self._transform_string(x) for x in X])

    def _transform_string(self, X):
        tokens = apply_bpe_tree(X, self._bpe_tree)
        return np.array([0 if t is None else t for t in tokens])                

    def inverse_transform(self, X):
        return [self._reverse_vocab[t] for t in X]

    def _split_X(self, X):
        return [word + self._space_escape for word in X.split()]

    def _compute_num_subwords(self, vocab):
        return len(set(subword for word, _ in vocab for subword in word))

    @property
    def vocab_size(self):
        return len(self.vocab)
