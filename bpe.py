import time

import sklearn
import numpy as np

from bpe_utils import get_stas, merge_vocab


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.index = None

    def __repr__(self):
        return f'Node(index={self.index}, children={self.children})'
    
    def get(self, key, default=None):
        return self.children.get(key, default)
    
    def __getitem__(self, key):
        return self.children[key]
    
    def __setitem__(self, key, value):
        self.children[key] = value
        
    def __contains__(self, key):
        return key in self.children


def build_bpe_tree(vocab):
    root = Node()
    for word, index in vocab.items():
        current_node = root
        for n, c in enumerate(word, start=1):
            if not c in current_node:
                current_node[c] = Node()
            current_node = current_node[c]
            if n == len(word):
                current_node.index = index
    return root


def apply_bpe_tree(text, tree):
    output = []
    pos = 0
    last_node = tree
    while pos <= len(text) - 1:
        node = last_node.get(text[pos])
        # we can't search the tree any further
        if node is None:
            # we couldn't search the tree any further but we
            # ended up at a node that doesn't correspond to a
            # word in the learned vocabulary.
            # In this case we'll traverse back through the tree
            # until we hit a node with an index.
            if last_node.index is None:
                while last_node.index is not None:
                    last_node = last_node.parent
                    pos -= 1
            # add the last seen index to the output
            # and reset variables for next run through
            output.append(last_node.index)
            if last_node is not tree:
                last_node = tree
                continue
            node = tree
        last_node = node
        pos += 1
    output.append(last_node.index)
    return output


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
