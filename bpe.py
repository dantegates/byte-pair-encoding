import collections

import numpy as np
import sklearn


def get_stats(vocab, removed_indices):
    pairs = collections.defaultdict(int)
    indices = collections.defaultdict(list)
    valid_indices = (i for i in range(len(vocab) - 1)
                     if not i in removed_indices)
    i_left = next(valid_indices)
    for i_right in valid_indices:
        pair = vocab[i_left], vocab[i_right]
        pairs[pair] += 1
        indices[pair].append(i_left)
    return pairs, indices

def merge_vocab(pair, vocab, pair_indices, removed_indices):
    new = ''.join(pair)
    for i in reversed(pair_indices):
        vocab[i] = new
    removed_indices.update(pair_indices)
    return vocab


class BytePairEncoder(sklearn.base.TransformerMixin):
    def __init__(self, n_merges):
        self.n_merges = n_merges

        self.vocab = {}
        self._reverse_vocab = {}
        self._bpe_tree = None

        self._space_escape = '▁'
        self._unkown_token = 0

    def fit(self, X):
        vocab = list(self._process_X(X))
        initial_vocab = set(vocab)
        removed_indices = set()
        for _ in range(self.n_merges):
            pairs, pair_index = get_stats(vocab, removed_indices)
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab, pair_index[best], removed_indices)

        # reserve 0 for unkowns
        vocab = set(vocab)
        vocab.update(initial_vocab)
        self.vocab = {k: i for i, k in enumerate(vocab, start=1)}
        self._reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._bpe_tree = build_bpe_tree(self.vocab)

    def transform(self, X):
        X = self._process_X(X)
        tokens = apply_bpe_tree(X, self._bpe_tree)
        return np.array([self._unkown_token if t is None else t for t in tokens])

    def inverse_transform(self, X):
        return [self._reverse_vocab[t] if t > 0 else '<unk>' for t in X]

    def _process_X(self, X):
        return self._space_escape.join(X.split())   


class Node:
    def __init__(self):
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
    last_node = tree
    pos = 0
    while pos <= len(text) - 1:
        node = last_node.get(text[pos])
        if node is None:
            output.append(last_node.index)
            if last_node is not tree:
                last_node = tree
                continue
            node = tree
        last_node = node
        pos += 1
    output.append(last_node.index)
    return output
