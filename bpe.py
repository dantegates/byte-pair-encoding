import collections

import sklearn
from bpe_utils import get_stats, merge_vocab


class BytePairEncoder(sklearn.base.TransformerMixin):
    def __init__(self, n_merges, vocab_size):
        self.n_merges = n_merges
        self.vocab_size = vocab_size
        self._space_escape = '▁'
        self._unkown_token = 0
        self.vocab = {}
        self._bpe_tree = None

    def fit(self, X):
        vocab = self._split_X(X)
        for _ in range(self.n_merges):
            pairs, pair_index = get_stats(vocab)
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab, pair_index[best])

        vocab = collections.Counter(vocab)
        # reserve 0 for unkowns
        self.vocab = {v: i for i, v in enumerate(sorted(vocab, key=vocab.get, reverse=True), start=1)}
        self._bpe_tree = build_bpe_tree(self.vocab)

    def transform(self, X):
        tokens = apply_bpe_tree(X, self._bpe_tree)
        return [self._unkown_token if t is None else t
                for t in tokens]

    def inverse_transform(self, X):
        pass

    def _split_X(self, X):
        return list(self._space_escape.join(X.split()))


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
