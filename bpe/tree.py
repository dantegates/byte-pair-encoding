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
