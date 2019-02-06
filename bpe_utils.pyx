def get_stats(list vocab):
    cdef int i
    cdef dict pairs = {}
    cdef dict indices = {}
    for i in range(len(vocab) - 1):
        pair = vocab[i], vocab[i+1]
        if not pair in pairs:
            pairs[pair] = 0
        pairs[pair] += 1
        if not pair in indices:
            indices[pair] = []
        indices[pair].append(i)
    return pairs, indices

def merge_vocab(tuple pair, list vocab, list indices):
    cdef str new = ''.join(pair)
    cdef int i
    for i in reversed(indices):
        vocab[i] = new
        vocab.pop(i+1)
    return vocab