def get_stats(list vocab, set removed_indices):
    cdef int i_left, i_right
    cdef dict pairs = {}
    cdef dict indices = {}
    valid_indices = (i for i in range(len(vocab) - 1)
                     if not i in removed_indices)
    i_left = next(valid_indices)
    for i_right in valid_indices:
        pair = vocab[i_left], vocab[i_right]
        if not pair in pairs:
            pairs[pair] = 0
        pairs[pair] += 1
        if not pair in indices:
            indices[pair] = []
        indices[pair].append(i_left)
    return pairs, indices

def merge_vocab(tuple pair, list vocab, list pair_indices, set removed_indices):
    cdef str new = ''.join(pair)
    cdef int i
    for i in pair_indices:
        vocab[i] = new
    removed_indices.update(pair_indices)
    return vocab