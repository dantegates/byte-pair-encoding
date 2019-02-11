def get_stats(list vocab):
    cdef c1, c2
    cdef list word
    cdef int freq, vocab_pos, word_pos
    cdef dict pair_stats = {}
    cdef dict pair_indices = {}
    cdef tuple pair
    for vocab_pos in range(len(vocab)):
        word, freq = vocab[vocab_pos]
        for word_pos in range(len(word) - 1):
            pair = word[word_pos], word[word_pos + 1]
            if not pair in pair_stats:
                pair_stats[pair] = 0
            pair_stats[pair] += freq
            if not pair in pair_indices:
                pair_indices[pair] = []
            pair_indices[pair].append((vocab_pos, word_pos))
    return pair_stats, pair_indices


def merge_vocab(str new, list vocab, list pair_indices):
    cdef int vocab_pos, word_pos
    cdef list word
    for vocab_pos, word_pos in reversed(pair_indices):
        word, _ = vocab[vocab_pos]
        word[word_pos] = new
        word.pop(word_pos + 1)
    return vocab
