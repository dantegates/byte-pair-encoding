# byte-pair-encoding

This repository implements an `sklearn Transformer` compatible [byte pair encoder](https://arxiv.org/pdf/1508.07909.pdf)
for Neural Machine Translation and other NLP tasks.

# Performance

A [trie](https://en.wikipedia.org/wiki/Trie) is used to transform an input corpus in linear time by using one
`dict` lookup per character in the corpus. As a benchmark 2,177,020 characters can be encoded in 1.71 seconds
on my MacBook Pro.

Performance-wise, learning the encodings could use a little bit of work. This too should scale linearly with
the input corpus the embeddings are learned from, however learning encodings from 2,177,020 takes 27min 44s.

# Why another BPE library

There are several other libraries implementing byte pair encoding, however all seem to lack a `sklearn`,
`keras`, `jupyter notebook`, or `python` friendly API. This repository seeks to provide a byte pair encoder
that accomodate a smooth workflow with these other libraries.

# Implementation details

- As mentioned above `.fit()` and `.transform()` scale linearly, but `.fit()` could use some additional
  optimizations. Simply using multiprocessing `get_stats()` would be a simple place to start.
- In contrast to [Sennrich et el.](https://arxiv.org/pdf/1508.07909.pdf) whitespace is treated as a token
  in its own right, as in [sentencepiece](https://github.com/google/sentencepiece) and encodings can be
  learned accross word boundaries. Respecting word boundaries as in Sennrich et al. would improve `.fit()`
  performance and perhaps this could be added as a parameter.
- Several helper functions used during `.fit()` are implemented with `cython`. As each iteration in `.fit()`
  requires many loops over the training corpus simply typing the variable we loop over gives a big lift in
  performance.
- Currently the only available parameter is the number of merges, or iterations, in `.fit()`. Perhaps a
  better way to parameterize this would be with desired vocab size as in [sentencepiece](https://github.com/google/sentencepiece).
