# byte-pair-encoding

This repository implements an `sklearn Transformer` compatible [byte pair encoder](https://arxiv.org/pdf/1508.07909.pdf)
for Neural Machine Translation and other NLP tasks.

# Performance

A [trie](https://en.wikipedia.org/wiki/Trie) is used to transform an input corpus in linear time by using one
`dict` lookup per character in the corpus. As a benchmark 2,162,996 characters (from a bunch of concatenated
sentences) can be encoded in 2.5s seconds on my MacBook Pro.

Learning the encodings is more expensive. 12,386 subword units were learned from 2,162,996 characters in
29min 29s.

# Why another BPE library

There are several other libraries implementing byte pair encoding, however these implementations focus on
providing a command line interface [here](https://github.com/rsennrich/subword-nmt) and
[here](https://github.com/google/sentencepiece) or are embedded inside of a large framework such
as [tensor2tensor]().

This repository seeks to provide a lightweight implementation that you can `pip install` or tweek as desired
that easily fits in a workflow using tools such as `sklearn`, `keras` and `jupyter notebook`s.

# Implementation details

- As mentioned above `.fit()` and `.transform()` scale linearly, but `.fit()` could use some additional
  optimizations. Simply using multiprocessing `get_stats()` would be a simple place to start.
- As in other implementations whitespace is treated as a token in its own right and encodings are not
  learned accross word boundaries.
- Several helper functions used during `.fit()` are implemented with `cython`. As each iteration in `.fit()`
  requires many loops over the training corpus simply typing the variable we loop over gives a big lift in
  performance.
