"""Contextual-embedding semantic-space analysis (Wu & Wang 2025, adapted).

Task 1: build a monolingual Chinese semantic space of the Confucian cardinal
virtues from GujiRoBERTa final-layer embeddings, using per-occurrence mean over
subwords then max-pooling across occurrences.

Library modules only (model, vectors, analyze, occurrences, sep_occurrences);
the runnable entrypoints live in ``cli/`` (``embed_mengzi``, ``embed_sep``).
Run with ``scripts/`` as the working directory so the sibling packages
(``config``, ``graph``, ``nlp``, ``corpus``) resolve on ``sys.path``.
"""
