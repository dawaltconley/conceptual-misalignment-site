"""Contextual-embedding semantic-space analysis (Wu & Wang 2025, adapted).

Task 1: build a monolingual Chinese semantic space of the Confucian cardinal
virtues from GujiRoBERTa final-layer embeddings, using per-occurrence mean over
subwords then max-pooling across occurrences.

Modules are written to run with ``scripts/`` as the working directory (matching
the rest of the repo's scripts), so ``from utils import ...`` / ``from config
import ...`` resolve.
"""
