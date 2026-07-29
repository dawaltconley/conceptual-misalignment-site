from typing import Protocol, NamedTuple
from pathlib import Path
from collections import Counter
import config
from config import TERMS, CTEXT, SEP, EMBEDDINGS
import lib
from nlp.chinese import CHINESE_STOPWORDS
from spacy.tokens import Span, Doc
from corpus.build import build_chinese_corpus, build_english_corpus
from parse import parse_sep_article, parse_mengzi_chapter
from embeddings import analyze, vectors
from cooccurrence.pmi import filter_to_sent_node_lists, build_cooccurrence_network
from embeddings.occurrences_unified import content_key, build_vocab, build_segments, MatchFn, Segment, SourceDoc
from embeddings.model import Embedder
from numpy import ndarray
from networkx import Graph
import numpy as np


def get_cooccurence(
        term: str,
        sources: list[SourceDoc],
        min_freq: int,
        *,
        match_fn: MatchFn | None = None,
        content_pos: set[str] | None = None,
        stopwords: set[str] = set(),
        max_nodes: int = 15
) -> Graph | None:
    vocab = build_vocab(
        sources, min_freq,
        match_fn=match_fn,
        content_pos=content_pos,
        stopwords=stopwords
    )
    vocab.add(term)
    vocab_sents: list[Span] = []
    vocab_lemmas: list[list[str]] = []
    for source in sources:
        for sent in source.doc.sents:
            lemmas = [content_key(l, match_fn) for l in sent]
            lemmas = [l for l in lemmas if l]
            if any(l in vocab for l in lemmas):
                vocab_sents.append(sent)
                vocab_lemmas.append(lemmas)

    return build_cooccurrence_network(vocab_lemmas, vocab, term, max_nodes)


def run_mengzi(
        *,
        artifacts=config.ANALYSIS / "mengzi",
        min_freq=5,
        center=True,
        content_pos: set[str] | None = None,
        stopwords=CHINESE_STOPWORDS,
        reduce_to_dims: int = 50
) -> None:
    targets = frozenset(t.hanzi for t in TERMS)
    mengzi = build_chinese_corpus()
    parsed = [
        SourceDoc(c, parse_mengzi_chapter(c))
        for c in mengzi.chapters
    ]
    parsed = [SourceDoc(mengzi, Doc.from_docs(
        [p.doc for p in parsed]))] + parsed

    # calculate pmi
    cooccurrences: list[lib.Cooccurrence] = []
    for source_doc in parsed:
        source, doc = source_doc
        for target in targets:
            target_count = Counter([t.lemma_ for t in doc])[target]
            term = lib.TermData(target, target_count)
            # TODO add content_pos?
            cooc = get_cooccurence(
                target, [source_doc], min_freq, content_pos=content_pos, stopwords=stopwords)
            if cooc is None:
                print(f"no co-occurence for {term} in {source.title}")
            cooccurrences.append(lib.Cooccurrence(term, source, cooc))

    # save pmi
    for cooc in cooccurrences:
        cooc.save_json(CTEXT / f"{cooc.term.label}_{cooc.source.id}.json")

    # calculate embeddings
    embedder = Embedder("hsc748NLP/GujiRoBERTa_fan")
    word_vectors = embed(embedder, parsed, targets, min_freq=min_freq)
    labels, matrix = pool(word_vectors, targets)
    mean = None
    if center:
        matrix, mean = vectors.center_matrix(matrix)

    # save vectors
    vectors.save_vectors(artifacts, labels,
                         matrix, targets, word_vectors, mean=mean)

    # produce analysis artifacts
    target_occ = occurrences(word_vectors, targets, mean)
    is_target = np.array([lbl in targets for lbl in labels])

    # summary = analyze.run_analysis(
    #     labels, matrix, is_target, target_occ,
    #     artifacts, threshold, args.kmeans_k,
    #     method=method, knn_k=args.knn_k, sim_transform=args.sim_transform,
    #     max_nodes=max_nodes,
    # )
    # print_summary(summary, is_centered=center, out_dir=out_dir)

    # calculate similarity

    # save embeddings
    reduced = reduce_vectors(matrix, dims=reduce_to_dims)
    embeddings = lib.Embeddings.from_matrix(
        mengzi, labels, reduced, targets, communities={})
    embeddings.save_json(EMBEDDINGS / "mengzi.json")


def print_summary(summary: dict, is_centered: bool, out_dir: Path) -> None:
    print("\n=== summary ===")
    for row in summary["cohesion_variance"]:
        print(f"  {row['term']}: n={row['n']} cohesion={row['cohesion']} "
              f"variance={row['variance']}")
    net = (f"knn (k={summary['knn_k']})" if summary["method"] == "knn"
           else f"threshold {summary['threshold']}")
    print(f"  network: {net}  centered={is_centered}  ")
    print(f"  Louvain communities: {summary['louvain_communities']}")
    print(f"  artifacts written to: {out_dir.resolve()}")


def reduce_vectors(matrix: np.ndarray, dims: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    """Mean-center + L2-normalize, then PCA to `dims` (variance-ordered columns)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.clip(norms, 1e-12, None)
    n_components = min(dims, unit.shape[1], unit.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(unit)


def main() -> None:
    run_mengzi()


if __name__ == "__main__":
    main()


type WordVectors = dict[str, list[ndarray]]


def embed(emb: Embedder, sources: list[SourceDoc], target_labels: frozenset[str],  *, match_fn: MatchFn | None = None, min_freq=10, batch_size=32) -> WordVectors:
    # # a set of canonical labels for target terms
    # target_labels = frozenset(
    #     [t.label if isinstance(t, Rendering) else t for t in targets])

    # a set of terms of interest: the core vocab and those above a minimum frequency
    # minus stopwords and terms not matching target POS
    # terms are normalized to the result of match_fn, else their lemma.
    # ensure the target terms are present, regardless of their frequency
    vocab = build_vocab(sources, min_freq, match_fn)
    vocab |= target_labels

    # warn if any of our core terms are unknown to the model
    unk_check(emb, target_labels)

    # segment the sources into chunks that can be managed by the embedder
    # they must be less than the model limit, which for BERT models is 512 tokens
    segments = segment(emb, sources, vocab, match_fn)
    # segments = build_segments(
    #     sources,
    #     vocab,
    #     match_fn=match_fn,
    #     sent_len_fn=emb.token_lengths,
    #     max_tokens=emb.max_length
    # )

    return emb.embed(segments, batch_size)


def pool(word_vectors: WordVectors, target_labels: frozenset[str]):
    labels, matrix = vectors.max_pool(word_vectors)
    is_target = np.array([lbl in target_labels for lbl in labels])
    print(f"pooled     : {len(labels)} words ({int(is_target.sum())} targets)")
    return labels, matrix


def occurrences(word_vectors: WordVectors, targets: frozenset[str], mean: ndarray | None = None):
    """one entry per target term holding its stacked (n_occurrences x H) occurrence vectors,
    for cohesion / variance analysis."""
    return {
        w: np.stack(word_vectors[w]) - (mean or 0) for w in targets if word_vectors.get(w)
    }


def unk_check(emb: Embedder, words: set[str] | frozenset[str]) -> None:
    """Warn if any word we embed loses a character to [UNK]."""
    unk = emb.unk_words(sorted(words))
    if unk:
        print(f"WARNING: {len(unk)} word(s) tokenize to [UNK]:")
        for w, toks in list(unk.items())[:20]:
            print(f"    {w!r} -> {toks}")
    else:
        print(f"UNK check  : OK — all {len(words)} words in-vocab")


def segment(emb: Embedder, sources: list[SourceDoc], woi: set[str], match_fn: MatchFn | None) -> list[Segment]:
    """Split each passage into sentences and greedily pack under the token cap."""
    segments = build_segments(
        sources, woi,
        match_fn=match_fn,
        sent_len_fn=emb.token_lengths,
        max_tokens=emb.max_length
    )
    n_occ = sum(len(s.spans) for s in segments)
    n_tok = sum(len(s.doc) for s in sources)
    n_sources = len({s.source.id for s in sources})
    print(f"segmented  : {n_tok} tokens ({n_sources} source documents) -> "
          f"{len(segments)} segments (<= {emb.max_length} tok); {n_occ} occurrences")
    return segments
