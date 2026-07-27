from config import TERMS, DATA, SEP, CTEXT
from lib import TermData, NLPSource
from corpus.build import build_chinese_corpus, build_english_corpus
from cooccurrence.pmi import filter_to_sent_node_lists, build_cooccurrence_network
from nlp.english import tokenize_english_html
from nlp.chinese import tokenize_classical_chinese, STOPWORDS as CHINESE_STOPWORDS
from slugify import slugify
from networkx import Graph
import json


def get_cooccurence_english(term: str, text: str | list[list[str]]) -> Graph | None:
    tokens = text if isinstance(text, list) else tokenize_english_html(text)
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=22)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


def get_cooccurence_chinese(term: str, text: str) -> Graph | None:
    print(f"  [{term}] Tokenizing classical Chinese...")
    tokens = tokenize_classical_chinese(text)

    print(f"  [{term}] {len(tokens)} tokens — building cooccurrence network...")
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=3, stopwords=CHINESE_STOPWORDS)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


def main() -> None:
    mengzi = build_chinese_corpus()

    with open(DATA / "terms.json", "w") as file:
        serialized = [{"hanzi": t.hanzi, "english": t.english} for t in TERMS]
        file.write(json.dumps(serialized))

    for term_pairs in TERMS:
        print(f"\n=== {term_pairs.hanzi} / {', '.join(term_pairs.english)} ===")

        # First run NLP for the Chinese term in the Mengzi
        term = term_pairs.hanzi
        sources: list[NLPSource] = []
        for m in [mengzi] + [ch for ch in mengzi.chapters]:
            cooccurrence = get_cooccurence_chinese(term, m.text)
            sources.append(NLPSource(m, cooccurrence=cooccurrence))

        data = TermData(term, sources)
        data.save_json(CTEXT / f"{term}_pmi.json")

    sep = build_english_corpus()

    for term, search in sep:
        term_count = 0
        canonical: str = term.label
        stems: set[str] = set()
        sources: list[NLPSource] = []
        total_tokens: list[list[str]] = []
        for article in search.articles:
            tokens = tokenize_english_html(article.text)
            # replace words that match a stem with its canonical label
            for i, sent in enumerate(tokens):
                for j, t in enumerate(sent):
                    if term.matches(t):
                        term_count += 1
                        stems.add(t)
                        tokens[i][j] = canonical
            total_tokens = total_tokens + tokens
            source = NLPSource(
                article,
                cooccurrence=get_cooccurence_english(canonical, tokens)
            )
            sources.append(source)

        total = NLPSource(search, cooccurrence=get_cooccurence_english(
            canonical, total_tokens))
        sources = [total] + sources

        print(
            f"  [{term.label}] found {term_count} occurances accross {len(search.articles)} articles")
        data = TermData(term.label, stems=list(stems), sources=sources)
        data.save_json(SEP / f"{slugify(term.label)}_pmi.json")


if __name__ == "__main__":
    main()
