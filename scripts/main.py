from config import TERMS, SEP, CTEXT
from mengzi import fetch_mengzi, Chapter as MengziChapter
from scrape_sep import search_sep, SEP as SEPArticle
from utils import filter_to_sent_node_lists, build_cooccurrence_network
from nlp.english import tokenize_english_html
from nlp.chinese import tokenize_classical_chinese, STOPWORDS as CHINESE_STOPWORDS
from slugify import slugify
from networkx import node_link_data, Graph
from dataclasses import dataclass, asdict
import json
from pathlib import Path

cooccurrence = []

print("Fetching Mengzi...")
mengzi = fetch_mengzi()
print(f"  Fetched {len(mengzi)} chapters")


def get_cooccurence_english(term: str, text: str) -> Graph:
    tokens = tokenize_english_html(text)
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=22)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


@dataclass
class Source:
    url: str
    title: str
    description: str
    co_occurance: object

    @classmethod
    def from_sep(cls, term: str, sep: SEPArticle) -> "Source":
        print(f"  [{term}] Processing article: {sep.title!r}")
        tokens = tokenize_english_html(sep.text)
        sent_node_lists, nodes = filter_to_sent_node_lists(
            tokens, term, min_freq=22)
        G = build_cooccurrence_network(
            sent_node_lists, nodes, term, max_nodes=15)

        return cls(
            url=sep.url,
            title=sep.title,
            description=sep.description,
            co_occurance=node_link_data(G)
        )

    @classmethod
    def from_mengzi(cls, term: str, chapter: MengziChapter) -> "Source":
        print(f"  [{term}] Tokenizing classical Chinese...")
        tokens = tokenize_classical_chinese(chapter.text)

        print(f"  [{term}] {len(tokens)} tokens — building cooccurrence network...")
        sent_node_lists, nodes = filter_to_sent_node_lists(
            tokens, term, min_freq=3, stopwords=CHINESE_STOPWORDS)
        G = build_cooccurrence_network(
            sent_node_lists, nodes, term, max_nodes=15)

        return cls(
            url=chapter.url,
            title=chapter.title,
            description=chapter.description,
            co_occurance=node_link_data(G)
        )


@dataclass
class TermData:
    term: str
    sources: list[Source]

    def save_json(self, filepath: Path) -> None:
        serialized = json.dumps(asdict(self), indent=2)
        filepath.write_text(serialized, encoding="utf-8")


def save_term_data_json(term: TermData, filepath: Path) -> None:
    serialized = json.dumps(term, indent=2)
    filepath.write_text(serialized, encoding="utf-8")


for term_pairs in TERMS:
    print(f"\n=== {term_pairs.hanzi} / {', '.join(term_pairs.english)} ===")

    # First run NLP for the Chinese term in the Mengzi
    term = term_pairs.hanzi
    data = TermData(
        term_pairs.hanzi,
        [Source.from_mengzi(term, chapter) for chapter in mengzi]
    )
    data.save_json(CTEXT / f"{term}.json")

    # Now iterate for each associated english term
    for term in term_pairs.english:
        print(f"\n  [{term}] Searching SEP...")
        sep_articles = search_sep(term, 4)
        print(f"  [{term}] {len(sep_articles)} articles found")

        # run NLP on each article
        data = TermData(term, [Source.from_sep(term, a) for a in sep_articles])
        data.save_json(SEP / f"{slugify(term)}.json")
