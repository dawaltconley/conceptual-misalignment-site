from config import TERMS, DATA, SEP, CTEXT
from mengzi import fetch_mengzi_full, fetch_mengzi_chapters
from scrape_sep import search_sep, SEP as SEPArticle
from inpho import is_chinese_philosophy
from utils import filter_to_sent_node_lists, build_cooccurrence_network
from nlp.english import tokenize_english_html
from nlp.chinese import tokenize_classical_chinese, STOPWORDS as CHINESE_STOPWORDS
from slugify import slugify
from networkx import node_link_data, Graph
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Protocol

cooccurrence = []

print("Fetching Mengzi full text...")
mengzi = [fetch_mengzi_full()]
print("Fetching Mengzi chapters...")
mengzi = mengzi + fetch_mengzi_chapters()


def get_cooccurence_english(term: str, text: str) -> Graph:
    tokens = tokenize_english_html(text)
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=22)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


def get_cooccurence_chinese(term: str, text: str) -> Graph:
    print(f"  [{term}] Tokenizing classical Chinese...")
    tokens = tokenize_classical_chinese(text)

    print(f"  [{term}] {len(tokens)} tokens — building cooccurrence network...")
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=3, stopwords=CHINESE_STOPWORDS)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


class Source(Protocol):
    url: str
    title: str
    description: str


@dataclass
class NLPSource(Source):
    url: str
    title: str
    description: str
    co_occurance: object

    def __init__(self, source: Source, /, co_occurance: Graph):
        self.url = source.url
        self.title = source.title
        self.description = source.description
        self.co_occurance = node_link_data(co_occurance)


@dataclass
class TermData:
    term: str
    sources: list[NLPSource]

    def save_json(self, filepath: Path) -> None:
        serialized = json.dumps(asdict(self), indent=2)
        filepath.write_text(serialized, encoding="utf-8")


def save_term_data_json(term: TermData, filepath: Path) -> None:
    serialized = json.dumps(term, indent=2)
    filepath.write_text(serialized, encoding="utf-8")


def filter_chinese_philosophy(sep_url: str) -> bool:
    if is_chinese_philosophy(sep_url):
        print(" " * 4 + "skipping chinese philosophy search result: " + sep_url)
        return False
    return True


with open(DATA / "terms.json", "w") as file:
    serialized = [{"hanzi": t.hanzi, "english": t.english} for t in TERMS]
    file.write(json.dumps(serialized))


for term_pairs in TERMS:
    print(f"\n=== {term_pairs.hanzi} / {', '.join(term_pairs.english)} ===")

    # First run NLP for the Chinese term in the Mengzi
    term = term_pairs.hanzi
    data = TermData(
        term,
        [NLPSource(m, co_occurance=get_cooccurence_chinese(term, m.text))
         for m in mengzi]
    )
    data.save_json(CTEXT / f"{term}_pmi.json")

    # Now iterate for each associated english term
    for term in term_pairs.english:
        print(f"\n  [{term}] Searching SEP...")
        sep_articles = search_sep(
            term, 4, pre_filter=filter_chinese_philosophy)
        print(f"  [{term}] {len(sep_articles)} articles found")

        slugified_search_term = term.replace(" ", "+")
        all_articles = SEPArticle(
            url=f"https://plato.stanford.edu/search/searcher.py?query={slugified_search_term}",
            title="Combined",
            description="Results from all articles",
            text="\n\n".join(a.text for a in sep_articles)
        )
        sep_articles = [all_articles] + sep_articles

        sources: list[NLPSource] = []
        for article in sep_articles:
            source = NLPSource(
                article,
                co_occurance=get_cooccurence_english(term, article.text)
            )
            sources.append(source)

        data = TermData(term, sources)
        data.save_json(SEP / f"{slugify(term)}_pmi.json")
