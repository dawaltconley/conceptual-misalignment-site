from typing import NamedTuple
from config import TERMS
from models import Rendering
from corpus.mengzi import Mengzi, fetch_mengzi_full
from corpus.sep import SEPSearch
from corpus.inpho import is_chinese_philosophy


def build_chinese_corpus() -> Mengzi:
    print("Fetching the Mengzi...")
    return fetch_mengzi_full()


class SEPTermSearch(NamedTuple):
    term: Rendering
    search: SEPSearch


def build_english_corpus(max_per_term=4) -> list[SEPTermSearch]:
    results: list[SEPTermSearch] = []
    terms: list[Rendering] = [t for pairs in TERMS for t in pairs.renderings]

    for term in terms:
        print(f"\n  [{term.label}] Searching SEP...")
        search = SEPSearch.from_term(
            term,
            max_results=max_per_term,
            pre_filter=is_not_chinese_philosophy
        )
        print(f"  [{term.label}] {search.total_articles} articles found")
        results.append(SEPTermSearch(term, search))

    return results


def is_not_chinese_philosophy(sep_url: str) -> bool:
    # this filters out the most obvious entries, but also some entries on indian philosophy
    if is_chinese_philosophy(sep_url, 0.25):
        print(" " * 4 + "skipping chinese philosophy search result: " + sep_url)
        return False
    return True
