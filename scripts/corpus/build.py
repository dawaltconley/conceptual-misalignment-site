from typing import NamedTuple
from config import TERMS
from models import Rendering
from corpus.mengzi import Mengzi, fetch_mengzi_full
from corpus.sep import SEP, SEPSearch
from corpus.inpho import is_chinese_philosophy
from corpus.parse import parse_sep_article, unparse_sep_article


def build_chinese_corpus() -> Mengzi:
    print("Fetching the Mengzi...")
    return fetch_mengzi_full()


class SEPTermSearch(NamedTuple):
    term: Rendering
    search: SEPSearch


def build_english_corpus(max_per_term: int = 4, *, max_chinese_topic: float | None = None, min_freq: int = 1) -> list[SEPTermSearch]:
    results: list[SEPTermSearch] = []
    terms: list[Rendering] = [t for pairs in TERMS for t in pairs.renderings]

    def is_not_chinese_philosophy(sep_url: str) -> bool:
        # this filters out the most obvious entries, but also some entries on indian philosophy
        if is_chinese_philosophy(sep_url, max_chinese_topic):
            print(" " * 4 + "skipping article, chinese philosophy: " + sep_url)
            return False
        return True

    for term in terms:
        def has_min_occurrences(article: SEP) -> bool:
            if min_freq < 1:
                return True
            doc = parse_sep_article(article)
            count = 0
            for token in doc:
                if term.matches(token.lemma_.lower(), token.pos_):
                    count += 1
                if count >= min_freq:
                    return True
            unparse_sep_article(article)
            print(
                " " * 4 + f"skipping article, target below minimum frequency ({min_freq}): " + article.url)
            return False

        print(f"\n  [{term.label}] Searching SEP...")
        search = SEPSearch.from_term(
            term,
            max_results=max_per_term,
            pre_filter=is_not_chinese_philosophy,
            filter=has_min_occurrences,
            capture_excluded=True,
        )
        print(f"  [{term.label}] {search.total_articles} articles found")
        results.append(SEPTermSearch(term, search))

    return results
