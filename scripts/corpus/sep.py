from typing import NamedTuple
from bs4 import BeautifulSoup
from collections.abc import Callable
import requests
import time
import re
from corpus import cache
from dataclasses import dataclass
from urllib.parse import urlparse
from random import random
from models import Source, Rendering
cache.install()

_Response = requests.Response


@dataclass
class SEP(Source):
    url: str
    title: str
    description: str
    text: str

    def __init__(self, *, url: str, title: str, description: str | None = None, text: str):
        self.id = get_doc_id(url)
        self.url = url
        self.title = title
        self.description = description or text[:160]
        self.text = text

    @classmethod
    def from_url(cls, url: str):
        if url in _cached_articles:
            return _cached_articles[url]

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.from_cache:  # type: ignore
            time.sleep(random() * 2 + 1)

        soup = BeautifulSoup(r.content, "html.parser", from_encoding="utf-8")

        title = soup.title or soup.find("h1")
        if title:
            title = title.get_text().strip().replace(
                " (Stanford Encyclopedia of Philosophy)", "")
        else:
            title = "UNTITLED"

        article = {
            "preamble": soup.find("div", id="preamble"),
            # "toc": soup.find("div", id="toc"),
            "main-text": soup.find("div", id="main-text"),
        }

        if article["main-text"] is not None:
            for id in ("bibliography", "academic-tools", "other-internet-resources", "related-entries"):
                tag = article["main-text"].find(id)
                if tag:
                    tag.decompose()

        description = soup.description and str(soup.description.text).strip()
        if not description and article["preamble"]:
            description = article["preamble"].text.strip()[:160]

        text = ""
        for tag in article.values():
            if tag is not None:
                text += tag.get_text()  # Just text, strip html
        text = _strip_citations(text).strip()

        sep = cls(
            url=url,
            title=title,
            description=str(description),
            text=text
        )
        _cached_articles[url] = sep
        return sep


_cached_articles: dict[str, SEP] = {}


@dataclass
class SEPSearch(Source):
    id: str
    url: str
    title: str
    description: str
    articles: list[SEP]
    total_articles: int

    @property
    def text(self) -> str:
        return "\n\n".join([a.text.strip() for a in self.articles])

    @classmethod
    def from_term(cls, term: str | Rendering, *, max_results: int | None = None, filter: Callable[[SEP], bool] | None = None, pre_filter: Callable[[str], bool] | None = None):
        sep: SEPSearch | None = None
        page = 1

        search_term: str
        search_string: str
        if isinstance(term, str):
            search_term, search_string = term
        else:
            search_term = term.label
            search_string = ' '.join([stem for stem in term.patterns])

        if search_term in _cached_searches:
            return _cached_searches[search_term]

        while True:
            search = _search(search_string, page)
            results, total_results = _parse_search_results(search)
            if sep is None:
                sep = cls(
                    id=f"{search_term}_combined",
                    url=search.url,
                    title=f"SEP search: {search_term}",
                    description=f"SEP search results {search_string}",
                    articles=[],
                    total_articles=total_results,
                )
            if not results:
                break  # empty search page, break loop
            for url in results:
                if pre_filter and not pre_filter(url):
                    continue
                article = SEP.from_url(url)
                if filter and not filter(article):
                    continue
                sep.articles.append(article)
                if max_results and len(sep.articles) >= max_results:
                    break  # got enough articles, break inner loop
            if max_results and len(sep.articles) >= max_results:
                break  # ...and stop paging (the for-break alone kept looping)
            page += 1

        global _max_len_citation
        print("max length citation:")
        print(_max_len_citation)
        _max_len_citation = ""

        return sep


_cached_searches: dict[str, SEPSearch] = {}


def get_doc_id(sep_url: str) -> str:
    err = ValueError(f"Bad SEP entry url: {sep_url}")
    paths = urlparse(sep_url).path.split("/")
    try:
        e = paths.index("entries")
        doc_id = paths[e + 1]
    except (IndexError, ValueError):
        raise err
    if not doc_id:
        raise err
    return doc_id


def _search(term: str, page: int = 1) -> _Response:
    """Fetches and parses the HTML of a search results page on the SEP. Returns a list of the article links."""
    r = requests.get(
        "https://plato.stanford.edu/search/searcher.py",
        params={"query": term, "page": max(1, page)},
        timeout=30
    )
    if not r.from_cache:  # type: ignore
        time.sleep(random() * 2 + 1)
    r.raise_for_status()
    return r


class _ParsedSearchPage(NamedTuple):
    results: list[str]
    total: int


SEARCH_TOTAL_RE = re.compile(
    r"\d+[-–—]\d+ of (?P<total_docs>\d+) documents found")


def _parse_search_results(r: _Response) -> _ParsedSearchPage:
    soup = BeautifulSoup(r.content, 'html.parser', from_encoding="utf-8")
    result_links = soup.find_all("div", class_="result_url")
    total_results = len(result_links)
    search_total = soup.find("div", class_="search_total")
    if search_total:
        match = SEARCH_TOTAL_RE.fullmatch(search_total.text)
        if match:
            try:
                docs = int(match.group("total_docs"))
                total_results = docs or total_results
            except (TypeError, ValueError):
                ...
        else:
            print(
                f"Couldn't find total documents in SEP search results page: {r.url}")
    return _ParsedSearchPage([r.text.strip() for r in result_links], total_results)


_PARENS_RE = re.compile(r"\((.*?)[\(\)]", re.DOTALL)
_CITATION_RE = re.compile(
    r"[^();.]+\s\d{4}[a-z]?(,\s(\w+\.\s)?[-–—0-9]+)?[^();.]*", re.DOTALL)

_max_len_citation = ""


def _strip_citations(text: str) -> str:
    stripped = text
    citation_spans: list[tuple[int, int]] = []
    for parens in _PARENS_RE.finditer(text):
        inner = parens.group(1)
        global _max_len_citation
        if len(inner) > len(_max_len_citation):
            _max_len_citation = inner
        for match in _CITATION_RE.finditer(inner):
            start = parens.start() + match.start() + 1
            end = parens.start() + match.end() + 1
            match_text = match.group(0)
            if match_text != text[start:end]:
                print("MISMATCH: couldn't strip citations, aborting")
                print("  text:  " + text[start:end].replace("\n", " "))
                print("  match: " + match_text.replace("\n", " "))
                return text
            citation_spans.append((start, end))
    citation_spans.reverse()
    for start, end in citation_spans:
        stripped = stripped[:start] + stripped[end:]
    return stripped


def search_sep(search_term: str | Rendering, max_results: int | None = None, *, filter: Callable[[SEP], bool] | None = None, pre_filter: Callable[[str], bool] | None = None) -> list[SEP]:
    articles = []
    page = 1

    search_string: str
    if isinstance(search_term, str):
        search_string = search_term
    else:
        search_string = ' '.join([stem for stem in search_term.patterns])

    while True:
        results, _ = _parse_search_results(_search(search_string, page))
        if not results:
            # empty search page, break loop
            return articles
        for url in results:
            if pre_filter and not pre_filter(url):
                continue
            article = SEP.from_url(url)
            if filter and not filter(article):
                continue
            articles.append(article)
            if max_results and len(articles) >= max_results:
                # got enough articles, break loop
                return articles
        page += 1


SEP_CORPUS = Source(
    id="sep",
    url="https://plato.stanford.edu/",
    title="SEP (combined)",
    description="Combined SEP corpus for the English renderings of 仁 / 義"
)
