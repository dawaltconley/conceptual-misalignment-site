from typing import NamedTuple
from bs4 import BeautifulSoup
from collections.abc import Callable
import requests
import time
import re
from corpus import cache
from dataclasses import dataclass
from random import random
from lib import Source, Rendering
cache.install()

_Response = requests.Response


@dataclass
class SEP(Source):
    url: str
    title: str
    description: str
    text: str

    @classmethod
    def from_url(cls, url: str):
        if url in _cached_articles:
            return _cached_articles[url]

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.from_cache:  # type: ignore
            time.sleep(random() * 2 + 1)

        soup = BeautifulSoup(r.content, "html.parser", from_encoding="utf-8")

        title = soup.title and str(soup.title.text)
        if not title:
            h1 = soup.find("h1")
            title = str(h1.text) if h1 else "UNTITLED"

        article = {
            "preamble": soup.find("div", id="preamble"),
            "toc": soup.find("div", id="toc"),
            "main-text": soup.find("div", id="main-text"),
        }

        description = soup.description and str(soup.description.text)
        if not description:
            description = str(article["preamble"]
                              and article["preamble"].text)[:160]

        sep = cls(
            url=url,
            title=str(title).strip().replace(
                " (Stanford Encyclopedia of Philosophy)", ""),
            description=str(description),
            text=str(article["preamble"]) +
            str(article["toc"]) + str(article["main-text"])
        )
        _cached_articles[url] = sep
        return sep


_cached_articles: dict[str, SEP] = {}


@dataclass
class SEPSearch(Source):
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
                    break  # got enough articles, break loop
            page += 1

        return sep


_cached_searches: dict[str, SEPSearch] = {}


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


class ParsedSearchPage(NamedTuple):
    results: list[str]
    total: int


SEARCH_TOTAL_RE = re.compile(
    r"\d+[-–—]\d+ of (?P<total_docs>\d+) documents found")


def _parse_search_results(r: _Response) -> ParsedSearchPage:
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
    return ParsedSearchPage([r.text.strip() for r in result_links], total_results)


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
