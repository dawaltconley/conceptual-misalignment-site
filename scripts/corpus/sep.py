from bs4 import BeautifulSoup
from collections.abc import Callable
import requests
import time
from corpus import cache
from dataclasses import dataclass
from random import random
from lib import Rendering
cache.install()


@dataclass
class SEP:
    url: str
    title: str
    description: str
    text: str

    @classmethod
    def from_url(cls, url: str):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.from_cache:  # type: ignore
            time.sleep(random() * 2 + 1)

        soup = BeautifulSoup(r.text, "html.parser")

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
            description = str(article["preamble"])[:160]

        return cls(
            url=url,
            title=str(title).strip().replace(
                " (Stanford Encyclopedia of Philosophy)", ""),
            description=str(description),
            text=str(article["preamble"]) +
            str(article["toc"]) + str(article["main-text"])
        )


def _search(term: str, page: int = 1) -> str:
    """Returns the HTML of a search results page on the SEP for a given search term."""
    r = requests.get(
        "https://plato.stanford.edu/search/searcher.py",
        params={"query": term, "page": max(1, page)},
        timeout=30
    )
    if not r.from_cache:  # type: ignore
        time.sleep(random() * 2 + 1)
    r.raise_for_status()
    return r.text


def _parse_search_results(search_page: str) -> list[str]:
    """Parses the HTML from a search results page on SEP. Returns a list of the article links on the first page."""
    soup = BeautifulSoup(search_page, 'html.parser')
    results = soup.find_all("div", class_="result_url")
    return [r.text.strip() for r in results]


def search_sep(search_term: str | Rendering, max_results: int | None = None, *, filter: Callable[[SEP], bool] | None = None, pre_filter: Callable[[str], bool] | None = None) -> list[SEP]:
    articles = []
    page = 1

    search_string: str
    if isinstance(search_term, str):
        search_string = search_term
    else:
        search_string = ' '.join([stem for stem in search_term.patterns])

    while True:
        results = _parse_search_results(_search(search_string, page))
        if not results:
            # empty search page, break loop
            return articles
        for url in results:
            if pre_filter and not pre_filter(url):
                continue
            article = SEP.from_url(url)
            if filter and not filter(article):
                continue
            articles.append(SEP.from_url(url))
            if max_results and len(articles) >= max_results:
                # got enough articles, break loop
                return articles
        page += 1
