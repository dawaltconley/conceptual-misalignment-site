import time
import requests
from bs4 import BeautifulSoup


class SEP:
    def __init__(self, url: str):
        self.url = url
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        self.title = soup.title and soup.title.text
        if not self.title:
            h1 = soup.find("h1")
            self.title = h1.text if h1 else "UNTITLED"
        article = {
            "preamble": soup.find("div", id="preamble"),
            "toc": soup.find("div", id="toc"),
            "main-text": soup.find("div", id="main-text"),
        }
        self.text = str(article["preamble"]) + \
            str(article["toc"]) + str(article["main-text"])


def _search(term: str) -> str:
    """Returns the HTML of a search results page on the SEP for a given search term."""
    r = requests.get(
        "https://plato.stanford.edu/search/searcher.py",
        params={"query": term},
        timeout=30
    )
    r.raise_for_status()
    return r.text


def _parse_search_results(search_page: str) -> list[str]:
    """Parses the HTML from a search results page on SEP. Returns a list of the article links on the first page."""
    soup = BeautifulSoup(search_page, 'html.parser')
    results = soup.find_all("div", class_="result_url")
    return [r.text.strip() for r in results]


def search_sep(search_term: str, max_results: int | None = None) -> list[SEP]:
    articles = []
    results = _parse_search_results(_search(search_term))
    if max_results:
        results = results[:max_results]
    for url in results:
        articles.append(SEP(url))
        time.sleep(1)
    return articles
