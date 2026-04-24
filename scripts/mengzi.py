"""
Download the entire Mengzi from the ctext.org free API and save each
book as a plain-text file under text/ctext/.

Books and their ctext URNs:
  1A  梁惠王上  ctp:mengzi/liang-hui-wang-i
  1B  梁惠王下  ctp:mengzi/liang-hui-wang-ii
  2A  公孫丑上  ctp:mengzi/gongsun-chou-i
  2B  公孫丑下  ctp:mengzi/gongsun-chou-ii
  3A  滕文公上  ctp:mengzi/teng-wen-gong-i
  3B  滕文公下  ctp:mengzi/teng-wen-gong-ii
  4A  離婁上    ctp:mengzi/li-lou-i
  4B  離婁下    ctp:mengzi/li-lou-ii
  5A  萬章上    ctp:mengzi/wan-zhang-i
  5B  萬章下    ctp:mengzi/wan-zhang-ii
  6A  告子上    ctp:mengzi/gaozi-i
  6B  告子下    ctp:mengzi/gaozi-ii
  7A  盡心上    ctp:mengzi/jin-xin-i
  7B  盡心下    ctp:mengzi/jin-xin-ii
"""

import time

import requests

API_BASE = "https://api.ctext.org"

BOOKS = [
    "ctp:mengzi/liang-hui-wang-i",
    "ctp:mengzi/liang-hui-wang-ii",
    "ctp:mengzi/gong-sun-chou-i",
    "ctp:mengzi/gong-sun-chou-ii",
    "ctp:mengzi/teng-wen-gong-i",
    "ctp:mengzi/teng-wen-gong-ii",
    "ctp:mengzi/li-lou-i",
    "ctp:mengzi/li-lou-ii",
    "ctp:mengzi/wan-zhang-i",
    "ctp:mengzi/wan-zhang-ii",
    "ctp:mengzi/gaozi-i",
    "ctp:mengzi/gaozi-ii",
    "ctp:mengzi/jin-xin-i",
    "ctp:mengzi/jin-xin-ii",
]


def _fetch_text(urn: str) -> dict:
    r = requests.get(f"{API_BASE}/gettext", params={"urn": urn}, timeout=30)
    r.raise_for_status()
    return r.json()


class Chapter:
    def __init__(self, urn):
        data = _fetch_text(urn)
        self.title = data.get("title", "")
        paragraphs = data.get("fulltext", [])
        if not paragraphs:
            raise RuntimeError(
                f"No text returned for {urn}. Full response: {data}")
        self.text = "\n".join(p.strip() for p in paragraphs if p.strip())


def fetch_mengzi() -> list[Chapter]:
    full_text = []
    for urn in BOOKS:
        full_text.append(Chapter(urn))
        time.sleep(1)
    return full_text
