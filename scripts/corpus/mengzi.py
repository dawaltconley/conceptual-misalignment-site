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

import requests
import time
from lib import Source
from corpus import cache
from config import MENGZI_DIR
from typing import Literal
from random import random
cache.install()

_CHAPTER_ID = Literal["1A", "1B", "2A", "2B", "3A",
                      "3B", "4A", "4B", "5A", "5B", "6A", "6B", "7A", "7B"]

_ID_TITLE_DICT: dict[str, str] = {
    '1A':  '梁惠王上',
    '1B':  '梁惠王下',
    '2A':  '公孫丑上',
    '2B':  '公孫丑下',
    '3A':  '滕文公上',
    '3B':  '滕文公下',
    '4A':  '離婁上',
    '4B':  '離婁下',
    '5A':  '萬章上',
    '5B':  '萬章下',
    '6A':  '告子上',
    '6B':  '告子下',
    '7A':  '盡心上',
    '7B':  '盡心下',
}

for id, title in list(_ID_TITLE_DICT.items()):
    _ID_TITLE_DICT[title] = id

API_BASE = "https://api.ctext.org"

# _BOOK_URNS = [
#     "ctp:mengzi/liang-hui-wang-i",
#     "ctp:mengzi/liang-hui-wang-ii",
#     "ctp:mengzi/gong-sun-chou-i",
#     "ctp:mengzi/gong-sun-chou-ii",
#     "ctp:mengzi/teng-wen-gong-i",
#     "ctp:mengzi/teng-wen-gong-ii",
#     "ctp:mengzi/li-lou-i",
#     "ctp:mengzi/li-lou-ii",
#     "ctp:mengzi/wan-zhang-i",
#     "ctp:mengzi/wan-zhang-ii",
#     "ctp:mengzi/gaozi-i",
#     "ctp:mengzi/gaozi-ii",
#     "ctp:mengzi/jin-xin-i",
#     "ctp:mengzi/jin-xin-ii",
# ]


def _fetch_text(urn: str) -> dict:
    r = requests.get(f"{API_BASE}/gettext", params={"urn": urn}, timeout=30)
    r.raise_for_status()
    if not r.from_cache:  # type: ignore
        time.sleep(random() * 2 + 1)
    return r.json()


def _fetch_text_local(id: _CHAPTER_ID) -> str:
    with open(MENGZI_DIR / f"{id}.txt", "r") as file:
        return file.read()


def _fetch_text_url(urn: str) -> str:
    print(f"fetching url: {urn}")
    r = requests.get(f"{API_BASE}/getlink", params={"urn": urn}, timeout=30)
    r.raise_for_status()
    if not r.from_cache:  # type: ignore
        print("not cached, staggering fetches")
        time.sleep(random() * 2 + 1)
    return r.json().get("url", "")


class Chapter(Source):
    id: str
    urn: str
    url: str
    title: str
    description: str
    text: str

    def __init__(self, *, urn: str, title: str, description: str | None = None, text: str):
        self.id = _ID_TITLE_DICT[title]
        self.urn = urn
        self.url = _fetch_text_url(urn)
        self.title = title
        self.description = description or text[:160]
        self.text = text

    @classmethod
    def from_urn(cls, urn):
        data = _fetch_text(urn)
        title = data.get("title", "")
        paragraphs: list[str] = data.get("fulltext", [])
        subsections: list[str] = data.get("subsections", [])
        if len(paragraphs) > 0:
            return cls(
                urn=urn,
                title=title,
                description=paragraphs[0][:160],
                text="\n".join(p.strip() for p in paragraphs if p.strip())
            )
        elif len(subsections) > 0:
            chapters = [Chapter.from_urn(s) for s in subsections]
            return cls(
                urn=urn,
                title=title,
                description=chapters[0].description,
                text="\n".join(c.text.strip()
                               for c in chapters if c.text.strip())
            )
        else:
            raise RuntimeError(
                f"No text returned for {urn}. Full response: {data}")


class Mengzi(Chapter):
    chapters = [
        Chapter(
            title='梁惠王上',
            urn='ctp:mengzi/liang-hui-wang-i',
            text=_fetch_text_local('1A'),
        ),
        Chapter(
            title='梁惠王下',
            urn='ctp:mengzi/liang-hui-wang-ii',
            text=_fetch_text_local('1B'),
        ),
        Chapter(
            title='公孫丑上',
            urn='ctp:mengzi/gongsun-chou-i',
            text=_fetch_text_local('2A'),
        ),
        Chapter(
            title='公孫丑下',
            urn='ctp:mengzi/gongsun-chou-ii',
            text=_fetch_text_local('2B'),
        ),
        Chapter(
            title='滕文公上',
            urn='ctp:mengzi/teng-wen-gong-i',
            text=_fetch_text_local('3A'),
        ),
        Chapter(
            title='滕文公下',
            urn='ctp:mengzi/teng-wen-gong-ii',
            text=_fetch_text_local('3B'),
        ),
        Chapter(
            title='離婁上',
            urn='ctp:mengzi/li-lou-i',
            text=_fetch_text_local('4A'),
        ),
        Chapter(
            title='離婁下',
            urn='ctp:mengzi/li-lou-ii',
            text=_fetch_text_local('4B'),
        ),
        Chapter(
            title='萬章上',
            urn='ctp:mengzi/wan-zhang-i',
            text=_fetch_text_local('5A'),
        ),
        Chapter(
            title='萬章下',
            urn='ctp:mengzi/wan-zhang-ii',
            text=_fetch_text_local('5B'),
        ),
        Chapter(
            title='告子上',
            urn='ctp:mengzi/gaozi-i',
            text=_fetch_text_local('6A'),
        ),
        Chapter(
            title='告子下',
            urn='ctp:mengzi/gaozi-ii',
            text=_fetch_text_local('6B'),
        ),
        Chapter(
            title='盡心上',
            urn='ctp:mengzi/jin-xin-i',
            text=_fetch_text_local('7A'),
        ),
        Chapter(
            title='盡心下',
            urn='ctp:mengzi/jin-xin-ii',
            text=_fetch_text_local('7B'),
        ),
    ]

    def __init__(self):
        self.id = 'mengzi'
        self.urn = 'ctp:mengzi'
        self.url = _fetch_text_url(self.urn)
        self.title = '孟子'
        self.description = 'Full text of the Mengzi'
        self.text = '\n'.join(
            [c.text.strip() for c in self.chapters if c.text.strip()]
        )

    def get_chapter(self, id: _CHAPTER_ID) -> Chapter:
        title = _ID_TITLE_DICT[id]
        for c in self.chapters:
            if c.title == title:
                return c
        raise ValueError('Bad chapter ID, none found: ' + id)


_mengzi = None


def fetch_mengzi_full() -> Mengzi:
    global _mengzi
    if _mengzi is None:
        _mengzi = Mengzi()
    return _mengzi


def fetch_mengzi_chapters() -> list[Chapter]:
    return fetch_mengzi_full().chapters
