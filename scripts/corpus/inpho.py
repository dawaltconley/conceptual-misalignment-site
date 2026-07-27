import requests
from corpus import cache
import time
import csv
from urllib.parse import urlparse
from random import random
from io import StringIO
cache.install()


# doc ids to automatically exclude
# some SEP articles were published after the hypershelf API was last updated
# so they have to be excluded manually
_EXCLUDE = {
    'han-dynasty',
    'confucianism-modern',
}


def _get_doc_id(sep_url: str) -> str:
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


def _get_topics(doc_id: str) -> str | None:
    r = requests.get(
        f"https://www.hypershelf.org/sep/100/doc_topics/{doc_id}", timeout=30)
    if not r.from_cache:  # type: ignore
        time.sleep(random() * 2 + 1)
    if not r.ok:
        return None
    return r.text


def _get_chinese_philosophy_percentage(topics: str) -> float:
    reader = csv.reader(StringIO(topics))
    for topic, prob in reader:
        if topic == '78':  # chinese philosophy topic on the 100 topic model
            return float(prob)
    return 0


def is_chinese_philosophy(sep_url: str) -> bool:
    doc_id = _get_doc_id(sep_url)
    if doc_id in _EXCLUDE:
        return True
    topics = _get_topics(doc_id)
    if topics:
        percentage = _get_chinese_philosophy_percentage(topics)
        # this filters out the most obvious entries, but also some entries on indian philosophy
        return percentage > 0.25
    return False
