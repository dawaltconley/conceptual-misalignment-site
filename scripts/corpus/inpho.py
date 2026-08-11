import requests
from corpus import cache
from corpus.sep import get_doc_id
import time
import csv
from random import random
from io import StringIO
cache.install()


# doc ids to automatically exclude
# some SEP articles were published after the hypershelf API was last updated
# so they have to be excluded manually
_EXCLUDE = {
    'han-dynasty',
    'confucianism-modern',
    'chinese-mind',
    'emotions-chinese',
    'confucian-gender',
    'korean-confucianism',
    'dai-zhen',
}


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
    for line in reader:
        if len(line) < 2:
            continue
        topic, prob = line
        if topic == '78':  # chinese philosophy topic on the 100 topic model
            return float(prob)
    return 0


def _is_mostly_chinese_philosophy(topics: str) -> bool:
    main_topic: tuple[int, float] = (0, 0)
    reader = csv.reader(StringIO(topics))
    for line in reader:
        if len(line) < 2:
            continue
        topic = int(line[0])
        prob = float(line[1])
        # chinese philosophy topic on the 100 topic model
        if topic == 78 and prob < main_topic[1]:
            return False  # can exit early
        if main_topic[1] < prob:
            main_topic = (topic, prob)
    return main_topic[0] == 78


def is_chinese_philosophy(sep_url: str, threshold: float | None = None) -> bool:
    """Checks whether the dominant topic for an SEP article is Chinese philosophy. If threshold is provided, it instead checks whether the article's topics include a percentage of Chinese philosophy above the threshold."""
    doc_id = get_doc_id(sep_url)
    if doc_id in _EXCLUDE:
        return True
    topics = _get_topics(doc_id)
    if topics:
        if threshold is not None:
            percentage = _get_chinese_philosophy_percentage(topics)
            return percentage > threshold
        else:
            return _is_mostly_chinese_philosophy(topics)
    return False
