import re

# Split after 。！？；, keeping the delimiter, and pulling any trailing closing
# quotes/brackets back onto the sentence they close.
_SENT_RE = re.compile(r"[^。！？；]*[。！？；]+[」』）】》\)]*")


def split_sentences(text: str) -> list[str]:
    """Split a classical-Chinese text into sentences, preserving all characters."""
    text = text.replace("\r", "")
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pos = 0
        for m in _SENT_RE.finditer(line):
            s = m.group().strip()
            if s:
                sentences.append(s)
            pos = m.end()
        # trailing text with no terminal punctuation
        tail = line[pos:].strip()
        if tail:
            sentences.append(tail)
    return sentences
