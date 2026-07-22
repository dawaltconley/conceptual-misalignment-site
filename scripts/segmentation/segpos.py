from segmentation.utils import ARCH

# --------------------------------------------------------------------------- #
# POS tagset. A compact, classical-Chinese-appropriate set. ADJUST to match the
# reference standard you will validate against; the distinction of modal/final
# particles (y) matters a great deal for 文言文.
# --------------------------------------------------------------------------- #
TAGSET = {
    "n":  "名词 noun",
    "nr": "人名 person name",
    "ns": "地名 place name",
    "t":  "时间词 time word",
    "v":  "动词 verb",
    "a":  "形容词 adjective",
    "d":  "副词 adverb",
    "r":  "代词 pronoun",
    "m":  "数词 numeral",
    "q":  "量词 classifier",
    "p":  "介词 preposition",
    "c":  "连词 conjunction",
    "u":  "助词 structural particle (之/所/者 ...)",
    "y":  "语气词 modal/final particle (也/矣/乎/哉 ...)",
    "w":  "标点 punctuation",
}
VALID_TAGS = set(TAGSET)
# --------------------------------------------------------------------------- #
# FEW-SHOT EXAMPLES  ---  PLACEHOLDERS. Replace with your own gold-verified
# annotations. These define the output format ("word/TAG word/TAG ...") and the
# tagset by demonstration. Two or three short, function-word-rich sentences from
# the Mengzi work best. The segmentation/tags below are a plausible first pass
# and are NOT authoritative -- you are the philologist; make them match your
# standard before trusting bulk output.
# --------------------------------------------------------------------------- #
FEWSHOT = [
    (
        "孟子見梁惠王。",
        "孟子/nr 見/v 梁惠王/nr 。/w",
    ),
    (
        "王何必曰利？",
        "王/n 何/r 必/d 曰/v 利/n ？/w",
    ),
    (
        "亦有仁義而已矣。",
        "亦/d 有/v 仁義/n 而已/y 矣/y 。/w",
    ),
]


SYSTEM_PROMPT = (
    "你是古汉语文本处理助手。请对输入的古文句子进行分词和词性标注。"
    "输出格式为“词/词性 词/词性 …”，词与词之间用一个空格分隔，"
    "每个词后用斜杠加词性标记。词性标记集为："
    + "，".join(f"{k}={v}" for k, v in TAGSET.items())
    + "。只输出标注结果，不要输出任何解释、编号或多余文字，"
    "保证标注结果中的字与原句完全一致（不增字、不减字、不改字）。"
)


def parse(output: str):
    """
    Parse a 'word/TAG word/TAG ...' string into [(word, tag), ...].
    Tolerates stray whitespace/newlines. Splits each token on its LAST slash so
    that a word containing '/' would still parse (rare in this domain).
    Returns None if any token is malformed.
    """
    output = output.strip()
    # If the model wrapped the answer in extra lines, keep the line that looks
    # most like a tagged sequence (contains '/').
    candidate_lines = [ln for ln in output.splitlines() if "/" in ln]
    if candidate_lines:
        output = max(candidate_lines, key=len)
    tokens: list[tuple[str, str]] = []
    for chunk in output.split():
        if "/" not in chunk:
            return None
        word, _, tag = chunk.rpartition("/")
        if not word or not tag:
            return None
        tokens.append((word, tag))
    return tokens or None


def get_strict_prompt(arch: ARCH) -> str:
    """Get the message used in the stricter retry."""
    if arch == "api":
        return "请严格只输出“词/词性”序列，字数须与原句完全相同。"
    if arch == "qwen1":
        return "（严格只输出“词/词性”序列，字数须与原句完全相同）\n"
    if arch == "qwen1.5":
        return "请严格只输出“词/词性”序列，字数须与原句完全相同。"
