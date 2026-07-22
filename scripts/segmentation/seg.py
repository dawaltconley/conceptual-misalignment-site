import os
import json
from segmentation.utils import ARCH

SEG_DATA = os.path.dirname(os.path.realpath(__file__)) + '/seg_data.json'

# FEW-SHOT EXAMPLES --- Drawn from the XunziALLM sample data: https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM/blob/a10d6e9ad4e03c8c0cc4370e9d76e04cfa3aa011/sample%20data/sample%20data%20for%20downstream%20tasks%20evaluation/seg_downstream.json
FEWSHOT: list[tuple[str, str]] = []
with open(SEG_DATA) as seg_data:
    data = json.load(seg_data)
    for shot in data:
        FEWSHOT.append((shot['input'], shot['output']))


SYSTEM_PROMPT = """
    You are an assistant to help in the processing of Classical Chinese texts.
    Please perform word segmentation on the input Classical Chinese sentences,
    using the "/" character to delimit word boundaries. Output only the segmented
    results; do not include explanations, numbering, or extraneous text. Ensure
    the characters in the segmented output match the original sentence exactly
    (no added, omitted, or altered characters).
"""


def parse(output: str):
    """
    Parse a 'word/word/word ...' string into [word, word, ...].
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
    return output.split("/") or None


def get_strict_prompt(arch: ARCH) -> str:
    """Get the message used in the stricter retry."""
    if arch == "api":
        return 'Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.'
    if arch == "qwen1":
        return '(Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.)\n'
    if arch == "qwen1.5":
        return 'Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.'
