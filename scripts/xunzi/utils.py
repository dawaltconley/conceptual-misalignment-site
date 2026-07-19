from typing import Literal, Any

ARCH = Literal["api", "qwen1", "qwen1.5"]
TOKENIZER = Any


def assert_arch(s: str) -> ARCH:
    arch: list[ARCH] = ["api", "qwen1", "qwen1.5"]
    if s in arch:
        return s
    raise ValueError('Invalid architecture: ' + s)
