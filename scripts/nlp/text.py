"""Small text/character utilities shared across the pipelines."""


def is_cjk(s: str) -> bool:
    """Return True if s is non-empty and consists entirely of CJK characters."""
    return bool(s) and all("一" <= c <= "鿿" for c in s)
