from datetime import timedelta
from pathlib import Path
import requests_cache

# Anchor the cache at the scripts/ root (corpus/..), so it stays put and is
# reused regardless of which package the caller lives in.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _CACHE_DIR.mkdir(exist_ok=True)
    requests_cache.install_cache(
        str(_CACHE_DIR / "http_cache"),
        backend="sqlite",
        expire_after=timedelta(days=7),
    )
    _installed = True
