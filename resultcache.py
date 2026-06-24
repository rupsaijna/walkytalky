"""On-disk cache for full pipeline results.

Running the pipeline is expensive (web fetch, embedding of hundreds of chunks,
language-model extraction, geocoding). This cache stores the final result of a
search keyed by its inputs, so an identical rerun returns instantly without
using any compute or network.

Results are stored as individual JSON files under ``.cache/results/`` named by a
hash of the normalised inputs.
"""

import hashlib
import json
import os

from config import EMBED_MODEL, LLM_MODEL, RESULTS_CACHE_DIR


def make_key(src, dst, mode, city, wiki_url, tourism_url, radius_m):
    """Build a stable cache key from the inputs that determine the result.

    The active language and embedding models are folded into the key, so
    upgrading either model naturally invalidates stale cached results.
    Coordinates are rounded so insignificant float noise does not cause misses.

    Args:
        src: Source waypoint dict ``{"name", "lat", "lng"}``.
        dst: Destination waypoint dict ``{"name", "lat", "lng"}``.
        mode: Travel mode string.
        city: City/region string.
        wiki_url: Wikipedia URL (may be empty).
        tourism_url: Tourism URL (may be empty).
        radius_m: Corridor radius in metres.

    Returns:
        str: A hex SHA-256 digest uniquely identifying the search.
    """
    payload = {
        "src": [round(float(src["lat"]), 5), round(float(src["lng"]), 5)],
        "dst": [round(float(dst["lat"]), 5), round(float(dst["lng"]), 5)],
        "mode": (mode or "").lower(),
        "city": (city or "").lower(),
        "wiki": (wiki_url or "").strip().lower(),
        "tourism": (tourism_url or "").strip().lower(),
        "radius": round(float(radius_m)),
        "llm": LLM_MODEL,
        "embed": EMBED_MODEL,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _path(key):
    """Return the file path for a given cache key.

    Args:
        key: Cache key from :func:`make_key`.

    Returns:
        str: Absolute path to the cache file.
    """
    return os.path.join(RESULTS_CACHE_DIR, f"{key}.json")


def load(key):
    """Load a cached result by key.

    Args:
        key: Cache key from :func:`make_key`.

    Returns:
        dict | None: The cached result payload, or ``None`` on a miss or read
        error.
    """
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def save(key, result):
    """Persist a result under the given key.

    Args:
        key: Cache key from :func:`make_key`.
        result: The result payload to store.
    """
    os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)
    with open(_path(key), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
