"""Geocode place names to coordinates via OpenStreetMap services.

Primary lookups use Nominatim (respecting its usage policy: >=1 request/second,
descriptive User-Agent), trying the city-qualified name, the bare name, and any
caller-supplied local-language aliases (e.g. "Kristiansand Cathedral" ->
"Kristiansand domkirke") that match how OSM labels local features. Those aliases
are produced by :func:`extract.localize_names` (LLM, generalised to any city's
language). When all Nominatim attempts miss, the query falls back to Photon — a
fuzzy, typo-tolerant geocoder over the same OSM data.

Every candidate is validated against the route's bounding box (when provided):
a result that lands far from the route is almost certainly the wrong same-named
place, so it is rejected rather than accepted and later mis-filtered. Results
(hits and misses) are cached on disk to avoid repeat lookups.
"""

import json
import os
import time

import requests

from config import (
    GEOCODE_CACHE,
    NOMINATIM_BASE,
    NOMINATIM_MIN_INTERVAL,
    PHOTON_BASE,
    USER_AGENT,
    ensure_cache_dir,
)

# Timestamp of the last Nominatim call, for rate limiting.
_last_call = [0.0]


def _in_viewbox(coords, viewbox):
    """Report whether ``coords`` falls inside the (padded) route bounding box.

    Args:
        coords: ``(lat, lng)`` to test.
        viewbox: ``(min_lng, min_lat, max_lng, max_lat)`` or ``None``. When
            ``None`` (e.g. geocoding the route endpoints themselves), any
            coordinate is accepted.

    Returns:
        bool: ``True`` if inside the box or no box was given.
    """
    if not viewbox:
        return True
    lat, lng = coords
    min_lng, min_lat, max_lng, max_lat = viewbox
    return (min_lat <= lat <= max_lat) and (min_lng <= lng <= max_lng)


def _load_cache():
    """Load the geocode cache from disk.

    Returns:
        dict: Mapping of cache key -> ``[lat, lng]`` (or ``None`` for misses).
    """
    if os.path.exists(GEOCODE_CACHE):
        try:
            with open(GEOCODE_CACHE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(cache):
    """Persist the geocode cache to disk.

    Args:
        cache: The cache dict to write.
    """
    ensure_cache_dir()
    with open(GEOCODE_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def _rate_limit():
    """Sleep as needed so calls respect the Nominatim minimum interval."""
    elapsed = time.time() - _last_call[0]
    if elapsed < NOMINATIM_MIN_INTERVAL:
        time.sleep(NOMINATIM_MIN_INTERVAL - elapsed)
    _last_call[0] = time.time()


def _nominatim(query, viewbox, timeout):
    """Look up a single query string via Nominatim.

    Args:
        query: The query string.
        viewbox: Optional ``(min_lng, min_lat, max_lng, max_lat)`` bias box.
        timeout: HTTP timeout in seconds.

    Returns:
        tuple[float, float] | None: ``(lat, lng)`` of the best match, or ``None``.
    """
    params = {"q": query, "format": "json", "limit": 1}
    if viewbox:
        params["viewbox"] = ",".join(str(v) for v in viewbox)
        params["bounded"] = 0  # bias, do not hard-restrict
    _rate_limit()
    try:
        resp = requests.get(
            f"{NOMINATIM_BASE}/search",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:  # noqa: BLE001 - treat any failure as a miss
        return None
    return None


def _photon(query, viewbox, timeout):
    """Fuzzy fallback lookup via Photon (typo-tolerant, same OSM data).

    Photon biases toward a single ``lat``/``lon`` point rather than a box, so the
    centre of ``viewbox`` (the route area) is used as the location prior.

    Args:
        query: The query string.
        viewbox: Optional ``(min_lng, min_lat, max_lng, max_lat)`` used to derive
            a centre point for location bias.
        timeout: HTTP timeout in seconds.

    Returns:
        tuple[float, float] | None: ``(lat, lng)`` of the best match, or ``None``.
    """
    params = {"q": query, "limit": 1, "lang": "en"}
    if viewbox:
        min_lng, min_lat, max_lng, max_lat = viewbox
        params["lat"] = (min_lat + max_lat) / 2.0
        params["lon"] = (min_lng + max_lng) / 2.0
    try:
        resp = requests.get(
            f"{PHOTON_BASE}/api",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        feats = resp.json().get("features") or []
        if feats:
            lng, lat = feats[0]["geometry"]["coordinates"]  # GeoJSON: [lng, lat]
            return (float(lat), float(lng))
    except Exception:  # noqa: BLE001 - treat any failure as a miss
        return None
    return None


def geocode(name, city="", viewbox=None, cache=None, timeout=20, aliases=None):
    """Resolve a place name to ``(lat, lng)`` using OSM geocoders.

    Tries Nominatim with the city-qualified query, the bare name, and any
    local-language ``aliases``, then falls back to fuzzy Photon. The first
    in-viewbox hit wins; the combined outcome (hit or miss) is cached under the
    city-qualified key.

    Args:
        name: The place name to geocode.
        city: Optional city/region appended to disambiguate the query.
        viewbox: Optional ``(min_lng, min_lat, max_lng, max_lat)`` bias box; when
            given, results are preferred within (Nominatim) or near (Photon) it
            and out-of-box hits are rejected.
        cache: Optional shared cache dict (loaded via :func:`_load_cache`). When
            ``None``, the on-disk cache is loaded and saved per call.
        timeout: HTTP timeout in seconds.
        aliases: Optional list of local-language name variants to try (from
            :func:`extract.localize_names`), each queried city-qualified and bare.

    Returns:
        tuple[float, float] | None: ``(lat, lng)`` of the best match, or ``None``
        if no result was found.
    """
    own_cache = cache is None
    if own_cache:
        cache = _load_cache()

    base = f"{name}, {city}" if city else name
    key = base.lower().strip()
    if key in cache:
        hit = cache[key]
        return tuple(hit) if hit else None

    # Nominatim query variants: city-qualified, bare, then local-language aliases
    # (city-qualified and bare). Deduplicated, order-preserving.
    candidates = [base, name]
    for alias in (aliases or []):
        candidates.append(f"{alias}, {city}" if city else alias)
        candidates.append(alias)
    seen, queries = set(), []
    for q in candidates:
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    # First in-viewbox hit wins; out-of-box hits are wrong same-named places.
    result = None
    for q in queries:
        cand = _nominatim(q, viewbox, timeout)
        if cand and _in_viewbox(cand, viewbox):
            result = cand
            break
    if result is None:
        # Fuzzy last resort: Photon tolerates the loose phrasings the LLM emits.
        cand = _photon(base, viewbox, timeout)
        if cand and _in_viewbox(cand, viewbox):
            result = cand

    cache[key] = list(result) if result else None
    if own_cache:
        _save_cache(cache)
    return result


def geocode_sites(sites, city="", viewbox=None):
    """Geocode a list of site dicts in place, sharing one cache.

    A site's optional ``aliases`` list (local-language name variants) is used as
    extra Nominatim queries and then removed from the dict.

    Args:
        sites: List of dicts each with at least a ``name`` key, optionally an
            ``aliases`` list.
        city: City/region used to disambiguate every query.
        viewbox: Optional bias box ``(min_lng, min_lat, max_lng, max_lat)``.

    Returns:
        list[dict]: The same dicts, each augmented with ``lat``/``lng`` (``None``
        when geocoding failed).
    """
    cache = _load_cache()
    for site in sites:
        coords = geocode(site["name"], city=city, viewbox=viewbox, cache=cache,
                         aliases=site.get("aliases"))
        site["lat"] = coords[0] if coords else None
        site["lng"] = coords[1] if coords else None
        site.pop("aliases", None)
    _save_cache(cache)
    return sites
