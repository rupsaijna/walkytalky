"""WalkyTalky — find historical sites along a walking/cycling route.

Given a Google Maps directions link plus a Wikipedia page and an official
tourism page, this tool discovers nearby locations of historical interest using
a RAG pipeline (Ollama embeddings + language model + web search), geocodes them,
filters to a corridor along the route, and reports them ordered by distance from
the source.

Usage::

    python walkytalky.py --maps "<maps_dir_link>" \\
        --wiki "https://en.wikipedia.org/wiki/Kristiansand" \\
        --tourism "https://en.visitsorlandet.com/destinations/kristiansand/" \\
        --radius 1500 --out results.json
"""

import argparse
import json
import sys

import config
import extract
import geocode
import ingest
import maps_parser
import progress as progress_mod
import resultcache
import routing
import spatial
from vectorstore import VectorStore, chunk_text


def _line_buffer_stdout():
    """Make stdout line-buffered so progress prints appear immediately.

    No-op when stdout does not support reconfiguration (e.g. already wrapped).
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def _route_viewbox(route, radius_m=0.0):
    """Compute a route bounding box, padded to comfortably cover the corridor.

    Used both to bias geocoding toward the route and to validate results (a hit
    outside this box is rejected). The padding scales with the corridor radius so
    it never clips a legitimately near-route site, with a ~5 km floor.

    Args:
        route: List of ``(lat, lng)`` route points.
        radius_m: Corridor half-width in metres (drives the padding).

    Returns:
        tuple | None: ``(min_lng, min_lat, max_lng, max_lat)`` or ``None`` if the
        route is empty.
    """
    if not route:
        return None
    # ~111 km per degree; pad by twice the radius (so the box clears the
    # corridor) but at least 0.05 deg (~5.5 km) for small radii.
    pad = max(0.05, (radius_m / 111000.0) * 2.0)
    lats = [p[0] for p in route]
    lngs = [p[1] for p in route]
    return (min(lngs) - pad, min(lats) - pad, max(lngs) + pad, max(lats) + pad)


def _print_table(sites):
    """Print the ranked sites as an aligned console table.

    Args:
        sites: Ranked list of site dicts with ``name``,
            ``distance_from_source_m``, ``detour_from_route_m``, ``description``.
    """
    if not sites:
        print("\nNo historical sites found within the route corridor.")
        return
    print(f"\n{'#':<3} {'Site':<34} {'From src':>9} {'Detour':>8}  Description")
    print("-" * 100)
    for i, s in enumerate(sites, 1):
        name = (s["name"][:33]) if len(s["name"]) > 33 else s["name"]
        desc = s.get("description", "")
        desc = (desc[:40] + "...") if len(desc) > 43 else desc
        print(f"{i:<3} {name:<34} {s['distance_from_source_m']:>8.0f}m "
              f"{s['detour_from_route_m']:>7.0f}m  {desc}")


def run(maps_url, wiki_url, tourism_url, radius_m, out_path, city_override=None,
        refresh=False, progress=None):
    """Execute the full WalkyTalky pipeline.

    Args:
        maps_url: Google Maps directions link.
        wiki_url: Wikipedia page URL (may be empty).
        tourism_url: Official tourism page URL (may be empty).
        radius_m: Corridor half-width in metres.
        out_path: Path to write the JSON results.
        city_override: Optional explicit city/region (overrides auto-detection).
        refresh: When ``True``, ignore any cached result and recompute.
        progress: Optional callback receiving progress snapshot dicts.

    Returns:
        dict: The full result payload that was written to ``out_path``.
    """
    config.load_env()
    _line_buffer_stdout()

    tracker = progress_mod.ProgressTracker(
        on_event=progress, label_overrides={"input": "Parsing Google Maps link"})

    # 1. Parse the Maps link.
    tracker.start("input")
    parsed = maps_parser.parse_maps_url(maps_url)
    src = parsed["source"]
    dst = parsed["dest"]
    city = city_override or maps_parser.derive_city(src)
    tracker.note(f"{src['name']}  ->  {dst['name']}  ({parsed['mode']}, city: {city})")
    return run_pipeline(src, dst, parsed["mode"], city, wiki_url, tourism_url,
                        radius_m, out_path, refresh=refresh, tracker=tracker)


def run_from_query(source, destination, mode="walking", wiki_url="", tourism_url="",
                   radius_m=config.DEFAULT_RADIUS_M, out_path=None, city_override=None,
                   refresh=False, progress=None):
    """Run the pipeline from free-text source/destination names (web UI entry).

    The source and destination strings are geocoded via Nominatim to obtain
    coordinates, after which the standard pipeline runs.

    Args:
        source: Free-text source place name/address.
        destination: Free-text destination place name/address.
        mode: Travel mode (``walking``/``cycling``/``driving``).
        wiki_url: Optional Wikipedia page URL.
        tourism_url: Optional official tourism page URL.
        radius_m: Corridor half-width in metres.
        out_path: Optional path to write the JSON result.
        city_override: Optional explicit city/region.
        refresh: When ``True``, ignore any cached result and recompute.
        progress: Optional callback receiving progress snapshot dicts.

    Returns:
        dict: The full result payload.

    Raises:
        ValueError: If the source or destination cannot be geocoded.
    """
    config.load_env()
    _line_buffer_stdout()
    tracker = progress_mod.ProgressTracker(
        on_event=progress, label_overrides={"input": "Geocoding source & destination"})
    tracker.start("input")
    s = geocode.geocode(source)
    d = geocode.geocode(destination)
    if not s:
        tracker.fail(f"Could not geocode source: {source!r}")
        raise ValueError(f"Could not geocode source: {source!r}")
    if not d:
        tracker.fail(f"Could not geocode destination: {destination!r}")
        raise ValueError(f"Could not geocode destination: {destination!r}")
    src = {"name": source, "lat": s[0], "lng": s[1]}
    dst = {"name": destination, "lat": d[0], "lng": d[1]}
    # Derive the city robustly: reverse-geocode the source coordinate (immune to
    # how the source was phrased), then fall back to name-based parsing of the
    # source or destination, and finally the raw source string.
    city = (city_override
            or geocode.reverse_city(s[0], s[1])
            or maps_parser.derive_city({"name": source})
            or maps_parser.derive_city({"name": destination})
            or source)
    tracker.note(f"{source}  ->  {destination}  ({mode}, city: {city})")
    return run_pipeline(src, dst, mode, city, wiki_url, tourism_url, radius_m,
                        out_path, refresh=refresh, tracker=tracker)


def run_pipeline(src, dst, mode, city, wiki_url, tourism_url, radius_m,
                 out_path=None, refresh=False, tracker=None):
    """Run stages 2-6 of the pipeline on already-resolved waypoints.

    Shared core used by both :func:`run` (Maps link) and :func:`run_from_query`
    (free-text). Steps: route geometry, document gathering, embedding/indexing,
    LLM extraction, geocoding, corridor filtering and ranking.

    Results are cached on disk by input (see :mod:`resultcache`); an identical
    rerun returns the cached payload instantly unless ``refresh`` is set.

    Args:
        src: Source waypoint dict ``{"name", "lat", "lng"}``.
        dst: Destination waypoint dict ``{"name", "lat", "lng"}``.
        mode: Travel mode (``walking``/``cycling``/``driving``).
        city: City/region used for discovery, geocoding and prompts.
        wiki_url: Wikipedia page URL (may be empty).
        tourism_url: Official tourism page URL (may be empty).
        radius_m: Corridor half-width in metres.
        out_path: Optional path to write the JSON result.
        refresh: When ``True``, ignore any cached result and recompute.
        tracker: Optional :class:`progress.ProgressTracker`; one is created if not
            supplied (e.g. when this core is invoked directly rather than via the
            entry points).

    Returns:
        dict: Result payload with ranked ``sites`` and the ``route_points``
        polyline (list of ``[lat, lng]``) for map rendering. A served-from-cache
        payload additionally carries ``"cached": True``.
    """
    if tracker is None:
        tracker = progress_mod.ProgressTracker()

    # 0. Result cache: skip the whole expensive pipeline on a repeat search.
    cache_key = resultcache.make_key(src, dst, mode, city, wiki_url, tourism_url,
                                     radius_m)
    if not refresh:
        cached = resultcache.load(cache_key)
        if cached is not None:
            print(f"[cache] hit -> returning saved result ({cached.get('count', 0)} "
                  "site(s)); no compute used.")
            cached["cached"] = True
            tracker.done(f"served from cache — {cached.get('count', 0)} site(s)")
            _print_table(cached.get("sites", []))
            return cached

    src_pt = (src["lat"], src["lng"])
    dst_pt = (dst["lat"], dst["lng"])

    # 2. Route geometry.
    tracker.start("route", label="Fetching route geometry")
    route, route_meta = routing.get_route(src_pt, dst_pt, mode)
    tracker.note(f"route via {route_meta['source']}, {len(route)} points")

    # 3. Gather documents.
    tracker.start("docs", label="Gathering documents (Wikipedia, tourism, web search)")
    docs = ingest.gather_documents(wiki_url, tourism_url, city)
    if not docs["had_key"]:
        tracker.note("NOTE: OLLAMA_API_KEY not set -> web-search discovery skipped, "
                     "page fetch via requests fallback.")
    tracker.note(f"collected {len(docs['texts'])} document(s)")

    # 4. Build the vector store.
    tracker.start("embed", label="Embedding & indexing")
    store = VectorStore()
    for text in docs["texts"]:
        store.add(chunk_text(text))
    tracker.note(f"indexed {len(store.chunks)} chunks")

    # 5. RAG extraction of candidate sites.
    tracker.start("extract", label="Extracting historical sites with the language model")
    candidates = extract.extract_sites(store, city)
    tracker.note(f"{len(candidates)} candidate site(s)")

    # 6. Geocode + spatial filter + rank. Localise names to the city's local
    # language first so anglicised names still resolve on OSM (one LLM call).
    tracker.start("filter", label="Geocoding & filtering to the route corridor")
    local_map = extract.localize_names(city, [c["name"] for c in candidates])
    for cand in candidates:
        localized = local_map.get(cand["name"])
        if localized:
            cand["aliases"] = [localized]
    viewbox = _route_viewbox(route, radius_m)
    geocode.geocode_sites(candidates, city=city, viewbox=viewbox)
    ranked = spatial.filter_and_rank(candidates, route, src_pt, radius_m)

    result = {
        "source": src,
        "destination": dst,
        "mode": mode,
        "city": city,
        "radius_m": radius_m,
        "route": {"points": len(route), **route_meta},
        "route_points": [[round(lat, 6), round(lng, 6)] for lat, lng in route],
        "count": len(ranked),
        "sites": ranked,
        "cached": False,
    }
    # Persist to the result cache so an identical rerun is free.
    resultcache.save(cache_key, result)
    tracker.done(f"{len(ranked)} site(s) within the corridor")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON written to: {out_path}")

    _print_table(ranked)
    return result


def main(argv=None):
    """Parse command-line arguments and run the pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        int: Process exit code (0 on success, 1 on argument/parse errors).
    """
    parser = argparse.ArgumentParser(description="Find historical sites along a route.")
    parser.add_argument("--maps", required=True, help="Google Maps directions link")
    parser.add_argument("--wiki", default="", help="Wikipedia page URL")
    parser.add_argument("--tourism", default="", help="Official tourism page URL")
    parser.add_argument("--radius", type=float, default=config.DEFAULT_RADIUS_M,
                        help="Corridor half-width in metres (default: %(default)s)")
    parser.add_argument("--out", default=config.DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--city", default=None, help="Override auto-detected city")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the result cache and recompute")
    args = parser.parse_args(argv)

    try:
        run(args.maps, args.wiki, args.tourism, args.radius, args.out, args.city,
            refresh=args.refresh)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
