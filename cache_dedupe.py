"""Find (and optionally merge) duplicate entries in the on-disk caches.

The caches accumulate near-duplicates over time:

* **Result cache** (``.cache/results/*.json``) — the *same* route computed under
  slightly different inputs (e.g. a Maps-link run vs. a free-text run whose
  geocoded endpoints differ by a few metres) lands under different keys. These
  are mergeable into one richer result.
* **Sites within a result** — the LLM/geocoder can yield the same place twice
  (same name, or two names at the same coordinates).
* **Geocode cache** (``.cache/geocode_cache.json``) — different name keys that
  resolve to the same coordinates (same physical place). Reported for awareness;
  not merged (extra keys only speed future lookups, they do no harm).

It also flags **empty results** (``count == 0``) — failed/fruitless searches that
add nothing and clutter the published demo; ``--apply`` deletes them.

By default this only *reports* (dry run). Pass ``--apply`` to drop empty results,
merge result-cache duplicates, and de-duplicate sites in place.

Usage::

    python cache_dedupe.py                 # report only
    python cache_dedupe.py --apply         # perform the merges
    python cache_dedupe.py --endpoint-threshold 150 --site-threshold 40
"""

import argparse
import json
import os

import config
from spatial import haversine

# Two routes are "the same" if both endpoints are within this distance (m).
ENDPOINT_DUP_M = 100.0
# Two sites are the same place if within this distance (m) or share a name.
SITE_DUP_M = 30.0
# Two geocode entries are the same place if within this distance (m).
COORD_SAME_M = 5.0


def _norm(name):
    """Normalise a place name for comparison (lowercased, collapsed spaces)."""
    return " ".join(str(name or "").split()).lower()


def _coords(entry):
    """Return ``(lat, lng)`` from a dict with lat/lng, or ``None``."""
    lat, lng = entry.get("lat"), entry.get("lng")
    return (lat, lng) if lat is not None and lng is not None else None


def _site_count(data):
    """Number of sites in a result payload."""
    return data.get("count", len(data.get("sites", [])))


# --------------------------------------------------------------------------- #
# Result cache
# --------------------------------------------------------------------------- #
def load_results():
    """Load every cached result as a list of ``(path, data)`` tuples."""
    out = []
    src_dir = config.RESULTS_CACHE_DIR
    if not os.path.isdir(src_dir):
        return out
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(src_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "sites" in data:
            out.append((path, data))
    return out


def _same_route(a, b, endpoint_thresh):
    """Whether two result payloads describe effectively the same route."""
    if _norm(a.get("city")) != _norm(b.get("city")):
        return False
    if _norm(a.get("mode")) != _norm(b.get("mode")):
        return False
    if round(float(a.get("radius_m", 0))) != round(float(b.get("radius_m", 0))):
        return False
    sa, da = _coords(a.get("source", {})), _coords(a.get("destination", {}))
    sb, db = _coords(b.get("source", {})), _coords(b.get("destination", {}))
    if not all((sa, da, sb, db)):
        return False
    return haversine(sa, sb) <= endpoint_thresh and haversine(da, db) <= endpoint_thresh


def group_duplicate_routes(results, endpoint_thresh):
    """Cluster results into groups describing the same route. Groups of >1 only."""
    groups = []  # each: list of (path, data)
    for path, data in results:
        placed = False
        for group in groups:
            if _same_route(group[0][1], data, endpoint_thresh):
                group.append((path, data))
                placed = True
                break
        if not placed:
            groups.append([(path, data)])
    return [g for g in groups if len(g) > 1]


def dedup_sites(sites, site_thresh, source=None):
    """Merge duplicate sites (same name, or within ``site_thresh`` metres).

    Keeps the variant with the longer description; the smaller detour wins ties.
    If ``source`` is given, ``distance_from_source_m`` is recomputed from it.

    Returns:
        tuple[list, int]: ``(deduped_sites, n_removed)``.
    """
    kept = []
    for site in sites:
        name = _norm(site.get("name"))
        pt = _coords(site)
        match = None
        for k in kept:
            if name and name == _norm(k.get("name")):
                match = k
                break
            if pt and _coords(k) and haversine(pt, _coords(k)) <= site_thresh:
                match = k
                break
        if match is None:
            kept.append(dict(site))
            continue
        # Merge into the existing entry: prefer the richer description / closer.
        if len(str(site.get("description", ""))) > len(str(match.get("description", ""))):
            match["description"] = site.get("description", match.get("description"))
        if site.get("detour_from_route_m") is not None and (
                match.get("detour_from_route_m") is None
                or site["detour_from_route_m"] < match["detour_from_route_m"]):
            match["detour_from_route_m"] = site["detour_from_route_m"]
    n_removed = len(sites) - len(kept)
    if source:
        for k in kept:
            pt = _coords(k)
            if pt:
                k["distance_from_source_m"] = round(haversine(source, pt), 1)
    kept.sort(key=lambda s: s.get("distance_from_source_m", float("inf")))
    return kept, n_removed


def merge_route_group(group, site_thresh):
    """Merge a group of duplicate-route results into one canonical payload.

    The canonical is the member with the most sites (then most route points); its
    route/endpoints are kept and all members' sites are unioned and de-duplicated.

    Returns:
        tuple: ``(canonical_path, merged_data, removed_paths)``.
    """
    canonical_path, canonical = max(
        group, key=lambda pd: (len(pd[1].get("sites", [])),
                               len(pd[1].get("route_points", []))))
    source = _coords(canonical.get("source", {}))
    union = []
    for _, data in group:
        union.extend(data.get("sites", []))
    merged_sites, _ = dedup_sites(union, site_thresh, source=source)
    merged = dict(canonical)
    merged["sites"] = merged_sites
    merged["count"] = len(merged_sites)
    removed = [p for p, _ in group if p != canonical_path]
    return canonical_path, merged, removed


# --------------------------------------------------------------------------- #
# Geocode cache
# --------------------------------------------------------------------------- #
def find_geocode_duplicates(coord_thresh):
    """Group geocode-cache keys that resolve to the same coordinates."""
    try:
        with open(config.GEOCODE_CACHE, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    entries = [(k, tuple(v)) for k, v in cache.items() if isinstance(v, list)]
    groups, used = [], set()
    for i, (ki, vi) in enumerate(entries):
        if ki in used:
            continue
        group = [ki]
        for kj, vj in entries[i + 1:]:
            if kj in used:
                continue
            if haversine(vi, vj) <= coord_thresh:
                group.append(kj)
                used.add(kj)
        if len(group) > 1:
            used.add(ki)
            groups.append((vi, group))
    return groups


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    """Report (and optionally apply) cache de-duplication."""
    parser = argparse.ArgumentParser(description="Find/merge duplicate cache entries.")
    parser.add_argument("--apply", action="store_true",
                        help="Perform merges (default: report only)")
    parser.add_argument("--endpoint-threshold", type=float, default=ENDPOINT_DUP_M,
                        help="Max endpoint distance (m) to treat routes as the same")
    parser.add_argument("--site-threshold", type=float, default=SITE_DUP_M,
                        help="Max distance (m) to treat two sites as the same place")
    args = parser.parse_args(argv)

    all_results = load_results()
    print(f"Result cache: {len(all_results)} file(s) in {config.RESULTS_CACHE_DIR}")

    # 0) Empty results (count == 0): failed/fruitless searches worth dropping.
    empties = [(p, d) for p, d in all_results if _site_count(d) == 0]
    results = [(p, d) for p, d in all_results if _site_count(d) > 0]
    if empties:
        print(f"\n{len(empties)} empty result(s) (0 sites) to drop:")
        for path, data in empties:
            src = data.get("source", {}).get("name", "?")
            dst = data.get("destination", {}).get("name", "?")
            print(f"  {os.path.basename(path)}: {src} -> {dst}")

    # 1) Duplicate routes across files (empties excluded).
    route_groups = group_duplicate_routes(results, args.endpoint_threshold)
    if route_groups:
        print(f"\n{len(route_groups)} duplicate-route group(s):")
        for gi, group in enumerate(route_groups, 1):
            head = group[0][1]
            print(f"  [{gi}] {head.get('city')} / {head.get('mode')} / "
                  f"{round(float(head.get('radius_m', 0)))} m  — {len(group)} files:")
            for path, data in group:
                src = data.get("source", {}).get("name", "?")
                dst = data.get("destination", {}).get("name", "?")
                print(f"        {os.path.basename(path)}: {src} -> {dst} "
                      f"({data.get('count', len(data.get('sites', [])))} sites)")
            canon_path, merged, removed = merge_route_group(group, args.site_threshold)
            print(f"        => merge into {os.path.basename(canon_path)} "
                  f"({merged['count']} unique sites); remove {len(removed)} file(s)")
    else:
        print("\nNo duplicate-route groups found.")

    # 2) Duplicate sites within a single result.
    site_dup_files = []
    for path, data in results:
        _, removed = dedup_sites(data.get("sites", []), args.site_threshold)
        if removed:
            site_dup_files.append((path, removed))
    if site_dup_files:
        print(f"\n{len(site_dup_files)} file(s) with internal duplicate sites:")
        for path, n in site_dup_files:
            print(f"  {os.path.basename(path)}: {n} duplicate site(s)")

    # 3) Geocode-cache duplicates (informational only).
    gc_groups = find_geocode_duplicates(COORD_SAME_M)
    if gc_groups:
        print(f"\nGeocode cache: {len(gc_groups)} group(s) of keys at the same place "
              "(informational — not merged):")
        for coords, keys in gc_groups:
            print(f"  {coords}: {', '.join(keys)}")

    if not args.apply:
        if route_groups or site_dup_files or empties:
            print("\nDry run. Re-run with --apply to drop empties and merge.")
        return 0

    # --- Apply ---
    # Drop empty results first.
    dropped_empty = 0
    for path, _ in empties:
        try:
            os.remove(path)
            dropped_empty += 1
        except OSError:
            pass

    merged_count = removed_count = 0
    handled = set()
    for group in route_groups:
        canonical_path, merged, removed = merge_route_group(group, args.site_threshold)
        with open(canonical_path, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        handled.add(canonical_path)
        for path in removed:
            try:
                os.remove(path)
                removed_count += 1
            except OSError:
                pass
        merged_count += 1
    # De-dup sites within files not already rewritten by a merge.
    for path, data in results:
        if path in handled or not os.path.exists(path):
            continue
        deduped, removed = dedup_sites(data.get("sites", []),
                                       args.site_threshold,
                                       source=_coords(data.get("source", {})))
        if removed:
            data["sites"] = deduped
            data["count"] = len(deduped)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

    print(f"\nApplied: dropped {dropped_empty} empty result(s), merged "
          f"{merged_count} route group(s), removed {removed_count} redundant file(s).")
    print("Re-run `python export_cache.py` (or just commit — the pre-commit hook "
          "does it) to refresh the published web_cache/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
