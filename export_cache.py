"""Export locally-computed results into a static dataset for GitHub Pages.

Reads the on-disk result cache (``.cache/results/*.json``, which is gitignored)
and writes a committed ``web_cache/`` directory the static UI can load with no
backend:

* ``web_cache/index.json`` — manifest listing each cached city/route.
* ``web_cache/<id>.json``  — the full result payload for each entry.

Run after computing the routes you want to publish::

    python export_cache.py

Only the results you have actually computed are exported, so the published demo
is an honest snapshot of the cache. Commit ``web_cache/`` to publish it.
"""

import json
import os
import re

import config

_HERE = os.path.dirname(os.path.abspath(__file__))
WEB_CACHE_DIR = os.path.join(_HERE, "web_cache")


def _slug(text):
    """Make a filesystem/URL-safe slug from arbitrary text."""
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return text or "route"


def export():
    """Write ``web_cache/`` from the on-disk result cache. Returns entry count."""
    src_dir = config.RESULTS_CACHE_DIR
    if not os.path.isdir(src_dir):
        print(f"No result cache at {src_dir}; nothing to export.")
        return 0

    os.makedirs(WEB_CACHE_DIR, exist_ok=True)
    # Clear previously-exported JSON so the published set stays in sync with the
    # current cache (drops routes that are no longer cached).
    for old in os.listdir(WEB_CACHE_DIR):
        if old.endswith(".json"):
            try:
                os.remove(os.path.join(WEB_CACHE_DIR, old))
            except OSError:
                pass
    manifest = []
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(src_dir, fname), "r", encoding="utf-8") as fh:
                result = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or "sites" not in result:
            continue

        short = fname[:8]  # short hash from the cache key, keeps ids unique
        city = result.get("city", "")
        mode = result.get("mode", "walking")
        entry_id = f"{_slug(city)}-{mode}-{short}"
        out_name = f"{entry_id}.json"
        result["cached"] = True
        with open(os.path.join(WEB_CACHE_DIR, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

        src = result.get("source", {})
        dst = result.get("destination", {})
        manifest.append({
            "id": entry_id,
            "file": out_name,
            "city": city,
            "mode": mode,
            "radius_m": result.get("radius_m"),
            "source": src.get("name", ""),
            "destination": dst.get("name", ""),
            "count": result.get("count", len(result.get("sites", []))),
        })

    # Sort by city then source for a tidy dropdown.
    manifest.sort(key=lambda e: (e["city"].lower(), e["source"].lower()))
    with open(os.path.join(WEB_CACHE_DIR, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"routes": manifest}, fh, ensure_ascii=False, indent=2)

    print(f"Exported {len(manifest)} cached route(s) to {WEB_CACHE_DIR}")
    for e in manifest:
        print(f"  - {e['city']}: {e['source']} -> {e['destination']} "
              f"({e['mode']}, {e['count']} sites)")
    return len(manifest)


if __name__ == "__main__":
    export()
