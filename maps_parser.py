"""Parse a Google Maps *directions* link into structured waypoints.

A Maps directions URL looks like::

    google.com/maps/dir/<NameA>/<NameB>/@<lat>,<lng>,<zoom>z/data=...!1d<lng>!2d<lat>...!3e<mode>...

This module extracts the human-readable waypoint names from the path, the
precise coordinates from the ``data`` parameter, and the travel mode.
"""

import re
from urllib.parse import unquote

from config import MODE_CODES

# Each waypoint marker in the ``data`` blob carries ``!1d<lng>!2d<lat>``.
_COORD_RE = re.compile(r"!1d(-?\d+\.\d+)!2d(-?\d+\.\d+)")
# Travel mode marker, e.g. ``!3e2`` for walking.
_MODE_RE = re.compile(r"!3e(\d)")


def _clean_name(segment):
    """Turn a URL path segment into a readable place name.

    ``Aquarama,+Tangen+8,+4608+Kristiansand`` -> ``Aquarama, Tangen 8, 4608
    Kristiansand``.

    Args:
        segment: A single ``/``-delimited path segment.

    Returns:
        str: The decoded, space-normalised place name.
    """
    name = unquote(segment).replace("+", " ")
    return re.sub(r"\s+", " ", name).strip()


def parse_maps_url(url):
    """Parse a Google Maps directions URL into source/destination waypoints.

    Args:
        url: A ``maps/dir/...`` Google Maps link (with or without scheme).

    Returns:
        dict: ``{"source": {...}, "dest": {...}, "mode": str, "waypoints": [...]}``
        where each waypoint is ``{"name": str, "lat": float|None, "lng": float|None}``
        and ``mode`` is one of ``walking``/``cycling``/``driving``/``transit``/``unknown``.

    Raises:
        ValueError: If the URL is not a recognisable Maps *directions* link.
    """
    if "/dir/" not in url:
        raise ValueError("Not a Google Maps directions link (missing '/dir/').")

    # Names live between '/dir/' and the '/@' camera segment (or '/data=').
    after_dir = url.split("/dir/", 1)[1]
    path_part = re.split(r"/@|/data=", after_dir, maxsplit=1)[0]
    names = [_clean_name(s) for s in path_part.split("/") if s and not s.startswith("@")]

    # Coordinates: one (lng, lat) pair per waypoint marker, in order.
    coords = [(float(lng), float(lat)) for lng, lat in _COORD_RE.findall(url)]

    # Travel mode.
    mode_match = _MODE_RE.search(url)
    mode = MODE_CODES.get(mode_match.group(1), "unknown") if mode_match else "unknown"

    waypoints = []
    for i, name in enumerate(names):
        lng, lat = (coords[i] if i < len(coords) else (None, None))
        waypoints.append({"name": name, "lat": lat, "lng": lng})

    if len(waypoints) < 2:
        raise ValueError("Could not parse at least two waypoints from the link.")

    return {
        "source": waypoints[0],
        "dest": waypoints[-1],
        "mode": mode,
        "waypoints": waypoints,
    }


def derive_city(waypoint):
    """Best-effort guess of the city/region from a waypoint name.

    Uses the last comma-separated token, stripping a leading postal code, e.g.
    ``Aquarama, Tangen 8, 4608 Kristiansand`` -> ``Kristiansand``.

    Args:
        waypoint: A waypoint dict containing a ``name``.

    Returns:
        str: The inferred city name (may be empty if undeterminable).
    """
    parts = [p.strip() for p in waypoint.get("name", "").split(",") if p.strip()]
    if not parts:
        return ""
    tail = parts[-1]
    # Drop a leading postal code if present ("4608 Kristiansand" -> "Kristiansand").
    return re.sub(r"^\d{3,}\s+", "", tail).strip()
