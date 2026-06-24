"""Geospatial helpers: distances, corridor filtering, and ranking.

All distances are in metres. Coordinates are ``(lat, lng)`` tuples in degrees.
"""

import math

EARTH_RADIUS_M = 6371000.0


def haversine(a, b):
    """Great-circle distance between two ``(lat, lng)`` points, in metres.

    Args:
        a: First ``(lat, lng)`` point.
        b: Second ``(lat, lng)`` point.

    Returns:
        float: Distance in metres.
    """
    lat1, lng1 = math.radians(a[0]), math.radians(a[1])
    lat2, lng2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _segment_distance(pt, seg_a, seg_b):
    """Approximate distance from a point to a line segment, in metres.

    Uses a local equirectangular projection (accurate over the short distances
    relevant to a single walking/cycling route).

    Args:
        pt: The ``(lat, lng)`` point to measure from.
        seg_a: Segment start ``(lat, lng)``.
        seg_b: Segment end ``(lat, lng)``.

    Returns:
        float: Distance in metres from ``pt`` to the nearest point on the segment.
    """
    lat0 = math.radians(pt[0])

    def project(p):
        x = math.radians(p[1]) * math.cos(lat0) * EARTH_RADIUS_M
        y = math.radians(p[0]) * EARTH_RADIUS_M
        return x, y

    px, py = project(pt)
    ax, ay = project(seg_a)
    bx, by = project(seg_b)
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - ax, py - ay)
    # Projection factor of pt onto the segment, clamped to [0, 1].
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def distance_to_route(pt, route):
    """Minimum distance from a point to a route polyline, in metres.

    Args:
        pt: The ``(lat, lng)`` point.
        route: List of ``(lat, lng)`` points describing the route.

    Returns:
        float: Distance in metres to the nearest point on the route. Returns
        ``inf`` for an empty route.
    """
    if not route:
        return float("inf")
    if len(route) == 1:
        return haversine(pt, route[0])
    return min(
        _segment_distance(pt, route[i], route[i + 1])
        for i in range(len(route) - 1)
    )


def filter_and_rank(sites, route, source, radius_m):
    """Keep sites inside the route corridor and rank them by distance from source.

    Args:
        sites: Iterable of dicts each containing ``lat`` and ``lng`` keys (and any
            other metadata such as ``name``/``description``).
        route: List of ``(lat, lng)`` route points.
        source: ``(lat, lng)`` of the journey source.
        radius_m: Corridor half-width in metres; sites farther than this from the
            route are dropped.

    Returns:
        list[dict]: Surviving sites, each augmented with
        ``distance_from_source_m`` and ``detour_from_route_m`` and sorted in
        ascending order of ``distance_from_source_m``.
    """
    kept = []
    for site in sites:
        if site.get("lat") is None or site.get("lng") is None:
            continue
        pt = (site["lat"], site["lng"])
        detour = distance_to_route(pt, route)
        if detour <= radius_m:
            enriched = dict(site)
            enriched["detour_from_route_m"] = round(detour, 1)
            enriched["distance_from_source_m"] = round(haversine(source, pt), 1)
            kept.append(enriched)
    kept.sort(key=lambda s: s["distance_from_source_m"])
    return kept
