"""Route geometry via the public OSRM server, with a straight-line fallback.

Given source and destination coordinates and a travel mode, produce the list of
``(lat, lng)`` points describing the route. This polyline defines the corridor
within which historical sites are searched.
"""

import requests

from config import OSRM_BASE, OSRM_PROFILE, USER_AGENT


def _interpolate(src, dst, n=25):
    """Linearly interpolate ``n`` points between two coordinates.

    Used as a fallback when OSRM is unreachable; good enough to define a
    straight-line corridor.

    Args:
        src: ``(lat, lng)`` start.
        dst: ``(lat, lng)`` end.
        n: Number of points to generate (inclusive of both ends).

    Returns:
        list[tuple[float, float]]: Interpolated ``(lat, lng)`` points.
    """
    slat, slng = src
    dlat, dlng = dst
    return [
        (slat + (dlat - slat) * i / (n - 1), slng + (dlng - slng) * i / (n - 1))
        for i in range(n)
    ]


def get_route(src, dst, mode, timeout=20):
    """Fetch the route polyline between two points for the given travel mode.

    Args:
        src: ``(lat, lng)`` of the source.
        dst: ``(lat, lng)`` of the destination.
        mode: Travel mode string (``walking``/``cycling``/``driving``); anything
            else falls back to the ``foot`` profile.
        timeout: HTTP timeout in seconds.

    Returns:
        tuple[list[tuple[float, float]], dict]: The route points as
        ``(lat, lng)`` and metadata ``{"source": str, "distance_m": float|None,
        "duration_s": float|None}``. ``source`` is ``"osrm"`` or
        ``"straight_line"``.
    """
    profile = OSRM_PROFILE.get(mode, "foot")
    # OSRM expects lng,lat order.
    coords = f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
    url = f"{OSRM_BASE}/route/v1/{profile}/{coords}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        resp = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        resp.raise_for_status()
        route = resp.json()["routes"][0]
        # GeoJSON coordinates are [lng, lat]; convert to (lat, lng).
        points = [(lat, lng) for lng, lat in route["geometry"]["coordinates"]]
        meta = {
            "source": "osrm",
            "distance_m": route.get("distance"),
            "duration_s": route.get("duration"),
        }
        return points, meta
    except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
        points = _interpolate(src, dst)
        return points, {"source": "straight_line", "error": str(exc),
                        "distance_m": None, "duration_s": None}
