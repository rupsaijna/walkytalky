"""Gather source documents for the RAG corpus.

Primary path uses Ollama's cloud ``web_fetch``/``web_search`` (requires
``OLLAMA_API_KEY``). When the key is absent or a call fails, page fetching falls
back to ``requests`` + BeautifulSoup, and web-search *discovery* is skipped.
"""

import os

import requests
from bs4 import BeautifulSoup

from config import USER_AGENT, WEB_TIMEOUT, get_client


def _have_key():
    """Report whether an Ollama cloud API key is configured.

    Returns:
        bool: ``True`` if ``OLLAMA_API_KEY`` is set in the environment.
    """
    return bool(os.environ.get("OLLAMA_API_KEY"))


def _fetch_with_requests(url, timeout=25):
    """Fetch and strip a web page to plain text using requests + BeautifulSoup.

    Args:
        url: The page URL.
        timeout: HTTP timeout in seconds.

    Returns:
        str: Extracted visible text (empty string on failure).
    """
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        # Pass raw bytes (not resp.text): requests defaults to ISO-8859-1 for
        # text/html with no charset header, mangling UTF-8 pages that declare
        # their charset only in a <meta> tag. BeautifulSoup sniffs the real
        # encoding from the bytes/meta/BOM.
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:  # noqa: BLE001
        return ""


def fetch_page(url):
    """Fetch a page's text, preferring Ollama ``web_fetch``.

    Args:
        url: The page URL.

    Returns:
        dict: ``{"url": str, "title": str, "text": str, "links": list[str]}``.
    """
    if _have_key():
        try:
            res = get_client(WEB_TIMEOUT).web_fetch(url)
            return {
                "url": url,
                "title": getattr(res, "title", "") or "",
                "text": getattr(res, "content", "") or "",
                "links": list(getattr(res, "links", []) or []),
            }
        except Exception:  # noqa: BLE001 - fall through to requests
            pass
    return {"url": url, "title": "", "text": _fetch_with_requests(url), "links": []}


def discover_sites(city, max_results=5):
    """Discover extra historical-site text via Ollama web search.

    Args:
        city: City/region to search around.
        max_results: Results per query.

    Returns:
        list[str]: Text snippets from search results (empty if no API key).
    """
    if not _have_key() or not city:
        return []
    # Tour/day-trip listings are dense with the city's notable sights (the exact
    # named landmarks extraction is after), so they meaningfully boost recall.
    queries = [
        f"historical landmarks in {city}",
        f"museums and monuments in {city}",
        f"history and heritage sites {city}",
        f"day trips in {city}",
        f"hop on hop off tour {city}",
    ]
    client = get_client(WEB_TIMEOUT)
    snippets = []
    for q in queries:
        try:
            resp = client.web_search(q, max_results=max_results)
            results = resp.results if hasattr(resp, "results") else resp
            for r in results:
                title = getattr(r, "title", "") or ""
                content = getattr(r, "content", "") or ""
                snippets.append(f"{title}\n{content}".strip())
        except Exception:  # noqa: BLE001
            continue
    return snippets


def gather_documents(wiki_url, tourism_url, city):
    """Assemble the full text corpus for indexing.

    Combines the supplied Wikipedia and tourism pages with Ollama web-search
    discovery snippets.

    Args:
        wiki_url: Wikipedia page URL (may be empty).
        tourism_url: Tourism page URL (may be empty).
        city: City/region used for web-search discovery.

    Returns:
        dict: ``{"texts": list[str], "sources": list[str], "had_key": bool}``
        where ``sources`` parallels nothing but records provenance for logging.
    """
    texts = []
    sources = []
    for url in (wiki_url, tourism_url):
        if not url:
            continue
        page = fetch_page(url)
        if page["text"]:
            texts.append(page["text"])
            sources.append(page["url"])
    for i, snippet in enumerate(discover_sites(city)):
        if snippet:
            texts.append(snippet)
            sources.append(f"web_search#{i}")
    return {"texts": texts, "sources": sources, "had_key": _have_key()}
