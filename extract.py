"""RAG extraction of historical sites from the indexed corpus.

Retrieves the most relevant chunks for a history-focused query and asks the
Ollama language model to emit a JSON list of named historical locations,
grounded in that retrieved context.
"""

import json
import re

from config import (
    CHAT_TIMEOUT,
    EXTRACT_POOL_CHUNKS,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    MAX_CONTEXT_CHUNKS,
    RETRIEVE_K,
    get_client,
)

# Facet-specific retrieval queries. Each targets a distinct kind of historical
# place so the pooled chunks cover the corpus broadly instead of clustering
# around one generic query — the main lever on extraction recall.
_RETRIEVAL_QUERIES = [
    "fortress, fort, castle, citadel, defensive works, ramparts, fortification",
    "church, cathedral, chapel, monastery, historic religious building",
    "museum, gallery, heritage centre, historic house",
    "monument, statue, memorial, landmark, commemorative site",
    "old town, historic district, heritage buildings, town square, market square",
]

_LOCALIZE_SYSTEM = (
    "You normalise place names to the local language used on maps. You are "
    "precise and never invent or rename places."
)


def localize_names(city, names):
    """Map each place name to its likely local-language form for geocoding.

    Anglicised names the LLM emits (e.g. "Kristiansand Cathedral") often resolve
    only under the local-language label OSM actually uses ("Kristiansand
    domkirke"). A single LLM call renders every candidate name into the local
    language of ``city`` — translating just the generic feature word and keeping
    proper nouns — which generalises the previous hard-coded English->Norwegian
    table to any city/language.

    Args:
        city: City/region whose local language to target.
        names: List of place names to localise.

    Returns:
        dict: ``{original_name: localized_name}`` for names whose local form
        differs from the original. Empty on any failure (best-effort).
    """
    names = [n for n in names if n]
    if not names or not city:
        return {}
    listing = "\n".join(f"- {n}" for n in names)
    prompt = (
        f"The places below are in or near {city}. For each, give the name exactly "
        f"as it would most likely be labelled on a map in the LOCAL LANGUAGE of "
        f"{city}. Translate only the generic feature word (cathedral, fortress, "
        f"square, church, castle, manor, museum, etc.); keep all proper nouns "
        f"unchanged. If a name is already in the local language or has no generic "
        f"word, repeat it unchanged.\n\n"
        "Respond with ONLY a JSON object of the form:\n"
        '{"names": [{"original": "<as given>", "local": "<local-language form>"}]}\n\n'
        f"{listing}"
    )
    try:
        resp = get_client(CHAT_TIMEOUT).chat(
            model=LLM_MODEL,
            format="json",
            options={"temperature": 0.0, "num_ctx": LLM_NUM_CTX,
                     "num_predict": LLM_NUM_PREDICT},
            messages=[
                {"role": "system", "content": _LOCALIZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    except Exception:  # noqa: BLE001 - localisation is best-effort
        return {}

    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    items = data.get("names", []) if isinstance(data, dict) else []

    requested = {n.lower(): n for n in names}
    mapping = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        original = str(it.get("original", "")).strip()
        local = str(it.get("local", "")).strip()
        key = requested.get(original.lower())
        if key and local and local.lower() != key.lower():
            mapping[key] = local
    return mapping


_SYSTEM = (
    "You extract real, named places of historical interest from the provided "
    "context. Only use names that appear in or are clearly supported by the "
    "context. Do not invent places."
)


def _build_prompt(city, context):
    """Build the user prompt instructing the model to return JSON.

    Args:
        city: City/region name, used to anchor the request.
        context: Concatenated retrieved chunks.

    Returns:
        str: The prompt text.
    """
    return (
        f"From the CONTEXT below about {city or 'the area'}, list specific named "
        "locations of HISTORICAL interest (e.g. fortresses, churches, museums, "
        "monuments, historic districts, statues).\n\n"
        "Respond with ONLY a JSON object of the form:\n"
        '{"sites": [{"name": "<place name>", "description": "<one short sentence>"}]}\n'
        "Use the exact place names. Do not include hotels, restaurants, or shops "
        "unless they are historic landmarks.\n\n"
        f"CONTEXT:\n{context}"
    )


def _parse_json_sites(raw):
    """Extract the ``sites`` list from a model response string.

    Tolerates extra prose around the JSON by locating the first ``{``..``}`` block.

    Args:
        raw: The raw model output.

    Returns:
        list[dict]: Parsed site dicts (possibly empty).
    """
    candidates = []
    try:
        candidates.append(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                candidates.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass
    for obj in candidates:
        if isinstance(obj, dict) and isinstance(obj.get("sites"), list):
            return obj["sites"]
        if isinstance(obj, list):
            return obj
    return []


def _extract_from_context(city, context):
    """Run one grounded extraction LLM call over a single batch of context.

    Args:
        city: City/region name used to anchor the prompt.
        context: Concatenated chunk text for this batch.

    Returns:
        list[dict]: Raw site dicts parsed from the model response (may be empty).
    """
    resp = get_client(CHAT_TIMEOUT).chat(
        model=LLM_MODEL,
        format="json",
        options={
            "temperature": 0.0,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
        },
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_prompt(city, context)},
        ],
    )
    raw = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    return _parse_json_sites(raw)


def extract_sites(store, city, k=RETRIEVE_K):
    """Retrieve relevant context and extract named historical sites via the LLM.

    Pools a broad set of unique chunks across several facet queries, then runs
    extraction in ``MAX_CONTEXT_CHUNKS``-sized batches so the model sees far more
    of the corpus than a single capped call would — the results are unioned and
    deduplicated by name. This raises recall (more on-route landmarks surfaced)
    at the cost of a few extra chat calls per uncached run.

    Args:
        store: A populated :class:`vectorstore.VectorStore`.
        city: City/region name used to anchor the prompt.
        k: Number of chunks to retrieve per query.

    Returns:
        list[dict]: Deduplicated ``{"name", "description"}`` site dicts.
    """
    # Gather and de-duplicate retrieved chunks across the facet queries, ranked
    # by best score, then pool a broad set for batched extraction.
    seen_chunks = {}
    for query in _RETRIEVAL_QUERIES:
        for chunk, score in store.search(query, k=k):
            if chunk not in seen_chunks or score > seen_chunks[chunk]:
                seen_chunks[chunk] = score
    pooled = sorted(seen_chunks, key=seen_chunks.get, reverse=True)[:EXTRACT_POOL_CHUNKS]
    if not pooled:
        return []

    # Extract from each batch and union, deduping by lowercased name. First
    # occurrence wins (it carries the highest-ranked context for that name).
    deduped = {}
    for start in range(0, len(pooled), MAX_CONTEXT_CHUNKS):
        batch = pooled[start:start + MAX_CONTEXT_CHUNKS]
        context = "\n\n".join(batch)
        for s in _extract_from_context(city, context):
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key not in deduped:
                deduped[key] = {
                    "name": name,
                    "description": str(s.get("description", "")).strip(),
                }
    return list(deduped.values())
