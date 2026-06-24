"""Central configuration for WalkyTalky.

Holds model identifiers, public-API endpoints, tunable defaults, and a tiny
``.env`` loader so the Ollama cloud API key can live outside the source code.
"""

import os

import ollama

# --- Ollama models (already pulled locally) ---------------------------------
# Language model used for grounded extraction of historical sites.
# qwen2.5:7b-instruct: strong structured/JSON extraction (CPU-bound on this box,
# so per-search latency is minutes — mitigated by the on-disk result cache).
LLM_MODEL = "qwen2.5:7b-instruct"
# Embedding model used to vectorise document chunks.
EMBED_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"

# --- Public HTTP APIs (no key required) -------------------------------------
# OSRM public routing server. Profiles: ``foot``, ``bike``, ``car``.
OSRM_BASE = "https://router.project-osrm.org"
# Nominatim (OpenStreetMap) geocoding. Usage policy: max 1 request/second.
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
# Photon (OpenStreetMap) geocoding — fuzzy/typo-tolerant fallback for names
# Nominatim's stricter matching misses. Same OSM data, no API key required.
PHOTON_BASE = "https://photon.komoot.io"
# Descriptive User-Agent required by the Nominatim usage policy.
USER_AGENT = "WalkyTalky/1.0 (historical-route-finder)"

# --- Tunable defaults -------------------------------------------------------
DEFAULT_RADIUS_M = 1500       # corridor half-width around the route, in metres
CHUNK_WORDS = 220             # words per document chunk
CHUNK_OVERLAP = 40            # overlapping words between consecutive chunks
RETRIEVE_K = 8                # chunks retrieved per query from the vector store
MAX_CONTEXT_CHUNKS = 12       # chunks per LLM extraction call (one batch window)
EXTRACT_POOL_CHUNKS = 48      # total unique chunks pooled across queries, then
                              # processed in MAX_CONTEXT_CHUNKS-sized batches so
                              # the model sees far more of the corpus (recall)
NOMINATIM_MIN_INTERVAL = 1.1  # seconds between Nominatim calls (policy: >=1s)

# --- Network timeouts (seconds) ---------------------------------------------
# Every Ollama call goes through a Client with an explicit timeout so a stalled
# cloud/local endpoint can never hang the pipeline indefinitely.
WEB_TIMEOUT = 30              # web_fetch / web_search (cloud)
EMBED_TIMEOUT = 60           # embeddings (local)
CHAT_TIMEOUT = 600           # language-model generation (7B on CPU is slow)

# --- LLM generation bounds --------------------------------------------------
LLM_NUM_CTX = 8192           # context window (fits the larger chunk budget)
LLM_NUM_PREDICT = 768        # max tokens generated (room for more sites)

# Travel-mode codes embedded in Google Maps ``!3e<n>`` and their OSRM profiles.
MODE_CODES = {"0": "driving", "1": "cycling", "2": "walking", "3": "transit"}
OSRM_PROFILE = {"walking": "foot", "cycling": "bike", "driving": "car"}

# --- Cache / output paths ---------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, ".cache")
GEOCODE_CACHE = os.path.join(CACHE_DIR, "geocode_cache.json")
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")
DEFAULT_OUTPUT = os.path.join(_HERE, "results.json")


def load_env(path=None):
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables are never overwritten. Blank lines and lines
    starting with ``#`` are ignored. This avoids a hard dependency on
    ``python-dotenv``.

    Args:
        path: Path to the ``.env`` file. Defaults to ``.env`` next to this module.

    Returns:
        bool: ``True`` if the file existed and was read, ``False`` otherwise.
    """
    if path is None:
        path = os.path.join(_HERE, ".env")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    return True


def ensure_cache_dir():
    """Create the on-disk cache directory if it does not already exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_client(timeout):
    """Build an Ollama client with an explicit HTTP timeout.

    Routing the calls through a configured client guarantees every Ollama
    request (cloud web search/fetch and local embed/chat) fails fast instead of
    hanging forever when an endpoint stops responding.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        ollama.Client: A client configured with the given timeout.
    """
    return ollama.Client(timeout=timeout)
