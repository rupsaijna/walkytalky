# WalkyTalky — Change Log

All implementation changes are logged here, newest first. (Requested: "log all changes in a file.")

## 2026-06-24

### Static cached mode for GitHub Pages (+ extraction resilience)
- The web UI is a client+server app; a static host (GitHub Pages) has no backend,
  so live search there returned an HTML 404 that the page tried to parse as JSON
  ("Unexpected token '<'"). Added a backend-free **cached mode**:
  - `index.html` probes `GET /api/health`; with no backend it switches to cached
    mode — shows a banner, disables live-search inputs, and offers a **Cached
    examples** dropdown loaded from a committed manifest. An `asJson()` helper
    turns any non-JSON response into a clear message instead of a parse error.
  - `server.py` gains `GET /api/health` (`{"ok": true, "mode": "full"}`).
  - `export_cache.py` (new) writes `web_cache/` (manifest `index.json` + one file
    per route) from the on-disk result cache, so only actually-computed routes are
    published — an honest snapshot. Exported the two Kristiansand routes.
  - README documents the full-vs-cached modes and how to publish the demo.
  - Verified static serving (no `/api`): health 404 → cached mode; manifest and
    route files load 200.
- Extraction resilience: a single batch that errors (e.g. an Ollama `ReadTimeout`
  on the CPU-bound model — which aborted a full run) is now caught and skipped
  with a warning, so other batches still contribute instead of failing the run.

### Geocoder — LLM-driven local-language name normalization (generalised)
- Replaced the hard-coded English->Norwegian feature-word table with a general
  `extract.localize_names(city, names)`: one LLM call renders every extracted
  name into the LOCAL LANGUAGE of the city (translating just the generic feature
  word, keeping proper nouns), so the approach works for any city/country, not
  only Norway. Best-effort — returns `{}` on failure.
- `run_pipeline` calls it after extraction; localized forms attach to each
  candidate as `aliases` and feed `geocode()` as extra Nominatim queries (tried
  before the Photon fallback). `geocode_sites` consumes and strips `aliases`.
- `geocode.py` no longer hard-codes any language; `_alias_variants`/`re` removed.
- Verified mapping/parsing (stubbed LLM): changed names mapped, unchanged names
  dropped, empty/no-city guarded. Live end-to-end check pending the running run.

### Discovery — add tour/day-trip web searches (recall)
- `ingest.discover_sites` now also queries `"day trips in {city}"` and
  `"hop on hop off tour {city}"` alongside the existing landmark/museum/heritage
  searches. Tour and day-trip listings enumerate a city's notable sights, so
  their snippets enrich the indexed corpus with exactly the named landmarks
  extraction is after (e.g. Christiansholm festning, the cathedral). The snippets
  flow through the existing path (gather → chunk → embed → extract), so they
  become candidate places with no extraction change.
- Diagnostic note: Christiansholm festning *is* in the corpus (10/230 chunks) and
  the LLM extracts it correctly when its chunk is in context — the gap was
  retrieval/pooling depth (only ~26 unique chunks pooled at RETRIEVE_K=8). The
  added searches increase site density; raising RETRIEVE_K remains a further
  lever if needed.

### Geocoder — viewbox validation + Norwegian name aliasing (precision + recall)
- The fuzzy Photon fallback (and bare-name Nominatim) sometimes returned a
  confidently-wrong same-named place far from the route (e.g. "Nordberg Fort"
  ~80 km away; "Kristiansand Cathedral" ~7 km off). Added `_in_viewbox()`
  validation: any geocode landing outside the route's padded bounding box is
  rejected, and the cascade falls through to the next variant instead of
  accepting a wrong hit. `_route_viewbox()` now scales its padding with the
  corridor radius (≥0.05 deg floor) so it never clips a near-route site.
- Added English→Norwegian feature-word aliasing (`_alias_variants`): the LLM
  emits anglicised names OSM only knows in Norwegian, so each English feature
  word (cathedral→domkirke, fortress→festning, square→torg, manor→gård, …) yields
  an extra Nominatim query *before* the Photon fallback. This both fixes recall
  and improves precision (an exact Nominatim hit beats a fuzzy guess).
- Verified: "Kristiansand Cathedral" now resolves correctly on-route via the
  "Domkirke" alias (was lost to Photon's 7 km-off coords); "Nordberg Fort"'s
  80 km-off match is now rejected; other sites unchanged.

### Progress tracking + ETA (CLI and web)
- New `progress.py` — `ProgressTracker` emits structured snapshots (step index,
  percent, elapsed, **ETA**, per-step checklist) as the pipeline advances. ETA
  is scaled by the run's observed pace and seeded from per-step durations
  *learned* across runs (persisted to `.cache/timings.json`, EMA capped at 20
  samples) with CPU-tuned first-run defaults. The same tracker drives the CLI's
  `[i/6]` console output (unchanged) and the web UI.
- `walkytalky.py` — `run`/`run_from_query`/`run_pipeline` take an optional
  `progress`/`tracker`; all step prints now route through the tracker. Geocode
  failures call `tracker.fail()` before raising.
- `server.py` — `/api/search` is now job-based: it validates, starts the
  pipeline on a background thread, and returns `{"job_id"}` (202). New
  `GET /api/progress?job=<id>` returns the latest snapshot plus the final
  `result` (done) or `error`. In-memory job registry, thread-safe, pruned to 40.
- `index.html` — progress panel: animated bar with %, current step + detail,
  elapsed time and "~ETA remaining", and a live step checklist. The client POSTs
  then polls `/api/progress` every 1 s until done/error.
- Verified: tracker snapshots (mid-run % and ETA, all-done states); server
  GET/POST validation (200/400/404); and the full async job path (POST → poll →
  error propagation) via an ungeocodable input.

### Extraction — batched map-reduce over broader retrieval (recall)
- The old extractor showed the LLM only `MAX_CONTEXT_CHUNKS` (12) chunks in a
  single call — ~5% of a 226-chunk corpus — so on-route landmarks present in the
  docs but ranked below the top-12 (e.g. Christiansholm festning) were never
  extracted. Reworked `extract.py`:
  - Expanded `_RETRIEVAL_QUERIES` from 2 generic queries to 5 facet queries
    (fortifications / religious / museums / monuments / old-town) for broader
    chunk coverage.
  - Pool up to `EXTRACT_POOL_CHUNKS` (48) unique chunks, then extract in
    `MAX_CONTEXT_CHUNKS`-sized batches and union+dedupe by name — the model now
    sees most of the corpus across ~4 cheap calls instead of 5% in one.
  - Split out `_extract_from_context()`; `extract_sites()` keeps its signature
    so nothing downstream changes. Added `EXTRACT_POOL_CHUNKS` to `config.py`.
  - Cost: a few extra chat calls per *uncached* run (pay-once via result cache).

### Geocoder — query relaxation + Photon fallback (more matches)
- Geocode misses silently drop LLM-found candidates before the corridor filter,
  so they directly cap match count. Hardened `geocode.py` to raise the hit rate:
  it now tries Nominatim with `"{name}, {city}"`, then the bare `name`, then
  falls back to **Photon** (`photon.komoot.io`) — a fuzzy/typo-tolerant geocoder
  over the same OSM data, biased to the route-area centre. No API key.
- Split `_nominatim()`/`_photon()` helpers; `geocode()` orchestrates the cascade
  and caches the combined outcome under the city-qualified key.
- Added `PHOTON_BASE` to `config.py`.
- Verified: "Gimle Gård Manor" (a previously dropped Nominatim miss) now resolves
  via the cascade; "Christiansholm Fortress" resolves; a nonsense name still
  returns None (no false positives). Routing engine (OSRM) left as-is — it only
  shapes the corridor centreline and is not the match-count bottleneck.
- NOTE: existing `null` entries in `.cache/geocode_cache.json` from the old
  strict logic will short-circuit the new fallback; purge them (and rerun with
  `--refresh`) to benefit on already-cached routes.

### Fix — encoding bug in requests fallback (`ingest.py`)
- `_fetch_with_requests` passed `resp.text` to BeautifulSoup. `requests` defaults
  to ISO-8859-1 for `text/html` responses with no charset in the HTTP header, so
  UTF-8 pages that declare their charset only in a `<meta>` tag were silently
  mangled (e.g. Norwegian ø/å/æ → mojibake). Now passes raw `resp.content` so
  BeautifulSoup sniffs the real encoding. Only the no-key / web_fetch-failed
  fallback path was affected; the Ollama `web_fetch` path was already correct.

## 2026-06-23

### Session start — scaffolding
- Created `.env` with `OLLAMA_API_KEY` (user-supplied) so the app can authenticate to Ollama
  web_search/web_fetch cloud APIs. Persistent `setx` was blocked by the auto-mode classifier,
  so the key lives in a project-local `.env` loaded at runtime by `config.py` instead.
- Added `requirements.txt` (ollama, requests, beautifulsoup4, numpy).
- Added `config.py` — model names, defaults, and a minimal `.env` loader.

### Pipeline modules (RAG along-route historical-site finder)
- `maps_parser.py` — parse Google Maps directions link into source/dest/mode +
  coords; `derive_city` helper. Verified against both example links (walking &
  cycling) from the prompt.
- `routing.py` — OSRM public routing (foot/bike) for the route polyline, with a
  straight-line interpolation fallback.
- `spatial.py` — haversine, point-to-route-polyline distance, corridor filter +
  rank by distance from source.
- `geocode.py` — Nominatim geocoding with 1 req/s rate limiting and on-disk cache.
- `vectorstore.py` — word-window chunking, Ollama `embed` (bge-base-en-v1.5),
  in-memory cosine-similarity `VectorStore` with save/load.
- `ingest.py` — document gathering via Ollama `web_fetch`/`web_search` with a
  `requests`+BeautifulSoup fallback when no API key.
- `extract.py` — RAG: retrieve chunks + Ollama `chat` (Llama-3.2-1B, JSON) to
  extract named historical sites; dedup.
- `walkytalky.py` — CLI orchestrator wiring all stages; prints a table and writes
  `results.json`.
- `README.md` — setup, usage, architecture, limitations.
- `.gitignore` — excludes `.env`, `.cache/`, `results.json`, `__pycache__/`.

### Verification
- Confirmed Ollama `web_search`, `web_fetch`, and `embed` work with the supplied
  key (returned Christiansholm Fortress, etc.; 768-dim embeddings).
- Installed deps (requests, beautifulsoup4; numpy/ollama already present).
- Ran full pipeline end-to-end on the Kristiansand example: 231 chunks indexed,
  3 LLM candidates (Kristiansand Kanonmuseum, Odderøya Museum, Gimle Gård Manor),
  1 within the 1500 m corridor (Odderøya Museum, 775 m from source).

### Fix — pipeline hang (timeouts + bounds)
- First end-to-end run stalled ~23 min idle, blocked on an Ollama network call
  with no timeout. Fix: all Ollama calls now go through `config.get_client(timeout)`
  (`WEB_TIMEOUT`/`EMBED_TIMEOUT`/`CHAT_TIMEOUT`). Updated `ingest.py`,
  `vectorstore.py`, `extract.py`.
- Bounded LLM work: `MAX_CONTEXT_CHUNKS=8`, `RETRIEVE_K=6`, `num_ctx`/`num_predict`
  caps in `config.py` so generation can't crawl on the CPU-bound 1B model.
- `walkytalky.py` now line-buffers stdout for live progress.
- Confirmed the geocode cache works: `.cache/geocode_cache.json` stores hits and
  misses (Gimle Gård Manor cached as null -> never retried).

### Web UI (Leaflet map + backend)
- Refactored `walkytalky.py`: extracted shared `run_pipeline()` core; added
  `run_from_query()` (geocodes free-text source/destination via Nominatim) used
  by the web UI; result now includes `route_points` for map rendering.
- `server.py` — stdlib `ThreadingHTTPServer`; serves `index.html` and
  `POST /api/search` -> runs the pipeline -> returns JSON.
- `index.html` — Leaflet + OpenStreetMap (no API key needed; Google Maps would
  require a billing key). Source/destination/mode/radius/optional URLs inputs;
  draws route polyline, source/dest pins, numbered site pins; hovering a pin
  highlights the matching list row and vice versa; click a row to zoom.
- Verified server serves the page (GET / -> 200) and imports are clean.
- Verified full web path end-to-end via POST /api/search (geocode source/dest ->
  OSRM route -> docs -> embed -> LLM -> geocode -> rank), 197 s.
- Set the UI default destination to "Fiskebrygga, Kristiansand" (Nominatim does
  not recognise "Fish Market"); confirmed clear 400 on un-geocodable input.

### Result cache (avoid recompute on reruns)
- `resultcache.py` — stores each full search result under `.cache/results/<key>.json`,
  keyed by a SHA-256 of the inputs (rounded source/dest coords, mode, city, wiki,
  tourism, radius) **plus the active LLM + embedding model ids**, so a model
  upgrade invalidates stale results automatically.
- `run_pipeline()` checks the cache first and returns instantly on a hit (no
  network/compute); saves the result on a miss. New `refresh` flag (CLI
  `--refresh`, API `"refresh": true`) forces recompute.
- Verified cache key stability + input sensitivity + JSON round-trip.

### Model upgrade (1B -> qwen2.5:7b-instruct)
- Low recall with Llama-3.2-1B (only ~3 candidates; one web run returned 0 within
  corridor). Upgraded `LLM_MODEL` to `qwen2.5:7b-instruct` for stronger structured
  extraction (user choice).
- Relaxed extraction bounds for the stronger model: `RETRIEVE_K` 6->8,
  `MAX_CONTEXT_CHUNKS` 8->12, `LLM_NUM_CTX` 4096->8192, `LLM_NUM_PREDICT` 512->768,
  `CHAT_TIMEOUT` 180->600 (7B on CPU is slower).
- Pulling the model (~4.7 GB); result cache makes the higher per-search latency a
  pay-once cost per unique route.
