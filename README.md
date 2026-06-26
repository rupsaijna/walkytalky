# WalkyTalky

Find **locations of historical interest along a walking or cycling route**.

Given a Google Maps directions link plus a Wikipedia page and an official tourism
page, WalkyTalky runs a small RAG pipeline — **Ollama embeddings + language model
+ web search** — to discover nearby historical sites, geocodes them, filters them
to a corridor along your route, and lists them **ordered by distance from the
source**.

## How it works

```
Maps link ──> parse source/dest/mode ──> OSRM route polyline
Wikipedia + tourism pages ─┐
Ollama web search          ├─> chunk ─> embed (bge) ─> vector store
                           ┘                              │
                                   retrieve + LLM (Llama) ─> candidate sites
                                          │
            Nominatim geocode ──> corridor filter ──> rank by distance from source
                                          │
                              console table + results.json
```

| Stage | Module | Tech |
|-------|--------|------|
| Parse Maps link | `maps_parser.py` | regex |
| Route geometry | `routing.py` | OSRM public API (foot/bike) |
| Fetch & discover docs | `ingest.py` | Ollama `web_fetch`/`web_search` (requests fallback) |
| Embed & index | `vectorstore.py` | Ollama `embed` (bge-base-en-v1.5) + NumPy |
| Extract sites | `extract.py` | Ollama `chat` (qwen2.5:7b-instruct, JSON) |
| Geocode | `geocode.py` | Nominatim (OpenStreetMap) |
| Filter & rank | `spatial.py` | haversine + point-to-polyline |
| Orchestrate / CLI | `walkytalky.py` | argparse |

## Prerequisites

1. **Ollama** running locally (`ollama serve`) with both models pulled:
   ```
   ollama pull qwen2.5:7b-instruct
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ```
   (The language model is configurable via `LLM_MODEL` in `config.py`; a smaller
   model such as `llama3.2:3b-instruct` is faster on CPU but extracts fewer sites.)
2. **Ollama cloud API key** for web search/fetch — put it in `walkytalky/.env`:
   ```
   OLLAMA_API_KEY=your_key_here
   ```
   Get a free key at <https://ollama.com>. Without it, the supplied Wikipedia /
   tourism pages are fetched via a `requests` fallback and web-search discovery
   is skipped.
3. Python deps:
   ```
   pip install -r requirements.txt
   ```

## Usage

```bash
python walkytalky.py \
  --maps "google.com/maps/dir/Aquarama,+Tangen+8,+4608+Kristiansand/Fish+Market,+Gravane+6,+4610+Kristiansand/@58.14,7.99,16z/data=...!3e2" \
  --wiki "https://en.wikipedia.org/wiki/Kristiansand" \
  --tourism "https://en.visitsorlandet.com/destinations/kristiansand/" \
  --radius 1500 \
  --out results.json
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--maps` | (required) | Google Maps **directions** link |
| `--wiki` | "" | Wikipedia page URL |
| `--tourism` | "" | Official tourism page URL |
| `--radius` | 1500 | Corridor half-width in metres |
| `--out` | `results.json` | Output JSON path |
| `--city` | auto | Override the auto-detected city |
| `--refresh` | off | Ignore the result cache and recompute |

Output is a console table **and** a JSON file. Each site includes its name,
description, `distance_from_source_m`, and `detour_from_route_m`.

## Web UI

```
python server.py        # then open http://127.0.0.1:8000
```

Enter a source and destination (free text — geocoded via Nominatim), pick mode
and radius, optionally add Wikipedia/tourism URLs, and click **Search**. A
**live progress panel** shows the current pipeline step, a percent bar, elapsed
time and an **estimated time to finish** (the ETA self-calibrates from per-step
timings learned across runs). The map then draws the route, source/destination
pins, and numbered pins for each historical site; **hovering a pin highlights
the matching list row and vice versa**, and clicking a row zooms to it.

Under the hood the UI is job-based: `POST /api/search` starts the pipeline on a
background thread and returns a `job_id`; the page polls
`GET /api/progress?job=<id>` once a second for the snapshot and final result.

### Two deployment modes

The same `index.html` runs in two modes and detects which one it's in (it probes
`GET /api/health`):

| | **Full (Python server)** | **Cached (static host, e.g. GitHub Pages)** |
|---|---|---|
| How | `python server.py` → `localhost:8000` | static files only — no backend |
| Live free-text search | ✅ runs the full RAG pipeline | ❌ disabled (shown as a banner) |
| Cached cities | ✅ available | ✅ the only option |
| Needs Ollama | yes | no |

**The full pipeline cannot run on GitHub Pages** — it needs the Python backend
and a local Ollama (plus OSRM/Nominatim/web-search calls), none of which a static
host provides. So the published static site is a **cached demo**: it serves only
precomputed results and says so. Pick a city from the **Cached examples**
dropdown to view its route and sites.

#### Publishing the cached demo

1. Compute the routes you want to publish (CLI or web UI) so they land in the
   result cache.
2. Export them to the committed static dataset:
   ```
   python export_cache.py        # writes web_cache/ (index.json + one file per route)
   ```
3. Commit `web_cache/` and enable **GitHub Pages → Deploy from branch → main / root**.

Only the routes you actually computed are exported, so the demo is an honest
snapshot. To add more cities, compute them and re-run `export_cache.py`.

#### Keeping the published cache in sync automatically

A tracked **pre-commit hook** (`hooks/pre-commit`) regenerates `web_cache/` from
your local result cache and stages it on every commit, so a new run's results
reach GitHub the next time you push — no manual export. Enable it once per clone:

```
git config core.hooksPath hooks
```

The hook never blocks a commit; if `export_cache.py` can't run it just warns and
proceeds. To publish a cache-only update with no other changes:

```
git commit -m "update cached results" --allow-empty   # hook stages web_cache/
git push
``` The map uses Leaflet + OpenStreetMap (no API key); to
use Google Maps instead, swap the tile layer for the Google Maps JavaScript API
(requires a billing-enabled key).

## Caching (no wasted recompute)

- **Geocoding** is cached in `.cache/geocode_cache.json` (hits *and* misses).
- **Full search results** are cached in `.cache/results/` keyed by the inputs and
  the active model ids — an identical rerun returns instantly with no network or
  compute. Pass `--refresh` (CLI) or `"refresh": true` (API) to force a recompute.
  Changing `LLM_MODEL`/`EMBED_MODEL` invalidates cached results automatically.

## Notes & limitations

- Extraction quality is the main quality lever and scales with `LLM_MODEL`. The
  default `qwen2.5:7b-instruct` gives good recall but, on a CPU-only machine, an
  uncached search takes a few minutes (the result cache makes reruns instant).
  Drop to `llama3.2:3b-instruct` for speed at the cost of recall.
- OSRM and Nominatim are free, rate-limited public services (geocoding is paced
  at ~1 request/second and cached under `.cache/`).
- The travel mode is read from the Maps link (`!3e2`=walking, `!3e1`=cycling).
- `.env` holds a secret — it is excluded from version control via `.gitignore`.
