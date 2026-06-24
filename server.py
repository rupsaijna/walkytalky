"""Minimal HTTP server backing the WalkyTalky web UI.

Serves ``index.html`` and exposes a small job-based search API so the browser can
show live progress:

* ``POST /api/search`` — validates the request, starts the pipeline in a
  background thread, and immediately returns ``{"job_id": ...}``.
* ``GET /api/progress?job=<id>`` — returns the latest progress snapshot for that
  job (step, percent, elapsed, ETA), plus the final ``result`` once done or an
  ``error`` on failure.

Uses only the Python standard library so no extra web framework is required.

Run::

    python server.py            # then open http://localhost:8000
"""

import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import config
import walkytalky

_HERE = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
PORT = 8000

# In-memory job registry: job_id -> dict(state, progress, result, error, ...).
# Guarded by _JOBS_LOCK; pruned to _MAX_JOBS to bound memory.
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 40


def _set_progress(job_id, snapshot):
    """Store the latest progress snapshot for a running job (thread-safe)."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["progress"] = snapshot


def _run_job(job_id, params):
    """Worker thread body: run the pipeline and record result/error on the job."""
    try:
        result = walkytalky.run_from_query(
            progress=lambda snap: _set_progress(job_id, snap), **params)
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(state="done", result=result)
    except ValueError as exc:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(state="error", error=str(exc), error_status=400)
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to client
        traceback.print_exc()
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(
                    state="error", error=f"{type(exc).__name__}: {exc}", error_status=500)


def _prune_jobs():
    """Drop the oldest finished jobs once the registry exceeds _MAX_JOBS."""
    with _JOBS_LOCK:
        if len(_JOBS) <= _MAX_JOBS:
            return
        finished = sorted(
            (jid for jid, j in _JOBS.items() if j["state"] != "running"),
            key=lambda jid: _JOBS[jid]["created"])
        for jid in finished[:len(_JOBS) - _MAX_JOBS]:
            del _JOBS[jid]


class Handler(BaseHTTPRequestHandler):
    """Request handler serving the static UI and the search API."""

    def _send_json(self, status, payload):
        """Write a JSON response with the given HTTP status.

        Args:
            status: HTTP status code.
            payload: JSON-serialisable response body.
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        """Send a static file from the project directory.

        Args:
            path: Absolute path to the file.
            content_type: MIME type for the response.
        """
        if not os.path.isfile(path):
            self.send_error(404, "Not found")
            return
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_progress(self):
        """Return the latest progress snapshot (and result/error) for a job."""
        params = parse_qs(urlparse(self.path).query)
        job_id = (params.get("job") or [""])[0]
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            snap = dict(job) if job else None
        if snap is None:
            self._send_json(404, {"error": "unknown or expired job"})
            return
        payload = {"state": snap["state"], "progress": snap.get("progress")}
        if snap["state"] == "done":
            payload["result"] = snap.get("result")
        elif snap["state"] == "error":
            payload["error"] = snap.get("error")
        self._send_json(200, payload)

    def do_GET(self):  # noqa: N802 - required name from BaseHTTPRequestHandler
        """Serve the progress API, ``index.html`` at ``/``, and static files."""
        path = self.path.split("?", 1)[0]
        if path == "/api/progress":
            self._handle_progress()
            return
        rel = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        rel = rel.split("?", 1)[0]
        # Prevent path traversal outside the project directory.
        target = os.path.normpath(os.path.join(_HERE, rel))
        if not target.startswith(_HERE):
            self.send_error(403, "Forbidden")
            return
        ctype = "text/html; charset=utf-8" if target.endswith(".html") else "text/plain"
        self._send_file(target, ctype)

    def do_POST(self):  # noqa: N802 - required name from BaseHTTPRequestHandler
        """Handle ``POST /api/search`` by starting a background pipeline job."""
        if self.path.split("?", 1)[0] != "/api/search":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            source = (req.get("source") or "").strip()
            destination = (req.get("destination") or "").strip()
            if not source or not destination:
                self._send_json(400, {"error": "source and destination are required"})
                return
            params = {
                "source": source,
                "destination": destination,
                "mode": req.get("mode") or "walking",
                "wiki_url": (req.get("wiki") or "").strip(),
                "tourism_url": (req.get("tourism") or "").strip(),
                "radius_m": float(req.get("radius") or config.DEFAULT_RADIUS_M),
                "refresh": bool(req.get("refresh")),
            }
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        _prune_jobs()
        job_id = uuid.uuid4().hex
        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "state": "running", "progress": None, "result": None,
                "error": None, "created": time.time(),
            }
        threading.Thread(target=_run_job, args=(job_id, params), daemon=True).start()
        self._send_json(202, {"job_id": job_id})

    def log_message(self, fmt, *args):
        """Quiet the default per-request logging (keep pipeline prints clean)."""
        return


def main():
    """Start the threaded HTTP server until interrupted."""
    config.load_env()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"WalkyTalky UI running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
