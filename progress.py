"""Progress tracking with ETA for the WalkyTalky pipeline.

Emits structured progress snapshots as the pipeline advances through its fixed
sequence of steps, estimating time-to-finish from historical per-step durations
(persisted under the cache dir) with sensible first-run defaults. The same
tracker drives both the CLI console output and the web UI progress bar.

A snapshot (see :meth:`ProgressTracker.snapshot`) is a plain JSON-serialisable
dict so the web layer can return it verbatim to the browser.
"""

import json
import os
import time

import config

# Canonical ordered pipeline steps: (key, default label). Step 1's label is
# overridden per entry point (Maps-link parse vs. free-text geocoding).
STEPS = [
    ("input", "Resolving inputs"),
    ("route", "Fetching route geometry"),
    ("docs", "Gathering documents"),
    ("embed", "Embedding & indexing"),
    ("extract", "Extracting historical sites"),
    ("filter", "Geocoding & filtering to corridor"),
]

# First-run per-step duration estimates (seconds), tuned for this CPU-only box.
# Extraction dominates (batched 7B calls). Learned values replace these over time.
_DEFAULT_DURATIONS = {
    "input": 3.0,
    "route": 3.0,
    "docs": 25.0,
    "embed": 30.0,
    "extract": 150.0,
    "filter": 15.0,
}

_TIMINGS_PATH = os.path.join(config.CACHE_DIR, "timings.json")
_MAX_SAMPLES = 20  # cap the sample count so the moving average keeps adapting


def _load_timings():
    """Load learned per-step durations, or an empty dict on any failure."""
    try:
        with open(_TIMINGS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_timings(timings):
    """Persist learned per-step durations; never fatal."""
    try:
        config.ensure_cache_dir()
        with open(_TIMINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(timings, fh, indent=2)
    except OSError:
        pass


class ProgressTracker:
    """Tracks step progress and estimates time-to-finish for one pipeline run.

    Args:
        on_event: Optional callback invoked with a snapshot dict on every state
            change (step start / note / done). Used by the web layer.
        console: When ``True``, also print human-readable progress to stdout
            (preserves the CLI's ``[i/6]`` output).
        label_overrides: Optional ``{step_key: label}`` to customise step labels
            (e.g. step 1's wording differs by entry point).
    """

    def __init__(self, on_event=None, console=True, label_overrides=None):
        self._on_event = on_event
        self._console = console
        self._labels = {k: lbl for k, lbl in STEPS}
        if label_overrides:
            self._labels.update(label_overrides)
        self._timings = _load_timings()
        self._expected = {
            k: float(self._timings.get(k, {}).get("avg", _DEFAULT_DURATIONS[k]))
            for k, _ in STEPS
        }
        self._total_expected = sum(self._expected.values()) or 1.0
        self._index = {k: i for i, (k, _) in enumerate(STEPS)}
        self._start = time.time()
        self._current = None
        self._current_start = None
        self._durations = {}   # step_key -> actual seconds (for completed steps)
        self._detail = ""
        self._status = "running"

    def _close_current(self):
        """Record the elapsed duration of the step that was active, if any."""
        if self._current is not None:
            self._durations[self._current] = time.time() - self._current_start

    def start(self, key, label=None, detail=""):
        """Begin a pipeline step (closes/records the previous one)."""
        self._close_current()
        if label:
            self._labels[key] = label
        self._current = key
        self._current_start = time.time()
        self._detail = detail
        if self._console:
            idx = self._index[key] + 1
            print(f"[{idx}/{len(STEPS)}] {self._labels[key]}...")
            if detail:
                print(f"      {detail}")
        self._emit()

    def note(self, detail):
        """Attach a sub-status detail to the current step."""
        self._detail = detail
        if self._console and detail:
            print(f"      {detail}")
        self._emit()

    def done(self, detail=""):
        """Mark the run complete and persist the learned step durations."""
        self._close_current()
        self._current = None
        self._status = "done"
        if detail:
            self._detail = detail
            if self._console:
                print(f"      {detail}")
        self._persist_timings()
        self._emit()

    def fail(self, message):
        """Mark the run as failed with an error message."""
        self._status = "error"
        self._detail = message
        self._emit()

    def _persist_timings(self):
        """Fold this run's actual step durations into the moving averages."""
        for key, dur in self._durations.items():
            entry = self._timings.get(key, {})
            avg = float(entry.get("avg", dur))
            n = min(int(entry.get("n", 0)), _MAX_SAMPLES)
            self._timings[key] = {"avg": avg + (dur - avg) / (n + 1), "n": n + 1}
        _save_timings(self._timings)

    def snapshot(self):
        """Return the current progress as a JSON-serialisable dict.

        Includes percent complete, elapsed seconds, an ETA (remaining seconds,
        scaled by the run's observed pace so a slow/fast machine adapts), and a
        per-step checklist with ``pending``/``active``/``done`` states.
        """
        now = time.time()
        elapsed = now - self._start
        completed_expected = sum(self._expected[k] for k in self._durations)
        progressed = completed_expected
        if self._current is not None:
            t_in = now - self._current_start
            exp = self._expected.get(self._current, 0.0)
            frac = min(t_in / exp, 0.95) if exp > 0 else 0.0
            progressed += exp * frac
        progressed = min(progressed, self._total_expected)

        if self._status == "done":
            percent, eta = 100.0, 0.0
        elif self._status == "error":
            percent, eta = round(100.0 * progressed / self._total_expected, 1), 0.0
        else:
            percent = min(99.0, 100.0 * progressed / self._total_expected)
            # Scale the remaining estimate by observed pace vs. expectation.
            scale = (elapsed / progressed) if progressed > 0 else 1.0
            eta = max(1.0, (self._total_expected - progressed) * scale)

        steps = []
        for key, _ in STEPS:
            if key in self._durations:
                state = "done"
            elif key == self._current:
                state = "active"
            else:
                state = "pending"
            steps.append({
                "key": key,
                "label": self._labels[key],
                "state": state,
                "duration_s": round(self._durations[key], 1) if key in self._durations else None,
            })
        cur_idx = (self._index[self._current] + 1) if self._current is not None else len(STEPS)
        return {
            "status": self._status,
            "step_index": cur_idx,
            "total_steps": len(STEPS),
            "step_key": self._current,
            "label": self._labels.get(self._current, "Done"),
            "detail": self._detail,
            "percent": round(percent, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
            "steps": steps,
        }

    def _emit(self):
        """Push a snapshot to the callback; progress must never break the run."""
        if self._on_event:
            try:
                self._on_event(self.snapshot())
            except Exception:  # noqa: BLE001
                pass
