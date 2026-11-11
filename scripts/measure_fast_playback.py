"""Measures where Fast replay playback loses time (a diagnostic, not a test).

Run from the repository root with the project virtualenv:

    ./venv/bin/python scripts/measure_fast_playback.py
    ./venv/bin/python scripts/measure_fast_playback.py --mbps 20 --latency 40
    ./venv/bin/python scripts/measure_fast_playback.py --mbps 0 --speed normal
    ./venv/bin/python scripts/measure_fast_playback.py --mbps 0 --stall-ms 100

`--site PATH` serves another directory instead of `docs/`, so a candidate build
can be measured next to the published one without touching either.

The published site in `docs/` is served over HTTP with the caching headers
GitHub Pages sends (`Cache-Control: public, max-age=600`). Playwright's Chromium
then plays the recording at the chosen speed twice in one browser context:

  * "cold" — a fresh context, so every chunk has to come over the network
  * "warm" — the same page reloaded, so the chunks are in the HTTP cache

`--mbps` throttles the browser's network (0 disables throttling) and
`--latency` adds RTT, which is how a GitHub Pages visit behaves on a real link.
Throttling happens in the browser, so the server still answers at localhost
speed; the server's own request log is reported to show when bytes were sent.

For each run it reports, from data collected inside the page:

  * wall time to the last recorded checkpoint, and how many checkpoints were
    actually drawn (Fast draws fewer frames than it advances checkpoints)
  * every visible stall: a stretch of at least `--stall-ms` in which the
    displayed checkpoint did not change
  * for every stall, whether a chunk request was in flight, and its duration
  * every chunk request: start, end, duration, bytes, whether it came from the
    HTTP cache, and its lead time — how long before the playback needed that
    chunk the request began (negative means it was late and the page froze)
  * animation-frame gaps and long tasks (main-thread work over 50 ms), so a
    blocked main thread can be told apart from waiting on the network. Each
    stall is labelled accordingly: network, main thread or mixed.

Nothing is modified, committed or uploaded; `docs/` and the recording are only
read. Only the standard library and Playwright are used — the same optional
dependency the browser tests use. If Chromium is missing:

    ./venv/bin/python -m pip install playwright
    ./venv/bin/python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import http.server
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"

# The dwell time of each speed in milliseconds, from frontend/replay.js.
SPEED_DWELL_MS = {"slow": 1000, "normal": 50, "fast": 5}

# The page updates #timeline only when a checkpoint has been decoded and drawn,
# so this probe samples the displayed checkpoint once per animation frame.
PROBE = r"""
(() => {
  const state = { frames: [], renders: [], notes: [], longtasks: [] };
  window.__probe = state;

  if (window.PerformanceObserver) {
    try {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          state.longtasks.push([entry.startTime, entry.duration]);
        }
      }).observe({ type: "longtask", buffered: true });
    } catch (error) {
      // The long-task timeline is optional; the frame sampler is enough.
    }
  }

  const noteText = () => {
    const element = document.getElementById("connection-note");
    return element ? element.textContent || "" : "";
  };
  const timelineIndex = () => {
    const element = document.getElementById("timeline");
    return element ? Number(element.value) : -1;
  };

  let lastNote = null;
  function sample(time) {
    state.frames.push([time, timelineIndex()]);
    const text = noteText();
    if (text !== lastNote) {
      lastNote = text;
      state.notes.push([time, text]);
      const match = /(?:drawn in )?([0-9.]+) ms/.exec(text);
      // A note that says "cached" reuses the existing connection bitmap; one
      // without it follows a rebuild, which is the expensive case.
      if (match) state.renders.push([time, Number(match[1]), /cached/.test(text)]);
    }
    window.requestAnimationFrame(sample);
  }
  window.requestAnimationFrame(sample);

  // The view redraws every connection whenever its host changes size, so record
  // each size the host takes, with the page scrollbars that explain it.
  state.resizes = [];
  function watchHost() {
    const canvas = document.getElementById("network-canvas");
    const host = canvas && canvas.parentElement;
    if (!host) return;
    new ResizeObserver(() => {
      const rect = host.getBoundingClientRect();
      const root = document.documentElement;
      state.resizes.push({
        time: performance.now(),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        scrollbar: window.innerWidth - root.clientWidth,
        noteLength: noteText().length,
      });
    }).observe(host);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watchHost);
  } else {
    watchHost();
  }

  state.reset = () => {
    state.frames.length = 0;
    state.renders.length = 0;
    state.notes.length = 0;
    state.longtasks.length = 0;
    state.resizes.length = 0;
  };

  state.dump = () => ({
    frames: state.frames.slice(),
    renders: state.renders.slice(),
    notes: state.notes.slice(),
    longtasks: state.longtasks.slice(),
    resizes: state.resizes.slice(),
    resources: performance
      .getEntriesByType("resource")
      .filter((entry) => /chunk-\d+\.bin$/.test(entry.name))
      .map((entry) => ({
        name: entry.name,
        start: entry.startTime,
        responseStart: entry.responseStart,
        end: entry.responseEnd,
        duration: entry.duration,
        transferSize: entry.transferSize,
        encodedBodySize: entry.encodedBodySize,
        decodedBodySize: entry.decodedBodySize,
      })),
  });
})();
"""



class ReplayHandler(http.server.BaseHTTPRequestHandler):
    """Serves docs/ with GitHub Pages' caching headers and logs each transfer."""

    server_version = "ReplayProbe"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - the name is fixed by the base class
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path.endswith("/"):
            path += "index.html"
        root = self.server.root
        target = (root / path.lstrip("/")).resolve()
        if root != target and root not in target.parents:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return

        body = target.read_bytes()
        started = time.monotonic()
        self.send_response(200)
        self.send_header("Content-Type", self.content_type(target))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=600")
        self.end_headers()
        self.wfile.write(body)
        with self.server.lock:
            self.server.transfers.append(
                {
                    "path": path.lstrip("/"),
                    "start": started,
                    "end": time.monotonic(),
                    "bytes": len(body),
                }
            )

    @staticmethod
    def content_type(target: Path) -> str:
        suffix = target.suffix.lower()
        return {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json",
        }.get(suffix, "application/octet-stream")


@contextlib.contextmanager
def static_server(directory: Path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
    server.root = directory
    server.transfers = []
    server.lock = threading.Lock()
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def chunk_number(path: str) -> int:
    return int(path.rsplit("-", 1)[1].split(".")[0])


def change_points(frames: list[list[float]]) -> list[tuple[float, int]]:
    """The moments the displayed checkpoint changed, as (time, index)."""
    points: list[tuple[float, int]] = []
    for time_ms, index in frames:
        if not points or points[-1][1] != index:
            points.append((time_ms, index))
    return points


def frame_gaps(frames: list[list[float]]) -> dict[str, float]:
    gaps = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
    if not gaps:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(gaps)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def play(page, speed: str, timeout_ms: int) -> dict:
    """Plays from the first checkpoint to the last and returns the measurements."""
    page.wait_for_function("!document.getElementById('btn-play').disabled", timeout=180_000)
    page.select_option("#speed-select", speed)
    page.evaluate("window.__probe.reset()")
    clicked = page.evaluate("performance.now()")
    started = time.monotonic()
    page.click("#btn-play")
    page.wait_for_function(
        "document.getElementById('status-badge').textContent.includes('60,000')",
        timeout=timeout_ms,
    )
    elapsed = time.monotonic() - started
    dump = page.evaluate("window.__probe.dump()")
    dump["elapsed_s"] = elapsed
    dump["click_ms"] = clicked
    return dump


def describe_run(label: str, result: dict, stalls_ms: float, chunk_size: int, cps: float) -> dict:
    """Prints one run's timeline and returns the same numbers as a dict."""
    frames = result["frames"]
    renders = result["renders"]
    resources = result["resources"]
    elapsed = result["elapsed_s"]

    points = change_points(frames)
    shown = len(points)
    last_index = points[-1][1] if points else -1

    # A stall is a stretch in which the displayed checkpoint does not change.
    longtasks = result.get("longtasks", [])

    def overlap(from_ms: float, to_ms: float, window_start: float, window_end: float) -> float:
        return max(0.0, min(to_ms, window_end) - max(from_ms, window_start))

    stalls = []
    for (start, from_index), (end, to_index) in zip(points, points[1:]):
        gap = end - start
        if gap < stalls_ms:
            continue
        in_flight = [entry for entry in resources if entry["start"] <= end and entry["end"] >= start]
        # The longest animation-frame gap inside the stall shows a blocked main thread.
        inner = [b[0] - a[0] for a, b in zip(frames, frames[1:]) if a[0] >= start and b[0] <= end]
        longest_block = max(inner, default=0.0)
        # Response timings include time the request waited behind a busy main
        # thread, so the two are separated instead of blaming the network for
        # time that was really spent inside a redraw.
        blocked_ms = sum(overlap(task_start, task_start + task_ms, start, end)
                         for task_start, task_ms in longtasks)
        chunk_ms = max(
            (overlap(entry["start"], entry["end"], start, end) for entry in in_flight),
            default=0.0,
        )
        waiting_ms = max(0.0, chunk_ms - blocked_ms)
        if blocked_ms >= gap * 0.8:
            kind = "main thread"
        elif waiting_ms >= gap * 0.8:
            kind = "network"
        else:
            kind = "mixed"
        stalls.append(
            {
                "start": start,
                "seconds": gap / 1000,
                "from": from_index,
                "to": to_index,
                "skipped": to_index - from_index,
                "chunks": [chunk_number(entry["name"]) for entry in in_flight],
                "chunk_seconds": chunk_ms / 1000,
                "blocked_seconds": blocked_ms / 1000,
                "max_frame_gap_ms": longest_block,
                "kind": kind,
            }
        )

    requests = []
    for entry in sorted(resources, key=lambda item: item["start"]):
        number = chunk_number(entry["name"])
        start = (entry["start"] - result["click_ms"]) / 1000
        # When uninterrupted playback reaches the first checkpoint of the chunk.
        needed_at = (number * chunk_size) / cps
        requests.append(
            {
                "chunk": number,
                "start": start,
                "seconds": entry["duration"] / 1000,
                "mb": entry["encodedBodySize"] / 1e6,
                "cached": entry["transferSize"] == 0,
                "needed_at": needed_at,
                # Positive: the request began before playback needed the chunk.
                "lead": needed_at - start,
            }
        )

    # Long tasks are main-thread work over 50 ms; they cannot be network.
    blockers = [
        ((start - result["click_ms"]) / 1000, duration / 1000)
        for start, duration in result.get("longtasks", [])
        if start + duration > result["click_ms"]
    ]
    # The connection bitmap is rebuilt from scratch whenever the canvas host is
    # resized, so list the sizes it took during playback.
    resizes = [
        {
            "at": (entry["time"] - result["click_ms"]) / 1000,
            "size": f"{entry['width']}x{entry['height']}",
            "scrollbar": entry["scrollbar"],
            "note_length": entry["noteLength"],
        }
        for entry in result.get("resizes", [])
        if entry["time"] >= result["click_ms"]
    ]
    timings = [value for _, value, _ in renders]
    rebuilds = [(start, value) for start, value, cached in renders if not cached]
    gaps = frame_gaps(frames)
    report = {
        "label": label,
        "elapsed_s": elapsed,
        "shown": shown,
        "last_index": last_index,
        "per_second": shown / elapsed if elapsed else 0.0,
        "stalls": stalls,
        "stall_seconds": sum(item["seconds"] for item in stalls),
        "stalls_waiting_on_chunk": sum(1 for item in stalls if item["chunks"]),
        "stall_kinds": {kind: sum(1 for item in stalls if item["kind"] == kind) for kind in
                        ("network", "main thread", "mixed")},
        "requests": requests,
        "frame_gaps": gaps,
        "longtasks": blockers,
        "longtask_seconds": sum(duration for _, duration in blockers),
        "resizes": resizes,
        "renders": len(timings),
        "rebuilds": [
            {
                "at": (start - result["click_ms"]) / 1000,
                "milliseconds": value,
            }
            for start, value in rebuilds
            if start >= result["click_ms"]
        ],
        "render_p50": statistics.median(timings) if timings else 0.0,
        "render_p95": (
            sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))] if timings else 0.0
        ),
        "render_max": max(timings) if timings else 0.0,
        "fps": 1000 / gaps["median"] if gaps["median"] else 0.0,
    }

    print(f"\n== {label} ==")
    print(f"wall time to the last checkpoint : {elapsed:6.2f} s")
    print(
        f"checkpoints drawn                : {shown:6d} "
        f"({report['per_second']:.1f}/s, last index {last_index})"
    )
    print(
        f"animation frames                 : {report['fps']:.1f}/s median gap "
        f"{gaps['median']:.1f} ms, p95 {gaps['p95']:.1f} ms, max {gaps['max']:.1f} ms"
    )
    print(
        f"main thread blocked (>50 ms)     : {len(blockers)} long tasks, "
        f"{report['longtask_seconds']:.2f} s total, "
        f"max {max((duration for _, duration in blockers), default=0) * 1000:.0f} ms"
    )
    for offset, duration in sorted(blockers, key=lambda item: -item[1])[:5]:
        print(f"    {duration * 1000:7.1f} ms starting {offset:6.2f} s after the click")
    sizes = ", ".join(
        f"{size} x{count}" for size, count in
        collections.Counter(item["size"] for item in resizes).most_common()
    )
    print(
        f"canvas host resized              : {len(resizes)} times "
        f"(each rebuilds every connection)  {sizes or '-'}"
    )
    for item in resizes[:6]:
        print(
            f"    {item['at']:6.2f} s  {item['size']:>10s}  "
            f"scrollbar {item['scrollbar']} px  note {item['note_length']} chars"
        )
    print(
        f"canvas connection draw           : p50 {report['render_p50']:.1f} ms, "
        f"p95 {report['render_p95']:.1f} ms, max {report['render_max']:.1f} ms, "
        f"{len(report['rebuilds'])} rebuilds during playback "
        f"({sum(item['milliseconds'] for item in report['rebuilds']) / 1000:.2f} s)"
    )
    print(
        f"stalls >= {stalls_ms:.0f} ms               : {len(stalls)} "
        f"({report['stall_seconds']:.2f} s total, "
        f"{report['stalls_waiting_on_chunk']} with a chunk request in flight)"
    )
    for number, stall in enumerate(stalls, start=1):
        chunks = ", ".join(f"chunk-{chunk:04d}" for chunk in stall["chunks"]) or "no chunk request"
        print(
            f"  stall {number:2d}: {stall['seconds']:5.2f} s {stall['kind']:11s} "
            f"displayed {stall['from']:5d} -> {stall['to']:5d} "
            f"(+{stall['skipped']:4d})  blocked {stall['blocked_seconds']:5.2f} s  "
            f"{chunks} ({stall['chunk_seconds']:.2f} s)"
        )

    print("chunk requests:")
    for request in requests:
        state = "cache hit" if request["cached"] else "network  "
        print(
            f"  chunk-{request['chunk']:04d}  start {request['start']:7.2f} s  "
            f"took {request['seconds']:5.2f} s  {request['mb']:5.2f} MB  {state}  "
            f"needed at {request['needed_at']:6.2f} s  lead {request['lead']:+7.2f} s"
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure replay playback smoothness.")
    parser.add_argument("--mbps", type=float, default=12.0, help="download speed, 0 to disable throttling")
    parser.add_argument("--latency", type=float, default=60.0, help="round-trip latency in milliseconds")
    parser.add_argument("--speed", default="fast", choices=("slow", "normal", "fast"))
    parser.add_argument("--stall-ms", type=float, default=150.0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=180_000)
    parser.add_argument(
        "--site",
        type=Path,
        default=DOCS,
        help="site directory to serve (default: docs/, so a prototype elsewhere can be compared)",
    )
    parser.add_argument("--json", type=Path, help="write the raw measurements here")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "this diagnostic needs Playwright with Chromium:\n"
            "  ./venv/bin/python -m pip install playwright\n"
            "  ./venv/bin/python -m playwright install chromium",
            file=sys.stderr,
        )
        return 2

    with static_server(args.site) as (server, base), sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.add_init_script(PROBE)
        cdp = context.new_cdp_session(page)
        cdp.send("Network.enable")
        if args.mbps > 0:
            cdp.send(
                "Network.emulateNetworkConditions",
                {
                    "offline": False,
                    "latency": args.latency,
                    "downloadThroughput": int(args.mbps * 1e6 / 8),
                    "uploadThroughput": 500_000,
                    "connectionType": "wifi",
                },
            )

        link = "unthrottled" if args.mbps <= 0 else f"{args.mbps:.0f} Mbit/s, {args.latency:.0f} ms RTT"
        cps = 1000 / SPEED_DWELL_MS[args.speed]
        print(f"serving {args.site} at {base} over {link}, playing at {args.speed} ({cps:.0f}/s)")

        page.goto(f"{base}/index.html")
        cold = play(page, args.speed, args.timeout)
        page.reload()
        warm = play(page, args.speed, args.timeout)

        runs = [
            describe_run(f"cold cache ({link})", cold, args.stall_ms, args.chunk_size, cps),
            describe_run(f"warm cache ({link})", warm, args.stall_ms, args.chunk_size, cps),
        ]

        print("\nsummary")
        for report in runs:
            kinds = ", ".join(f"{count} {kind}" for kind, count in report["stall_kinds"].items() if count)
            print(
                f"  {report['label']}: {report['stall_seconds']:.2f} s stalled of "
                f"{report['elapsed_s']:.2f} s ({kinds or 'no stalls'}), "
                f"{report['longtask_seconds']:.2f} s in long tasks, "
                f"{len(report['resizes'])} canvas resizes, "
                f"{len(report['rebuilds'])} connection rebuilds"
            )

        print("\nchunk requests that went to the network")
        for label, result in (("cold", cold), ("warm", warm)):
            count = sum(1 for entry in result["resources"] if entry["transferSize"] != 0)
            print(f"  {label}: {count} of {len(result['resources'])} chunk requests")
        with server.lock:
            transfers = list(server.transfers)
        chunk_transfers = [item for item in transfers if "chunks/" in item["path"]]
        print(f"  the server wrote {len(chunk_transfers)} chunk files")

        if args.json:
            args.json.write_text(
                json.dumps({"runs": runs, "transfers": transfers}, indent=2), encoding="utf-8"
            )
            print(f"\nraw measurements written to {args.json}")

        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

