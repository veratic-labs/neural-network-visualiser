"""Milestone 5C verification: the static GitHub Pages site in docs/.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone5c.py

The test runs `scripts/build_pages.py` itself, so it fails when the build is
broken and also when `docs/` is stale. It then checks, in order:

  1. the build succeeds, is reproducible, and `docs/` holds only the replay
     page, its four modules, the stylesheet, `.nojekyll` and one validated
     recording (no Python, no backend, no dataset, no tests, no virtualenv)
  2. every copied file is byte-identical to its source, every chunk size matches
     the manifest, and all 6,001 checkpoint steps are readable from `docs/data`
  3. the published page is the Replay page, points at `./data/`, keeps the run
     name, contains no socket, backend or absolute URL, and references only
     files that exist
  4. the scripts keep the replay behaviour: three speeds, one initial-weight
     load, Full mode drawn from the Step 0 weights, Partial mode from the
     recorded checkpoint weights, chunks fetched one at a time
  5. over HTTP from a nested GitHub Pages-style path, the page and every data
     file are reachable at relative URLs only
  6. with Playwright's Chromium, the page really plays: the recorded values
     match the binary data, seeking and Previous/Next move exactly one recorded
     checkpoint, Full mode's connections stay still while Partial mode follows
     the checkpoint, all three speeds run, and a missing chunk stops the replay
     with the file's own error
  7. the repository stays intact: the replay page, training/, backend/ and the
     recording are unchanged, `docs/` is not Git-ignored, README.md documents
     the build and this test

Nothing is committed, pushed or uploaded; the recording is only read.
"""

from __future__ import annotations

import contextlib
import difflib
import functools
import hashlib
import http.server
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
RECORDING = REPO_ROOT / "recordings" / "mnist-epoch1"
DOCS = REPO_ROOT / "docs"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_pages.py"
PAGE = DOCS / "index.html"
DATA = DOCS / "data"

CHECKPOINT_COUNT = 6001
CHUNK_COUNT = 13
CHUNK_SIZE = 500
FIRST_STEP = 0
LAST_STEP = 60000
CONNECTION_COUNT = 109184
PARTIAL_COUNT = 1010
TRAIN_SAMPLES = 60000
ASSETS = ("style.css", "replay.js", "recording.js", "network.js", "panels.js")
DATA_FILES = ("manifest.json", "model.bin", "model-biases.bin", "partial-indices.bin")
COLOUR_STOPS = ["#b91c1c", "#fdba74", "#cbd5e1", "#93c5fd", "#6d28d9"]

REQUIRED_IDS = (
    "architecture-summary",
    "run-name",
    "status-badge",
    "network-canvas",
    "connection-note",
    "mode-all",
    "mode-partial",
    "sample-canvas",
    "sample-step",
    "sample-index",
    "sample-true",
    "sample-predicted",
    "sample-verdict",
    "output-list",
    "timeline",
    "timeline-readout",
    "timeline-first",
    "timeline-middle",
    "timeline-last",
    "btn-play",
    "btn-pause",
    "btn-prev",
    "btn-next",
    "speed-select",
    "controls-note",
    "progress-bar",
    "progress-fill",
    "progress-text",
    "progress-percent",
)

CHECKS = 0
FAILURES: list[str] = []
MANIFEST_CACHE: dict | None = None


def check(condition: object, label: str, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS  {label}")
        return
    FAILURES.append(label + (f": {detail}" if detail else ""))
    print(f"FAIL  {label}" + (f"\n      {detail}" if detail else ""))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def magic_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recording_manifest() -> dict:
    global MANIFEST_CACHE
    if MANIFEST_CACHE is None:
        MANIFEST_CACHE = read_json(DATA / "manifest.json")
    return MANIFEST_CACHE


def manifest_layout() -> tuple[dict[str, dict], int]:
    data = recording_manifest()
    layout = {field["name"]: field for field in data["format"]["record"]["fields"]}
    return layout, data["format"]["record_size"]


def decode_checkpoint(index: int) -> dict:
    """Decodes one published record the way `recording.js` does (from docs/data)."""
    data = recording_manifest()
    layout, record_size = manifest_layout()
    chunk_size = data["format"]["chunks"]["chunk_size_records"]
    path = DATA / "chunks" / f"chunk-{index // chunk_size:04d}.bin"
    raw = path.read_bytes()
    offset = (index % chunk_size) * record_size

    def numbers(name: str, code: str, count: int) -> list:
        field = layout[name]
        return list(struct.unpack_from("<" + code * count, raw, offset + field["offset"]))

    image_field = layout["image"]
    return {
        "index": index,
        "step": numbers("step", "I", 1)[0],
        "sample_position": numbers("sample_position", "I", 1)[0],
        "true_label": numbers("true_label", "B", 1)[0],
        "predicted_label": numbers("predicted_label", "B", 1)[0],
        "has_sample": bool(numbers("flags", "B", 1)[0] & (1 << 0)),
        "image": raw[
            offset
            + image_field["offset"] : offset
            + image_field["offset"]
            + image_field["count"]
        ],
        "probabilities": numbers("probabilities", "f", layout["probabilities"]["count"]),
        "loss": numbers("loss", "f", 1)[0],
        "activations": numbers("activations", "f", layout["activations"]["count"]),
        "partial_weights": numbers("partial_weights", "f", layout["partial_weights"]["count"]),
    }


def floats_close(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= 1e-9 + 1e-6 * abs(b) for a, b in zip(left, right))


def tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_build() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """A static file server that does not print a line per request."""

    def log_message(self, *args) -> None:
        pass


@contextlib.contextmanager
def static_server(directory: Path):
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def nested_repository():
    """A temporary parent that publishes docs/ one directory down.

    GitHub Pages serves a project site under /<repository>/ and publishes only
    docs/, so this layout proves the site works at that path and that nothing
    outside docs/ is reachable.
    """
    with tempfile.TemporaryDirectory(prefix="mlp-visualisation-5c-") as directory:
        root = Path(directory)
        published = root / "mlp-visualisation" / "docs"
        published.parent.mkdir()
        published.symlink_to(DOCS, target_is_directory=True)
        yield root


@contextlib.contextmanager
def nested_copy_of_docs(missing: str | None = None):
    """A copy of docs/ one directory down, optionally with one file removed."""
    with tempfile.TemporaryDirectory(prefix="mlp-visualisation-5c-copy-") as directory:
        root = Path(directory)
        target = root / "mlp-visualisation" / "docs"
        shutil.copytree(DOCS, target)
        if missing:
            (target / missing).unlink()
        yield root


def fetch(url: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    """Returns status, headers (lower-case names) and body; an error status is not raised."""
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, lower_headers(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, lower_headers(error.headers), b""


def lower_headers(headers) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def chunk_files() -> set[str]:
    return {f"data/chunks/chunk-{index:04d}.bin" for index in range(CHUNK_COUNT)}


def check_build() -> None:
    print("\n-- build --")
    recording_before = tree_digest(RECORDING)
    frontend_before = tree_digest(FRONTEND)

    first = run_build()
    check(
        first.returncode == 0,
        "scripts/build_pages.py builds the site",
        (first.stderr or first.stdout).strip()[-1200:],
    )
    if first.returncode != 0:
        return

    published = tree_digest(DOCS)
    second = run_build()
    check(
        second.returncode == 0,
        "a second build succeeds",
        (second.stderr or second.stdout).strip()[-1200:],
    )
    check(tree_digest(DOCS) == published, "rebuilding produces byte-identical files")

    output = first.stdout
    check("Static site built:" in output, "the build prints where it wrote the site")
    check("checkpoints 6,001" in output, "the build summary counts 6,001 checkpoints", output)
    check("13 × 500 records" in output, "the build summary counts 13 chunks of 500 records", output)
    check("109,184 connections" in output, "the build summary counts the network's connections")
    check("1,010 Partial weights" in output, "the build summary counts the Partial weights")
    checksum = recording_manifest()["model"]["initial_weights_sha256"]
    check(checksum in output, "the build summary prints the verified initial-weight checksum", output[-600:])

    check(tree_digest(FRONTEND) == frontend_before, "the build does not change frontend/")
    check(tree_digest(RECORDING) == recording_before, "the build does not change the recording")


def check_structure() -> None:
    print("\n-- published site --")
    directories = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*") if path.is_dir()}
    files = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*") if path.is_file()}

    check(
        directories == {"data", "data/chunks"},
        "docs/ contains only the page and its data directory",
        directories,
    )
    top_level = {name for name in files if "/" not in name}
    check(
        top_level == {"index.html", ".nojekyll", *ASSETS},
        "docs/ holds the page, .nojekyll and the five frontend files",
        sorted(top_level),
    )
    check((DOCS / ".nojekyll").stat().st_size == 0, ".nojekyll is an empty file")

    published_data = {name for name in files if name.startswith("data/") and name.count("/") == 1}
    check(
        published_data == {f"data/{name}" for name in DATA_FILES},
        "docs/data holds the manifest and the three model files",
        published_data,
    )
    published_chunks = {name for name in files if name.startswith("data/chunks/")}
    check(
        published_chunks == chunk_files(),
        "docs/data/chunks holds exactly 13 numbered chunks",
        sorted(published_chunks ^ chunk_files()),
    )

    forbidden = [
        name
        for name in files
        if Path(name).suffix in {".py", ".pyc", ".pyo", ".md", ".txt", ".gz", ".ipynb"}
        or name.startswith(("training/", "backend/", "recorder/", "tests/", "venv/", "frontend/", "recordings/"))
    ]
    check(
        not forbidden,
        "the site carries no code, dataset or documentation from the repository",
        forbidden[:10],
    )

    for name in ASSETS:
        check(
            sha256(DOCS / name) == sha256(FRONTEND / name),
            f"docs/{name} is byte-identical to frontend/{name}",
        )

    for name in DATA_FILES:
        check(
            sha256(DATA / name) == sha256(RECORDING / name),
            f"docs/data/{name} is byte-identical to the recording",
        )

    data = recording_manifest()
    for chunk in data["checkpoints"]["chunks"]:
        name = Path(chunk["file"]).name
        check(
            (DATA / "chunks" / name).stat().st_size == chunk["bytes"],
            f"docs/data/chunks/{name} matches its manifest size",
        )
    check(
        sum(chunk["bytes"] for chunk in data["checkpoints"]["chunks"])
        == data["checkpoints"]["total_bytes"],
        "the chunks add up to the manifest's total byte count",
    )

    check(
        (DATA / "model.bin").stat().st_size
        == data["full"]["connection_count"] * 4
        == CONNECTION_COUNT * 4,
        "model.bin holds one float32 per connection",
    )
    check(
        (DATA / "model-biases.bin").stat().st_size == data["full"]["bias_count"] * 4,
        "model-biases.bin holds one float32 per bias",
    )
    check(
        (DATA / "partial-indices.bin").stat().st_size
        == data["partial"]["count"] * 2
        == PARTIAL_COUNT * 2,
        "partial-indices.bin holds one uint16 per Partial weight",
    )
    weights = (DATA / "model.bin").read_bytes()
    biases = (DATA / "model-biases.bin").read_bytes()
    digest = hashlib.sha256()
    weight_offset = 0
    bias_offset = 0
    for layer in data["model"]["layers"]:
        weight_bytes = layer["weight_count"] * 4
        digest.update(weights[weight_offset : weight_offset + weight_bytes])
        weight_offset += weight_bytes
        digest.update(biases[bias_offset : bias_offset + layer["bias_count"] * 4])
        bias_offset += layer["bias_count"] * 4
    check(
        digest.hexdigest() == data["model"]["initial_weights_sha256"]
        and weight_offset == len(weights)
        and bias_offset == len(biases),
        "the published weights and biases reproduce the manifest's initial-weight checksum",
        digest.hexdigest(),
    )
    check(
        data["checkpoints"]["count"] == CHECKPOINT_COUNT,
        "the published manifest still declares 6,001 checkpoints",
    )
    check(
        data["model"]["train_samples"] == TRAIN_SAMPLES,
        "the published manifest still declares 60,000 training samples",
    )

    site_bytes = sum(path.stat().st_size for path in DOCS.rglob("*") if path.is_file())
    check(
        20 * 1024 * 1024 < site_bytes < 80 * 1024 * 1024,
        f"the site is roughly 34 MiB (found {site_bytes / 1024 / 1024:.1f} MiB)",
    )


def check_recording() -> None:
    print("\n-- published recording --")
    data = recording_manifest()
    layout, record_size = manifest_layout()
    recorded_steps = data["training"]["recorded_steps"]
    check(len(recorded_steps) == CHECKPOINT_COUNT, "the manifest lists a step for every checkpoint")

    steps: list[int] = []
    for chunk in data["checkpoints"]["chunks"]:
        raw = (DATA / chunk["file"]).read_bytes()
        offset = layout["step"]["offset"]
        steps.extend(
            struct.unpack_from("<I", raw, index * record_size + offset)[0]
            for index in range(chunk["count"])
        )

    check(len(steps) == CHECKPOINT_COUNT, "6,001 steps are readable from docs/data")
    check(steps == recorded_steps, "every published step matches the manifest's recorded steps")
    check(
        steps[0] == FIRST_STEP and steps[-1] == LAST_STEP,
        "the published timeline starts at Step 0 and ends at Step 60,000",
    )
    check(
        steps == sorted(steps) and len(set(steps)) == len(steps),
        "the published steps are unique and increasing",
    )

    first = decode_checkpoint(0)
    second = decode_checkpoint(1)
    last = decode_checkpoint(CHECKPOINT_COUNT - 1)
    check(not first["has_sample"], "checkpoint 0 is the initial model with no sample")
    check(second["has_sample"] and second["step"] == 10, "checkpoint 1 is the Step 10 sample")
    check(last["step"] == LAST_STEP, "the last checkpoint is Step 60,000")
    check(len(first["partial_weights"]) == PARTIAL_COUNT, "a record carries the 1,010 Partial weights")
    check(
        len(first["activations"]) == data["activations"]["count"],
        "a record carries every recorded activation",
    )
    check(
        len(second["probabilities"]) == 10 and abs(sum(second["probabilities"]) - 1) < 1e-3,
        "a sampled record carries ten probabilities that sum to one",
        sum(second["probabilities"]),
    )
    check(
        first["probabilities"] == [0.0] * 10,
        "the Step 0 record carries no probabilities, because it has no sample",
        first["probabilities"],
    )
    check(sum(first["image"]) == 0, "checkpoint 0 carries no image", sum(first["image"]))
    check(sum(second["image"]) > 0, "the Step 10 checkpoint carries an image")


def check_page() -> None:
    print("\n-- published page --")
    source = (FRONTEND / "replay.html").read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")

    check(page != source, "docs/index.html is not a copy of the replay page")
    check(
        'data-recording="./data/"' in page and 'data-run-name="mnist-epoch1"' in page,
        "the page reads ./data/ and keeps the run name that path cannot provide",
    )
    check(
        "../recordings" not in page and "recordings/" not in page,
        "the page never refers to the repository's recording directory",
    )
    check(
        "The live training page is not part of this published site." in page,
        "the page explains that it is the static copy, without a live mode",
    )
    check('<a href="index.html">' not in page, "the page no longer links to the live page")

    changed = [
        line[1:].strip()
        for line in difflib.unified_diff(source.splitlines(), page.splitlines(), n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    accounted = [
        line
        for line in changed
        if "data-recording" in line or "live" in line or "index.html" in line or "backend" in line
    ]
    check(
        4 <= len(changed) <= 6 and len(accounted) == len(changed),
        "the only edits to the replay page are the recording path and the live-page note",
        changed,
    )

    lowered = page.lower()
    for pattern in ("ws://", "wss://", "http://", "https://", "localhost", "127.0.0.1", "new websocket"):
        check(pattern not in lowered, f"the page contains no {pattern!r}")

    missing = [name for name in REQUIRED_IDS if f'id="{name}"' not in page]
    check(not missing, "the page keeps every replay element", missing)
    ids = re.findall(r'\bid="([^"]+)"', page)
    repeated = sorted({name for name in ids if ids.count(name) > 1})
    check(not repeated, "the page's element ids are unique", repeated)

    modules = {name: (DOCS / name).read_text(encoding="utf-8") for name in ASSETS if name.endswith(".js")}

    def referenced(name: str) -> bool:
        return f'"{name}"' in page or any(f'"./{name}"' in script for script in modules.values())

    for name in ASSETS:
        check(referenced(name), f"the page or one of its modules loads {name}")

    references = re.findall(r'(?:src|href)="([^"]+)"', page)
    local = [value for value in references if not value.startswith(("#", "data:", "mailto:"))]
    missing_files = [value for value in local if not (DOCS / value).is_file()]
    check(not missing_files, "every file the page references exists in docs/", missing_files)
    check(
        all(not value.startswith("/") for value in local),
        "every reference is relative, so the site works under /<repository>/ on GitHub Pages",
        local,
    )
    check(
        len(re.findall(r'\bdata-recording="', page)) == 1,
        "the page names its recording exactly once",
    )

    select = re.search(r'<select[^>]*\bid="speed-select"[^>]*>(.*?)</select>', page, re.DOTALL)
    body = select.group(1) if select else ""
    check(re.findall(r'value="([^"]+)"', body) == ["slow", "normal", "fast"], "the page offers three replay speeds")
    check(len(re.findall(r"<option[^>]*\bselected", body)) == 1, "exactly one speed is selected by default")

    for name in ("btn-play", "btn-pause", "btn-prev", "btn-next", "timeline"):
        element = re.search(rf'<[^>]*\bid="{name}"[^>]*>', page)
        check(bool(element) and "disabled" in element.group(0), f"#{name} starts disabled")


def without_comments(script: str) -> str:
    """Strips JS comments so prose in the modules is not mistaken for code."""
    text = re.sub(r"/\*.*?\*/", "", script, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def check_scripts() -> None:
    print("\n-- published scripts --")
    scripts = {name: (DOCS / name).read_text(encoding="utf-8") for name in ASSETS if name.endswith(".js")}
    code = {name: without_comments(script) for name, script in scripts.items()}
    replay = code["replay.js"]

    check(
        re.search(
            r"const SPEEDS = Object\.freeze\(\{\s*slow: 1000,\s*normal: 50,\s*fast: 5\s*\}\);",
            replay,
        ),
        "replay.js maps the three speeds to 1000, 50 and 5 ms",
    )
    check(
        "document.body.dataset.runName" in replay and "RUN_NAME" in replay,
        "replay.js prefers data-run-name for the header",
    )
    check(
        "recording.prefetch(currentIndex + 1)" in replay,
        "replay.js fetches the next chunk only when the replay reaches for it",
    )
    check(
        "checkpointWeights = densePartialWeights(checkpoint.partialWeights, partialSelection)" in replay,
        "Partial mode draws the weights recorded with the checkpoint",
    )
    check(
        "view.applyWeights(partial && checkpointWeights ? checkpointWeights : initialWeights)" in replay,
        "Full mode keeps drawing the Step 0 weights",
    )
    check(
        "Could not read the recording:" in replay,
        "replay.js reports a failed fetch on the page",
    )
    check(
        "DEFAULT_CHUNK_CACHE_SIZE = 4" in code["recording.js"],
        "recording.js keeps only a few chunks in memory",
    )
    for name, script in code.items():
        check("WebSocket" not in script and "ws://" not in script, f"{name} opens no socket")
        check("http://" not in script and "https://" not in script, f"{name} hard-codes no absolute URL")
        check("127.0.0.1" not in script and "localhost" not in script, f"{name} names no local server")


def check_http() -> None:
    print("\n-- HTTP from a nested GitHub Pages-style path --")
    with nested_repository() as root, static_server(root) as base:
        site = f"{base}/mlp-visualisation/docs"
        status, headers, body = fetch(f"{site}/index.html")
        check(status == 200, "the page is served over HTTP", status)
        check(body == PAGE.read_bytes(), "the served page is docs/index.html")
        check("text/html" in headers.get("content-type", ""), "the page is served as HTML", headers.get("content-type"))

        status, _, body = fetch(f"{site}/")
        check(status == 200 and body == PAGE.read_bytes(), "a directory URL also serves the page, as GitHub Pages does")

        for name in ASSETS:
            status, headers, body = fetch(f"{site}/{name}")
            check(status == 200 and body == (DOCS / name).read_bytes(), f"{name} is served unchanged")
            if name.endswith(".js"):
                check(
                    "javascript" in headers.get("content-type", ""),
                    f"{name} is served as JavaScript so module imports are allowed",
                    headers.get("content-type"),
                )

        status, _, body = fetch(f"{site}/data/manifest.json")
        check(status == 200 and json.loads(body)["checkpoints"]["count"] == CHECKPOINT_COUNT, "the manifest is served")

        for name in DATA_FILES[1:]:
            url = f"{site}/data/{name}"
            status, headers, _ = fetch(url, method="HEAD")
            check(
                status == 200 and int(headers.get("content-length", -1)) == (DATA / name).stat().st_size,
                f"data/{name} downloads with its published size",
                (status, headers.get("content-length")),
            )

        for chunk in recording_manifest()["checkpoints"]["chunks"]:
            url = f"{site}/data/{chunk['file']}"
            status, headers, _ = fetch(url, method="HEAD")
            check(
                status == 200 and int(headers.get("content-length", -1)) == chunk["bytes"],
                f"{chunk['file']} downloads with its recorded size",
                (status, headers.get("content-length")),
            )

        status, _, body = fetch(f"{site}/data/chunks/chunk-0012.bin")
        check(
            status == 200 and hashlib.sha256(body).hexdigest() == sha256(DATA / "chunks/chunk-0012.bin"),
            "the last chunk downloads byte for byte",
        )

        for missing in (
            "training/training.py",
            "backend/server.py",
            "recordings/mnist-epoch1/manifest.json",
            "data/chunks/chunk-0013.bin",
            "data/model-weights.bin",
        ):
            status, _, _ = fetch(f"{base}/mlp-visualisation/{missing}")
            check(status == 404, f"the site does not publish {missing}", status)


BROWSER_TIMEOUT = 120_000
MAIN_INDICES = (0, 1, 777, 3000, 4000, 6000)

READY = """() => {
  const timeline = document.getElementById('timeline');
  return Boolean(timeline) && !timeline.disabled;
}"""

SEEK = """(index) => {
  const timeline = document.getElementById('timeline');
  timeline.value = String(index);
  timeline.dispatchEvent(new Event('input', { bubbles: true }));
}"""

SNAPSHOT = """() => {
  const canvas = document.getElementById('network-canvas');
  window.__milestone5c = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  return window.__milestone5c.length;
}"""

PIXELS = """() => document.getElementById('network-canvas').toDataURL()"""

# Differences between two frames outside the four neuron columns. Full mode
# draws the same Step 0 weights at every checkpoint, so only the activation
# dots and the output labels may change; a changed connection means the canvas
# was redrawn from a checkpoint's weights.
OUTSIDE_BANDS = """() => {
  const canvas = document.getElementById('network-canvas');
  const before = window.__milestone5c;
  const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  if (!before || before.length !== data.length) return { error: 'the canvas size changed' };

  const cssWidth = canvas.getBoundingClientRect().width;
  const dpr = canvas.width / cssWidth;
  const width = canvas.width / dpr;
  const padX = width < 620 ? 34 : 62;
  const usable = Math.max(1, width - padX * 2);
  const bands = [];
  for (let layer = 0; layer < 4; layer += 1) {
    const centre = (padX + (usable * layer) / 3) * dpr;
    bands.push([centre - 26 * dpr, centre + 26 * dpr]);
  }

  let inside = 0;
  let outside = 0;
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const i = (y * canvas.width + x) * 4;
      if (
        data[i] === before[i] &&
        data[i + 1] === before[i + 1] &&
        data[i + 2] === before[i + 2] &&
        data[i + 3] === before[i + 3]
      ) {
        continue;
      }
      const inBand = bands.some(([low, high]) => x >= low && x <= high);
      if (inBand) inside += 1;
      else outside += 1;
    }
  }
  return { inside, outside, width: canvas.width, height: canvas.height };
}"""

PAGE_DECODE = """async (site) => {
  const module = await import(new URL('recording.js', site).href);
  const recording = await module.Recording.open(new URL('data/', site).href);
  const decoded = [];
  for (const index of [0, 1, 777, 3000, 4000, 6000]) {
    const checkpoint = await recording.checkpoint(index);
    decoded.push({
      index: checkpoint.index,
      step: checkpoint.step,
      chunkIndex: checkpoint.chunkIndex,
      hasSample: checkpoint.hasSample,
      samplePosition: checkpoint.samplePosition,
      trueLabel: checkpoint.trueLabel,
      predictedLabel: checkpoint.predictedLabel,
      image: Array.from(checkpoint.image),
      probabilities: Array.from(checkpoint.probabilities),
      loss: checkpoint.loss,
      activations: Array.from(checkpoint.activations),
      partialWeights: Array.from(checkpoint.partialWeights),
      cachedChunks: recording.cachedChunkCount(),
    });
  }
  return {
    checkpoints: decoded,
    checkpointCount: recording.checkpointCount,
    trainSamples: recording.trainSamples,
    totalSteps: recording.totalSteps,
    cadence: recording.cadence,
  };
}"""


def browser_ready() -> tuple[bool, str]:
    """Returns whether a Playwright Chromium can be launched, and why not."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright is not installed"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
    except Exception as error:  # noqa: BLE001 - the message is the point
        reason = next(
            (
                line.strip()
                for line in str(error).splitlines()
                if "error while loading shared libraries" in line or "Executable doesn't exist" in line
            ),
            error.__class__.__name__,
        )
        return False, (
            f"Chromium will not start ({reason}). "
            "Install its system libraries with `./venv/bin/python -m playwright install-deps chromium`."
        )
    return True, ""


def read_text(page, selector: str) -> str:
    """An element's own text, with runs of whitespace collapsed.

    `textContent` rather than `innerText`: `innerText` reports what is rendered,
    including `text-transform: uppercase` (the `.tag` and `.metric` labels), so
    the DOM text and the recorded value would differ only in letter case.
    """
    return page.eval_on_selector(selector, r"node => node.textContent.replace(/\s+/g, ' ').trim()")


def wait_for_status(page, step: int, timeout: int = BROWSER_TIMEOUT) -> None:
    page.wait_for_function(
        "expected => document.getElementById('status-badge').textContent.trim() === expected",
        arg=f"Replay · step {step:,}",
        timeout=timeout,
    )


def seek_to(page, index: int, step: int) -> None:
    page.evaluate(SEEK, index)
    wait_for_status(page, step)


def wait_for_note(page, fragment: str) -> None:
    page.wait_for_function(
        "fragment => document.getElementById('connection-note').textContent.includes(fragment)",
        arg=fragment,
        timeout=BROWSER_TIMEOUT,
    )


def data_paths(urls: list[str]) -> list[str]:
    return [url.split("/docs/", 1)[1] for url in urls if "/docs/data/" in url]


def chunk_requests(urls: list[str]) -> set[str]:
    return {path for path in data_paths(urls) if path.startswith("data/chunks/")}


def check_browser_load(page, requests: list[str]) -> None:
    page.wait_for_function(READY, timeout=BROWSER_TIMEOUT)

    check(read_text(page, "#run-name") == "mnist-epoch1", "the page names the run it replays", read_text(page, "#run-name"))
    check(
        read_text(page, "#architecture-summary") == "784 → 128 → 64 → 10",
        "the page reads the architecture from the published manifest",
        read_text(page, "#architecture-summary"),
    )
    check(read_text(page, "#status-badge") == "Replay · step 0", "the page opens on Step 0", read_text(page, "#status-badge"))
    check(read_text(page, "#timeline-readout") == "step 0", "the timeline reads Step 0")
    check(read_text(page, "#timeline-last") == "step 60,000", "the timeline ends at Step 60,000", read_text(page, "#timeline-last"))
    check(read_text(page, "#progress-percent") == "0.0%", "the progress bar starts at 0.0%")
    check(page.is_disabled("#btn-prev"), "Previous is disabled at the first checkpoint")
    check(page.is_disabled("#btn-pause"), "Pause is disabled while stopped")
    check(not page.is_disabled("#btn-play"), "Play is enabled at the first checkpoint")
    check(read_text(page, "#sample-step") == "step 0", "the sample panel reads Step 0")
    check(read_text(page, "#sample-index") == "–", "Step 0 shows no sample counter")
    check(read_text(page, "#sample-verdict") == "No sample at Step 0", "Step 0 explains that it recorded no sample")
    check(read_text(page, "#sample-true") == "–" and read_text(page, "#sample-predicted") == "–", "Step 0 shows no digits")

    wait_for_note(page, "All 109,184 connections from the Step 0 initial weights")
    check("stays still" in read_text(page, "#connection-note"), "the page explains that Full mode stays still")

    loaded = data_paths(requests)
    check(
        set(loaded)
        == {
            "data/manifest.json",
            "data/model.bin",
            "data/model-biases.bin",
            "data/partial-indices.bin",
            "data/chunks/chunk-0000.bin",
        },
        "loading the page fetches the manifest, the model files and only the first chunk",
        sorted(set(loaded)),
    )
    check(len(loaded) == 5, "loading the page makes five data requests", loaded)
    check(
        chunk_requests(requests) == {"data/chunks/chunk-0000.bin"},
        "loading the page fetches one chunk, not the whole recording",
        sorted(chunk_requests(requests)),
    )


def check_browser_seeking(page, requests: list[str]) -> None:
    seen = set(data_paths(requests))
    seek_to(page, 3000, 30000)

    check(read_text(page, "#timeline-readout") == "step 30,000", "the readout follows the timeline")
    check(read_text(page, "#progress-percent") == "50.0%", "halfway through the epoch reads 50.0%")
    check(
        read_text(page, "#progress-text").startswith("30,000 / 60,000 steps"),
        "the progress bar counts recorded steps",
        read_text(page, "#progress-text"),
    )
    check(
        "checkpoint 3,001 of 6,001" in read_text(page, "#controls-note"),
        "the controls name the checkpoint on screen",
        read_text(page, "#controls-note"),
    )
    added = set(data_paths(requests)) - seen
    check(added == {"data/chunks/chunk-0006.bin"}, "seeking to Step 30,000 fetches only chunk 6", sorted(added))

    expected = decode_checkpoint(3000)
    check(
        read_text(page, "#sample-index") == f"{expected['sample_position']:,} / 60,000",
        "the sample counter matches the record",
        read_text(page, "#sample-index"),
    )
    check(read_text(page, "#sample-true") == str(expected["true_label"]), "the true digit matches the record")
    check(read_text(page, "#sample-predicted") == str(expected["predicted_label"]), "the predicted digit matches the record")
    verdict = "Correct" if expected["predicted_label"] == expected["true_label"] else "Incorrect"
    check(read_text(page, "#sample-verdict") == verdict, "the verdict matches the record", read_text(page, "#sample-verdict"))
    check(read_text(page, "#sample-step") == "step 30,000", "the sample panel follows the timeline")

    values = page.eval_on_selector_all(".output-value", "nodes => nodes.map(node => node.textContent)")
    percentages = [float(value.rstrip("%")) for value in values]
    recorded = [probability * 100 for probability in expected["probabilities"]]
    check(len(values) == 10, "the page draws ten output bars", values)
    check(
        all(abs(actual - wanted) <= 0.051 for actual, wanted in zip(percentages, recorded)),
        "the ten output bars match the recorded probabilities",
        (values, [f"{value:.1f}%" for value in recorded]),
    )
    predicted_row = page.eval_on_selector_all(
        ".output-row", "nodes => nodes.findIndex(node => node.classList.contains('is-predicted'))"
    )
    check(predicted_row == expected["predicted_label"], "the predicted bar is highlighted", predicted_row)

    before = len(data_paths(requests))
    page.click("#btn-next")
    wait_for_status(page, 30010)
    check(read_text(page, "#sample-step") == "step 30,010", "Next advances one recorded checkpoint")
    page.click("#btn-prev")
    wait_for_status(page, 30000)
    check(read_text(page, "#timeline-readout") == "step 30,000", "Previous returns one recorded checkpoint")
    check(len(data_paths(requests)) == before, "stepping inside a cached chunk fetches nothing new")


def check_browser_modes(page, requests: list[str]) -> None:
    seek_to(page, 3000, 30000)
    page.click("#mode-partial")
    wait_for_note(page, "Partial mode: the same 1,010 connections")
    note = read_text(page, "#connection-note")
    check("Step 0" in note, "Partial mode says the selection was fixed at Step 0", note)
    check("selected checkpoint's own weights" in note, "Partial mode recolours from the checkpoint's weights", note)
    partial_3000 = page.evaluate(PIXELS)

    seen = set(data_paths(requests))
    seek_to(page, 4000, 40000)
    wait_for_note(page, "Partial mode:")
    partial_4000 = page.evaluate(PIXELS)
    check(partial_3000 != partial_4000, "Partial mode redraws when the checkpoint's weights change")
    check(
        set(data_paths(requests)) - seen == {"data/chunks/chunk-0008.bin"},
        "seeking to Step 40,000 fetches only chunk 8",
        sorted(set(data_paths(requests)) - seen),
    )

    seek_to(page, 3000, 30000)
    wait_for_note(page, "Partial mode:")
    check(page.evaluate(PIXELS) == partial_3000, "returning to a checkpoint redraws the identical Partial frame")

    page.click("#mode-all")
    wait_for_note(page, "All 109,184 connections from the Step 0 initial weights")
    page.evaluate(SNAPSHOT)
    seen = set(data_paths(requests))
    seek_to(page, 6000, 60000)
    wait_for_note(page, "All 109,184 connections from the Step 0 initial weights")
    difference = page.evaluate(OUTSIDE_BANDS)
    check("error" not in difference, "the canvas keeps its size between checkpoints", difference)
    check(
        difference.get("outside") == 0,
        "Full mode draws the same connections at Step 60,000 as at Step 30,000",
        difference,
    )
    check(difference.get("inside", 0) > 0, "the activation dots do change between checkpoints", difference)
    check(
        set(data_paths(requests)) - seen == {"data/chunks/chunk-0012.bin"},
        "seeking to the last checkpoint fetches only chunk 12",
        sorted(set(data_paths(requests)) - seen),
    )
    check(read_text(page, "#status-badge") == "Replay · step 60,000", "the page reaches Step 60,000")


def check_browser_speeds(page, requests: list[str]) -> None:
    seek_to(page, 0, 0)
    page.select_option("#speed-select", "slow")
    started = time.monotonic()
    page.click("#btn-play")
    wait_for_status(page, 10, timeout=20_000)
    elapsed = time.monotonic() - started
    check(
        "1 checkpoint per second" in read_text(page, "#controls-note"),
        "Slow plays one checkpoint per second",
        read_text(page, "#controls-note"),
    )
    page.click("#btn-pause")
    check(read_text(page, "#timeline-readout") == "step 10", "pausing stops at the checkpoint on screen")
    check(0.8 <= elapsed <= 8.0, f"Slow takes about a second per checkpoint (took {elapsed:.2f} s)")

    seek_to(page, 0, 0)
    page.select_option("#speed-select", "normal")
    page.click("#btn-play")
    page.wait_for_timeout(800)
    check(
        "20 checkpoints per second" in read_text(page, "#controls-note"),
        "Normal plays twenty checkpoints per second",
        read_text(page, "#controls-note"),
    )
    page.click("#btn-pause")
    paused = read_text(page, "#timeline-readout")
    page.wait_for_timeout(200)
    check(read_text(page, "#timeline-readout") == paused, "pausing stops the readout where it was")
    normal_step = int(paused.replace("step ", "").replace(",", ""))
    check(50 <= normal_step <= 500, f"Normal advances about twenty checkpoints per second (reached {paused})")

    seek_to(page, 0, 0)
    page.select_option("#speed-select", "fast")
    started = time.monotonic()
    page.click("#btn-play")
    wait_for_status(page, LAST_STEP)
    elapsed = time.monotonic() - started
    check(elapsed < 90, f"Fast replays the epoch in about thirty seconds (took {elapsed:.1f} s)")
    check(read_text(page, "#progress-percent") == "100.0%", "the last checkpoint reads 100.0%")
    check(
        "6,001 / 6,001" in read_text(page, "#progress-text"),
        "the last checkpoint is 6,001 of 6,001",
        read_text(page, "#progress-text"),
    )
    check(page.is_disabled("#btn-play"), "Play is disabled at the last checkpoint")
    check(page.is_disabled("#btn-next"), "Next is disabled at the last checkpoint")
    check(not page.is_disabled("#btn-prev"), "Previous is enabled at the last checkpoint")

    page.click("#btn-prev")
    wait_for_status(page, LAST_STEP - 10)
    check(read_text(page, "#progress-percent") == "100.0%", "Step 59,990 also reads 100.0%")
    page.click("#btn-next")
    wait_for_status(page, LAST_STEP)

    check(
        chunk_requests(requests) == {f"data/chunks/chunk-{index:04d}.bin" for index in range(CHUNK_COUNT)},
        "by the end every chunk was fetched, and nothing outside data/chunks was",
        sorted(chunk_requests(requests)),
    )


def check_browser_recording(page, base: str) -> None:
    site = f"{base}/mlp-visualisation/docs/"
    response = page.evaluate(PAGE_DECODE, site)

    check(response["checkpointCount"] == CHECKPOINT_COUNT, "the browser reads 6,001 checkpoints from the manifest")
    check(response["trainSamples"] == TRAIN_SAMPLES, "the browser reads 60,000 training samples")
    check(response["cadence"] == 10 and response["totalSteps"] == LAST_STEP, "the browser reads the 10-step cadence")
    check(len(response["checkpoints"]) == len(MAIN_INDICES), "the browser decoded every requested checkpoint")

    for decoded in response["checkpoints"]:
        index = decoded["index"]
        local = decode_checkpoint(index)
        label = f"checkpoint {index:,}"
        check(
            (
                decoded["step"] == local["step"]
                and decoded["chunkIndex"] == index // CHUNK_SIZE
                and decoded["hasSample"] == local["has_sample"]
                and decoded["samplePosition"] == local["sample_position"]
                and decoded["trueLabel"] == local["true_label"]
                and decoded["predictedLabel"] == local["predicted_label"]
                and decoded["loss"] == local["loss"]
            ),
            f"{label}: the browser's step, sample and labels match the binary record",
            decoded,
        )
        check(decoded["image"] == list(local["image"]), f"{label}: the browser decodes the same 784 pixels")
        check(
            floats_close(decoded["probabilities"], local["probabilities"]),
            f"{label}: the browser decodes the same probabilities",
        )
        check(
            floats_close(decoded["activations"], local["activations"]),
            f"{label}: the browser decodes the same 202 activations",
        )
        check(
            floats_close(decoded["partialWeights"], local["partial_weights"]),
            f"{label}: the browser decodes the same 1,010 Partial weights",
        )
        check(
            decoded["cachedChunks"] <= 4,
            f"{label}: the browser keeps at most four chunks",
            decoded["cachedChunks"],
        )


def check_browser_failure(page, requests: list[str]) -> None:
    page.wait_for_function(READY, timeout=BROWSER_TIMEOUT)
    check(read_text(page, "#status-badge") == "Replay · step 0", "the incomplete copy still opens on Step 0")

    page.evaluate(SEEK, 6000)
    page.wait_for_function(
        "() => document.getElementById('status-badge').textContent.trim() === 'Replay data error'",
        timeout=BROWSER_TIMEOUT,
    )
    note = read_text(page, "#connection-note")
    check("Could not read the recording:" in note, "a missing chunk is reported on the page", note)
    check("chunk 12 is missing" in note, "the error names the missing chunk", note)
    check("HTTP 404" in note, "the error includes the HTTP status", note)
    check(any(url.endswith("chunk-0012.bin") for url in requests), "the browser asked for the missing chunk")
    check(page.is_disabled("#timeline"), "the timeline is disabled after a failed read")
    check(page.is_disabled("#btn-play"), "Play is disabled after a failed read")
    check(page.is_disabled("#btn-prev"), "Previous is disabled after a failed read")
    check(page.is_disabled("#btn-next"), "Next is disabled after a failed read")
    check(page.is_disabled("#speed-select"), "the speed selector is disabled after a failed read")
    check(read_text(page, "#timeline-readout") == "step 0", "the page keeps the last good checkpoint on the readout")
    check(
        "Reload the page" in read_text(page, "#controls-note"),
        "the controls say how to recover",
        read_text(page, "#controls-note"),
    )

    # A narrower window makes the canvas resize, which re-renders the network. The
    # error must survive that render: it is the only description of the failure.
    page.set_viewport_size({"width": 1100, "height": 900})
    page.wait_for_timeout(200)
    after_resize = read_text(page, "#connection-note")
    check(
        after_resize == note,
        "a window resize keeps the error under the canvas",
        after_resize,
    )


def check_browser() -> None:
    print("\n-- browser --")
    available, reason = browser_ready()
    if not available:
        print(f"SKIP  the page was not driven in a browser: {reason}")
        return

    from playwright.sync_api import sync_playwright

    with nested_repository() as root, nested_copy_of_docs("data/chunks/chunk-0012.bin") as broken:
        with static_server(root) as base, static_server(broken) as broken_base:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    context = browser.new_context(viewport={"width": 1280, "height": 900})
                    page = context.new_page()
                    requests: list[str] = []
                    console_errors: list[str] = []
                    page_errors: list[str] = []
                    failed: list[str] = []
                    sockets: list[str] = []

                    page.on("request", lambda request: requests.append(request.url))
                    page.on(
                        "console",
                        lambda message: console_errors.append(message.text)
                        if message.type == "error"
                        else None,
                    )
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.on("requestfailed", lambda request: failed.append(f"{request.url}: {request.failure}"))
                    page.on("websocket", lambda socket: sockets.append(socket.url))

                    page.goto(f"{base}/mlp-visualisation/docs/", wait_until="load")
                    check_browser_load(page, requests)
                    check_browser_seeking(page, requests)
                    check_browser_modes(page, requests)
                    check_browser_speeds(page, requests)
                    check_browser_recording(page, base)

                    check(not console_errors, "the page logs no errors", console_errors[:5])
                    check(not page_errors, "the page throws no exception", page_errors[:5])
                    check(not failed, "no request fails", failed[:5])
                    check(not sockets, "the page opens no WebSocket", sockets[:5])

                    broken_page = context.new_page()
                    broken_requests: list[str] = []
                    broken_errors: list[str] = []
                    broken_page.on("request", lambda request: broken_requests.append(request.url))
                    broken_page.on("console", lambda message: broken_errors.append(message.text))
                    broken_page.goto(f"{broken_base}/mlp-visualisation/docs/", wait_until="load")
                    check_browser_failure(broken_page, broken_requests)
                    check(
                        any("replay stopped" in message for message in broken_errors),
                        "the failed read is logged once, with the recording's error",
                        broken_errors[:3],
                    )
                finally:
                    browser.close()


def git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True)


def check_repository() -> None:
    print("\n-- repository --")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("docs/", "scripts/build_pages.py", "GitHub Pages", "tests/test_milestone5c.py"):
        check(phrase in readme, f"README.md mentions {phrase!r}")

    ignored = git("check-ignore", "-q", "docs/index.html")
    check(ignored.returncode == 1, "docs/ is not ignored, so the site is committed like any other source")

    ignored = git("check-ignore", "-q", "recordings/mnist-epoch1/manifest.json")
    check(ignored.returncode == 0, "recordings/ stays ignored, so the 34 MiB site is not committed twice")

    changed = git("status", "--porcelain", "--", "training", "backend", "recorder", "frontend")
    paths = sorted(line[3:].strip() for line in changed.stdout.splitlines() if line.strip())
    check(
        paths == ["frontend/replay.js"],
        "among frontend/, training/, backend/ and recorder/, only replay.js differs from the last commit",
        paths,
    )

    diff = git("diff", "-U0", "--", "frontend/replay.js").stdout
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removed = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    check(
        any("RUN_NAME" in line for line in added)
        and removed == ["-  elements.runName.textContent = runName(RECORDING_URL);"],
        "replay.js names the header's run from data-run-name, so a published copy can name its run",
        (added, removed),
    )
    check(
        any("if (failed) return;" in line for line in added)
        and any("failed = true;" in line for line in added),
        "replay.js keeps a read error under the canvas when a later render repaints it",
        added,
    )
    check(len(added) <= 20, "the uncommitted change to replay.js stays a handful of lines", len(added))

    before_refusals = tree_digest(DOCS)
    for target in (".", "frontend", "recordings/mnist-epoch1"):
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--output", target],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        check(
            result.returncode == 1 and "BUILD FAILED" in result.stderr,
            f"the build refuses to write into {target}",
            result.stderr.strip().splitlines()[-1:] or result.returncode,
        )

    survivors = [
        name
        for name in ("frontend/replay.html", "recordings/mnist-epoch1/manifest.json", "training/training.py")
        if not (REPO_ROOT / name).exists()
    ]
    check(not survivors, "the refused builds delete nothing", survivors)
    check(tree_digest(DOCS) == before_refusals, "the refused builds leave the published site alone")


def main() -> int:
    print(f"Milestone 5C: the static site in {DOCS.relative_to(REPO_ROOT).as_posix()}/")

    check_repository()
    check_build()
    if FAILURES:
        print(f"\n{CHECKS} checks, {FAILURES} failed")
        print("MILESTONE 5C FAILED")
        return 1
    check_structure()
    check_recording()
    check_page()
    check_scripts()
    check_http()
    check_browser()

    print(f"\n{CHECKS} checks, {FAILURES} failed")
    if FAILURES:
        print("MILESTONE 5C FAILED")
        return 1
    print("MILESTONE 5C PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

