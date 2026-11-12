"""Milestone 5B verification: the browser-based offline replay page.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone5b.py

Checks, in order:
  1. frontend/replay.html exists and is a self-contained page: it loads
     style.css and replay.js as a module, names its recording once through
     `data-recording`, and contains no WebSocket, no socket and no backend URL
  2. the page's element ids are unique and cover every id replay.js looks up;
     the mode toggle, the four transport buttons, the speed selector, the
     timeline and the progress bar are all present and start disabled
  3. replay.js reads the recording through recording.js, draws with the shared
     NetworkView and formats with the same panels.js helpers; it never opens a
     socket, never hard-codes a count the manifest owns and never interpolates
     between checkpoints
  4. replay.js imports panels.js instead of carrying its own copy of the helpers
  5. the five colour stops of the CSS weight scale are the renderer's stops and
     the manifest's stops
  6. the recording itself decodes the way the replay reads it: checkpoint 0 is
     the initial model with no sample, the last checkpoint is Step 60,000, and
     every decoded step matches the manifest's recorded step list
  7. training/'s executable code is unchanged, recordings/ is ignored by Git and
     README.md documents the replay page, the static server and this test
"""

from __future__ import annotations

import ast
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
RECORDING = REPO_ROOT / "recordings" / "mnist-epoch1"

CHECKS = 0
FAILURES: list[str] = []

PANEL_HELPERS = (
    "buildOutputRows",
    "formatNumber",
    "renderProgressPanel",
    "setStatus",
    "updateOutputRows",
    "updateSamplePanel",
)

# The five stops of network.js's fixed ±0.5 weight scale, low to high.
COLOUR_STOPS = ["#b91c1c", "#fdba74", "#cbd5e1", "#93c5fd", "#6d28d9"]

# Everything the replay page is required to offer, by id.
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


def without_comments(script: str) -> str:
    """Strips JS comments so prose like 'the 1,010 weights' is not mistaken for a constant."""
    text = re.sub(r"/\*.*?\*/", "", script, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def check(condition: bool, label: str, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}{'  -> ' + str(detail) if detail != '' else ''}")
        FAILURES.append(label)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ids_in(html: str) -> list[str]:
    return re.findall(r'\bid="([^"]+)"', html)


def element_ids(script: str) -> list[str]:
    return re.findall(r'getElementById\("([^"]+)"\)', script)



def check_files() -> None:
    print("\n-- files --")
    for name in ("replay.html", "replay.js", "recording.js", "panels.js", "network.js"):
        check((FRONTEND / name).is_file(), f"frontend/{name} exists")

    html = read(FRONTEND / "replay.html")
    check('href="style.css"' in html, "replay.html loads the shared stylesheet")
    check(
        re.search(r'<script type="module" src="replay\.js">', html) is not None,
        "replay.html loads replay.js as an ES module",
    )
    recording_attr = re.search(r'data-recording="([^"]+)"', html)
    check(recording_attr is not None, "replay.html names its recording in data-recording")
    if recording_attr:
        check(
            recording_attr.group(1).startswith("../recordings/") and recording_attr.group(1).endswith("/"),
            "the recording path is relative and ends in a slash",
            recording_attr.group(1),
        )
    check(
        (RECORDING / "manifest.json").is_file(),
        "the documented recording exists (recordings/mnist-epoch1)",
    )

    # Every asset reference must be relative, so the page works from a static
    # server root and from a project subdirectory such as GitHub Pages.
    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    check(references, "replay.html references its assets", references)
    absolute = [ref for ref in references if ref.startswith(("/", "http://", "https://", "//"))]
    check(not absolute, "every asset reference is relative", absolute)
    for asset in ("style.css", "replay.js"):
        check(asset in references, f"replay.html references {asset}")

    script = read(FRONTEND / "replay.js")
    imports = re.findall(r'from "([^"]+)"', script)
    check(imports and all(path.startswith("./") for path in imports), "replay.js imports only sibling modules", imports)
    check(not re.search(r'(?:src|href)\s*=\s*"/', html), "no root-absolute asset path is used")

    # The page has to say out loud what a checkpoint is.
    lowered = html.lower()
    check("replay" in lowered and "training" in lowered, "the page is labelled as a replay")
    check(
        "before" in lowered and "after" in lowered and "optimiser update" in lowered,
        "the page states the pre-update / post-update capture convention",
    )


def check_html() -> None:
    print("\n-- replay.html --")
    html = read(FRONTEND / "replay.html")
    script = read(FRONTEND / "replay.js")

    found = ids_in(html)
    duplicates = sorted({name for name in found if found.count(name) > 1})
    check(not duplicates, "every element id is unique", duplicates)

    missing = [name for name in REQUIRED_IDS if f'id="{name}"' not in html]
    check(not missing, "the page provides every replay element", missing)

    looked_up = element_ids(script)
    unknown = sorted({name for name in looked_up if name not in found})
    check(not unknown, "every id replay.js looks up exists in replay.html", unknown)

    lowered = html.lower()
    check("new websocket(" not in lowered and "ws://" not in lowered, "the replay page opens no socket")
    check(("http://" not in lowered) and ("https://" not in lowered), "the replay page contacts no absolute URL")

    for name in ("btn-play", "btn-pause", "btn-prev", "btn-next", "speed-select", "timeline"):
        control = re.search(rf'<[^>]*\bid="{name}"[^>]*>', html)
        element = control.group(0) if control else ""
        check("disabled" in element, f"#{name} starts disabled", element or "missing")

    timeline = re.search(r'<input[^>]*\bid="timeline"[^>]*>', html)
    if timeline:
        tag = timeline.group(0)
        check('type="range"' in tag, "the timeline is a range input")
        check('min="0"' in tag and 'step="1"' in tag, "the timeline moves one checkpoint at a time")
        check("aria-valuetext" in tag, "the timeline exposes its real step to assistive technology")
    else:
        check(False, "the timeline exists")

    for opening, closing in (
        ("<div", "</div>"),
        ("<section", "</section>"),
        ("<aside", "</aside>"),
        ("<footer", "</footer>"),
    ):
        check(
            html.count(opening) == html.count(closing),
            f"{opening}> tags balance",
            f"{html.count(opening)} vs {html.count(closing)}",
        )



def check_replay_script() -> None:
    print("\n-- replay.js --")
    script = read(FRONTEND / "replay.js")
    lowered = script.lower()

    check('from "./recording.js"' in script and "Recording.open(" in script, "replay.js opens the recording through recording.js")
    check("RecordingError" in script, "replay.js reports recording.js's own error messages")
    check('from "./network.js"' in script and "new NetworkView(" in script, "replay.js draws with the shared NetworkView")
    check("body.dataset.recording" in script, "replay.js takes the recording path from the page")

    for helper in PANEL_HELPERS:
        imported = re.search(rf"^  {helper},$", script, re.MULTILINE) is not None
        check(imported and script.count(helper) > 1, f"replay.js imports and uses {helper}()")

    check(("websocket" not in lowered) and ("ws://" not in lowered), "replay.js opens no socket")
    check(("http://" not in lowered) and ("https://" not in lowered), "replay.js fetches no absolute URL")
    check(re.search(r"\binterpolate\(|\blerp\(", script) is None, "replay.js interpolates nothing between checkpoints")

    code = without_comments(script)
    for count in ("6001", "60,000", "109184", "109,184", "5688", "5,688", "1010", "1,010"):
        check(count not in code, f"replay.js does not hard-code {count} (the manifest owns it)")

    check("const SPEEDS = Object.freeze({" in script, "the speeds are a fixed table")
    for key in ("slow:", "normal:", "fast:"):
        check(key in script, f"the speed table defines {key[:-1]}")

    check("seekToken" in script and "token !== seekToken" in script, "a late chunk can never replace a newer selection")
    check("checkpoint.index !== target" in script, "a decoded checkpoint must be the one that was requested")
    check(".stepOf(" in script, "the timeline labels come from the recorded steps, not from the index")
    check("timeline.max = String(last)" in script, "the timeline holds checkpoint indices")
    check("No sample at Step 0" in script, "Step 0 is shown as the initial model with no sample")
    check("last recorded checkpoint" in script, "playback stops visibly at the end of the recording")
    check("prefetch(currentIndex + 1)" in script, "the next checkpoint is prefetched")
    check(
        'if (speed === "fast")' in script
        and "recording.prefetch((recording.chunkOf(currentIndex) + 1) * recording.chunkSize)" in script,
        "Fast fetches the next chunk ahead of time, because its chunks last less than a download",
    )
    check("recording.partial.count" in script, "the Partial connection count comes from the manifest")

    check(
        "initialWeights" in script and "checkpointWeights" in script,
        "Full and Partial modes use different weights",
    )
    check(
        "view.applyWeights(partial && checkpointWeights ? checkpointWeights : initialWeights)" in script,
        "the mode switch puts the right matrices in place before drawing",
    )
    check("densePartialWeights" in script, "the recorded Partial weights are scattered onto the fixed selection")


def check_shared_panels() -> None:
    print("\n-- shared panels --")
    panels = read(FRONTEND / "panels.js")
    for helper in PANEL_HELPERS:
        check(f"export function {helper}(" in panels, f"panels.js exports {helper}()")

    replay = read(FRONTEND / "replay.js")
    check('from "./panels.js"' in replay, "replay.js imports the shared panel helpers")
    imported = re.search(r'import \{([^}]*)\} from "\./panels\.js";', replay)
    names = {
        name.strip()
        for name in (imported.group(1).split(",") if imported else [])
        if name.strip()
    }
    check(
        names == set(PANEL_HELPERS),
        "replay.js imports every panel helper instead of redefining one",
        sorted(names),
    )
    for helper in PANEL_HELPERS:
        check(
            re.search(rf"^function {helper}\(", replay, re.MULTILINE) is None,
            f"replay.js keeps no private copy of {helper}()",
        )

    check("renderProgressPanel(elements" in replay, "replay.js renders progress through the shared panel")


def check_colours() -> None:
    print("\n-- weight scale --")
    manifest_file = RECORDING / "manifest.json"
    expected = list(COLOUR_STOPS)
    if manifest_file.is_file():
        manifest = json.loads(read(manifest_file))
        expected = [stop["colour"].lower() for stop in manifest["colour"]["stops"]]
        check(expected == COLOUR_STOPS, "the manifest lists the documented five stops", expected)

    network = read(FRONTEND / "network.js").lower()
    for colour in expected:
        check(colour in network, f"network.js paints {colour}")

    css = read(FRONTEND / "style.css")
    rule = re.search(r"\.scale-bar\s*\{([^}]*)\}", css)
    check(rule is not None, "style.css styles the weight scale bar")
    if rule:
        gradient = re.search(r"linear-gradient\([^)]*\)", rule.group(1))
        check(gradient is not None, "the replay legend paints the scale as a CSS gradient")
        stops = re.findall(r"#[0-9a-f]{6}", gradient.group(0).lower()) if gradient else []
        check(stops == expected, "the CSS gradient uses the renderer's stops, in order", stops)

    for name in ("timeline", "timeline-head", "timeline-ticks"):
        check(f".{name}" in css, f"style.css styles .{name}")


def decode_checkpoint(base: Path, manifest: dict, index: int) -> dict:
    """Reads one checkpoint exactly as replay.js does: manifest offsets, chunk maths."""
    chunks = manifest["format"]["chunks"]
    record_size = manifest["format"]["record_size"]
    per_chunk = chunks["chunk_size_records"]
    chunk = base / chunks["naming"].format(index=index // per_chunk)
    blob = chunk.read_bytes()
    start = (index % per_chunk) * record_size
    record = blob[start : start + record_size]
    if len(record) != record_size:
        raise ValueError(f"checkpoint {index} is short: {len(record)} bytes")

    layout = {field["name"]: field for field in manifest["format"]["record"]["fields"]}
    dtypes = {"uint32": "I", "uint8": "B", "float32": "f"}

    def read_field(name: str):
        field = layout[name]
        return struct.unpack_from(f"<{dtypes[field['dtype']] * field['count']}", record, field["offset"])

    flags = read_field("flags")[0]
    return {
        "step": read_field("step")[0],
        "has_sample": bool(flags & (1 << manifest["format"]["record"]["flags"]["has_sample_bit"])),
        "true_label": read_field("true_label")[0],
        "predicted_label": read_field("predicted_label")[0],
        "probabilities": read_field("probabilities"),
        "image": read_field("image"),
        "activations": read_field("activations"),
        "partial_weights": read_field("partial_weights"),
    }


def check_recording() -> None:
    print("\n-- recording data --")
    manifest_file = RECORDING / "manifest.json"
    if not manifest_file.is_file():
        print("SKIP  recordings/mnist-epoch1 is not present")
        return

    manifest = json.loads(read(manifest_file))
    base = RECORDING
    record = manifest["format"]["record"]
    check(manifest["format"]["record_size"] == 5_688, "records are 5,688 bytes")
    check(record["flags"]["has_sample_bit"] == 0, "flag bit 0 marks a recorded sample")
    check(
        len(manifest["activations"]["sizes"]) == len(manifest["model"]["layers"]),
        "one activation size per computed layer",
    )
    check(
        sum(manifest["activations"]["sizes"]) == manifest["activations"]["count"],
        "the activation sizes add up to the recorded activation count",
    )
    check(
        manifest["partial"]["count"]
        == sum(layer["visible_connections"] for layer in manifest["partial"]["layers"]),
        "the Partial layers add up to the recorded count",
    )
    check(manifest["full"]["biases_file"] == "model-biases.bin", "the manifest names the biases file")
    bias_bytes = (base / manifest["full"]["biases_file"]).stat().st_size
    check(
        bias_bytes == manifest["full"]["bias_count"] * 4,
        "the biases file holds bias_count float32 values",
        bias_bytes,
    )

    # The fixed Partial selection is a recording artefact: the replay must never
    # choose its own connections, so the file has to describe exactly what the
    # manifest claims.
    count = manifest["training"]["checkpoint_count"]
    indices = (base / manifest["partial"]["indices_file"]).read_bytes()
    total_indices = manifest["partial"]["count"]
    check(len(indices) == total_indices * 2, "the index file holds one uint16 per connection", len(indices))
    values = struct.unpack(f"<{total_indices}H", indices)
    okay = True
    cursor = 0
    for layer in manifest["partial"]["layers"]:
        width = layer["in"]
        rows = layer["out"]
        per_row = layer["k"]
        block = values[cursor : cursor + layer["visible_connections"]]
        cursor += layer["visible_connections"]
        okay = okay and len(block) == rows * per_row
        okay = okay and all(value < width for value in block)
        for row in range(rows):
            row_values = block[row * per_row : (row + 1) * per_row]
            okay = okay and len(set(row_values)) == len(row_values)
    check(okay, "every recorded index is unique within its neuron and inside its layer")
    check(cursor == total_indices, "the index file covers every layer once", cursor)

    # Chunk sizes are what an on-demand read assumes: full chunks, then a short
    # tail chunk holding exactly the last records.
    chunks = manifest["format"]["chunks"]
    per_chunk = chunks["chunk_size_records"]
    chunk_files = sorted((base / chunks["directory"]).glob("*.bin"))
    expected_chunks = (count + per_chunk - 1) // per_chunk
    check(len(chunk_files) == expected_chunks, "the recording has one chunk file per 500 checkpoints", len(chunk_files))
    sizes = [path.stat().st_size for path in chunk_files]
    record_size = manifest["format"]["record_size"]
    check(
        sizes[:-1] and all(size == per_chunk * record_size for size in sizes[:-1]),
        "every full chunk holds exactly 500 records",
        sizes[:2],
    )
    check(
        sizes[-1] == (count - (expected_chunks - 1) * per_chunk) * record_size,
        "the last chunk holds exactly the remaining records",
        sizes[-1],
    )

    steps = manifest["training"]["recorded_steps"]
    check(
        count == len(steps) == 6_001,
        "the recording holds 6,001 checkpoints at steps 0..60,000",
        f"{count} / {len(steps)}",
    )
    check(steps[0] == 0, "the first checkpoint is Step 0")
    check(
        steps[-1] == manifest["model"]["total_steps"] == 60_000,
        "the last checkpoint is the last optimisation step",
        steps[-1],
    )
    check(manifest["training"]["snapshot_cadence"] == 10, "checkpoints are 10 steps apart")

    first = decode_checkpoint(base, manifest, 0)
    check(first["step"] == 0, "checkpoint 0 records Step 0")
    check(not first["has_sample"], "Step 0 has no recorded sample")
    check(set(first["image"]) == {0}, "Step 0's image is empty, so the page shows none")
    check(all(value == 0.0 for value in first["probabilities"]), "Step 0 records no probabilities")
    check(len(first["partial_weights"]) == manifest["partial"]["count"], "checkpoint 0 stores every Partial weight")

    middle = decode_checkpoint(base, manifest, count // 2)
    check(middle["step"] == steps[count // 2], "the middle checkpoint decodes to its manifest step")
    check(middle["has_sample"], "later checkpoints have a recorded sample")
    check(
        abs(sum(middle["probabilities"]) - 1.0) < 1e-4,
        "the recorded probabilities are a softmax",
        sum(middle["probabilities"]),
    )
    check(
        max(range(10), key=lambda index: middle["probabilities"][index]) == middle["predicted_label"],
        "the recorded prediction is the argmax of the probabilities",
    )
    check(
        len(middle["activations"]) == sum(manifest["activations"]["sizes"]),
        "the activations match the layer sizes",
    )
    check(
        len(middle["partial_weights"]) == manifest["partial"]["count"],
        "every checkpoint stores the fixed Partial count",
    )

    last = decode_checkpoint(base, manifest, count - 1)
    check(
        last["step"] == manifest["training"]["last_step"] == 60_000,
        "the last checkpoint records Step 60,000",
        last["step"],
    )
    check(last["has_sample"], "the last checkpoint has a recorded sample")


def check_repository() -> None:
    print("\n-- repository --")
    # Only training/ is asserted here: the recorder and the frontend are sources
    # this project still develops, so unrelated working-tree changes there must
    # not fail a functional test.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "training"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    changes = [line for line in status.stdout.splitlines() if line.strip()]
    unexpected = [line for line in changes if line[3:] != "training/training.py"]
    check(not unexpected, "training/ has no working-tree changes other than training/training.py", unexpected)

    committed = subprocess.run(
        ["git", "show", "HEAD:training/training.py"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    check(committed.returncode == 0, "training/training.py is tracked by Git", committed.stderr.strip())
    working = read(REPO_ROOT / "training" / "training.py")
    check(
        ast.dump(ast.parse(committed.stdout)) == ast.dump(ast.parse(working)),
        "training/training.py's executable code is unchanged",
    )

    ignored = subprocess.run(["git", "check-ignore", "-q", "recordings/example/manifest.json"], cwd=str(REPO_ROOT))
    check(ignored.returncode == 0, "recordings/ is ignored by Git")

    readme = read(REPO_ROOT / "README.md")
    check("replay.html" in readme, "README.md documents the replay page")
    check("http.server" in readme, "README.md documents the static server")
    check("tests/test_milestone5b.py" in readme, "README.md documents this test")
    check(
        "frontend/recording.js" in readme and "frontend/panels.js" in readme,
        "README.md lists the new frontend modules",
    )


def main() -> int:
    started = time.perf_counter()
    print(f"Repository: {REPO_ROOT}")

    check_files()
    check_html()
    check_replay_script()
    check_shared_panels()
    check_colours()
    check_recording()
    check_repository()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print(f"OK: {CHECKS} checks passed in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
