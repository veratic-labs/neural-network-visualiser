"""Builds the static GitHub Pages site for the offline replay page (Milestone 5C).

Run from the repository root with the project virtualenv:

    ./venv/bin/python scripts/build_pages.py

What it does
------------
1. Reads one recording (default `recordings/mnist-epoch1/`) and validates it
   against its own manifest before publishing anything: the record layout, the
   checkpoint list, every chunk's file name, record count and byte size, every
   record's own step field, the initial full weights' sha256, the bias count and
   the fixed Partial indices.
2. Rebuilds `docs/` from scratch with only what a published site needs: the
   replay page as `index.html`, `style.css`, the four replay modules
   (`replay.js`, `recording.js`, `network.js`, `panels.js`), `.nojekyll`, and
   the recording under `docs/data/` (manifest, initial weights, biases, Partial
   indices and one file per checkpoint chunk).
3. Changes exactly two things in the page: the recording path becomes the
   relative `./data/` (so a GitHub Pages project URL works unchanged) and the
   paragraph about the live page no longer links to a page this site does not
   contain. Everything else is copied byte for byte.

`frontend/replay.html`, `frontend/`, `training/`, `recorder/`, `backend/` and
the recording itself are only read, never modified. The build is reproducible:
unchanged sources produce byte-identical output, because files are copied with
their timestamps and only the two documented page edits are applied.

A missing or inconsistent recording file stops the build with a message naming
the file, and nothing is written.

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
DEFAULT_RECORDING = REPO_ROOT / "recordings" / "mnist-epoch1"
DEFAULT_OUTPUT = REPO_ROOT / "docs"

PAGE_SOURCE = "replay.html"
PAGE_OUTPUT = "index.html"
ASSETS = ("style.css", "replay.js", "recording.js", "network.js", "panels.js")
DATA_DIRECTORY = "data"
CHUNKS_DIRECTORY = "chunks"
NOJEKYLL = ".nojekyll"

EXPECTED_FORMAT = "mlp-visualiser-recording"
EXPECTED_VERSION = 1
DTYPE_BYTES = {"uint8": 1, "uint16": 2, "uint32": 4, "float32": 4}
# Every field `recording.js` reads, in manifest (offset) order.
RECORD_FIELDS = (
    "step",
    "sample_position",
    "true_label",
    "predicted_label",
    "flags",
    "reserved",
    "image",
    "probabilities",
    "loss",
    "activations",
    "partial_weights",
)

BODY_PATTERN = re.compile(r'<body class="replay" data-recording="[^"]*">')
LIVE_MODE_PATTERN = re.compile(
    r"The live\s+mode is still <a href=\"index\.html\">index\.html</a> with the Python backend running\."
)
PAGE_REFERENCE_PATTERN = re.compile(r'(?:href|src)="([^"]+)"')


class BuildError(Exception):
    """A problem that stops the build before anything is published."""


def fail(message: str) -> None:
    raise BuildError(message)


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"{path} is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{path} cannot be read as JSON: {error}")
    if not isinstance(manifest, dict):
        fail(f"{path} does not hold a JSON object")
    return manifest


def require_file_name(value: object, label: str) -> str:
    """A plain file name inside the recording directory, with no path parts."""
    if not isinstance(value, str) or not value:
        fail(f"the manifest does not name {label}")
    if Path(value).name != value or value in (".", ".."):
        fail(f"{label} {value!r} must be a plain file name inside the recording")
    return value


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"the manifest does not declare a positive {label} (found {value!r})")
    return value


def record_layout(manifest: dict) -> tuple[dict[str, dict], int]:
    """Checks the declared record layout and returns it with the record size."""
    record_format = manifest.get("format") or {}
    if record_format.get("name") != EXPECTED_FORMAT:
        fail(
            f"the manifest format is {record_format.get('name')!r}, "
            f"expected {EXPECTED_FORMAT!r}"
        )
    if record_format.get("version") != EXPECTED_VERSION:
        fail(
            f"recording format version {record_format.get('version')!r} is not supported "
            f"(this build writes version {EXPECTED_VERSION})"
        )

    record = record_format.get("record") or {}
    size = positive_int(record.get("size_bytes"), "record size")
    if record_format.get("record_size") != size:
        fail("format.record_size disagrees with format.record.size_bytes")

    fields = record.get("fields")
    if not isinstance(fields, list) or not fields:
        fail("the manifest does not describe the record fields")

    layout: dict[str, dict] = {}
    cursor = 0
    for field in fields:
        if not isinstance(field, dict):
            fail("the manifest record fields are not objects")
        name = field.get("name")
        if name not in RECORD_FIELDS:
            fail(f"the manifest declares an unknown record field {name!r}")
        if name in layout:
            fail(f"the manifest declares record field {name!r} twice")
        dtype = field.get("dtype")
        if dtype not in DTYPE_BYTES:
            fail(f"record field {name!r} has unknown dtype {dtype!r}")
        count = positive_int(field.get("count"), f"count of record field {name!r}")
        if field.get("bytes") != DTYPE_BYTES[dtype] * count:
            fail(f"record field {name!r} declares inconsistent bytes")
        if field.get("offset") != cursor:
            fail(
                f"record field {name!r} starts at byte {field.get('offset')!r}, "
                f"expected {cursor} (fields must be contiguous and in order)"
            )
        cursor += field["bytes"]
        layout[name] = field

    missing = [name for name in RECORD_FIELDS if name not in layout]
    if missing:
        fail(f"the record layout is missing the field(s) {', '.join(missing)}")
    if cursor != size:
        fail(f"the record fields cover {cursor} bytes, the manifest says {size}")
    return layout, size


def check_model(manifest: dict, layout: dict[str, dict]) -> tuple[list[dict], int, int]:
    """Checks the layer table against the record layout; returns the layer facts."""
    model = manifest.get("model") or {}
    layers = model.get("layers")
    if not isinstance(layers, list) or not layers:
        fail("the manifest does not describe the model layers")

    connections = 0
    biases = 0
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            fail("the manifest model layers are not objects")
        if layer.get("index") != index:
            fail(f"model layer {index} is declared with index {layer.get('index')!r}")
        inputs = positive_int(layer.get("in"), f"input count of layer {index}")
        outputs = positive_int(layer.get("out"), f"output count of layer {index}")
        if layer.get("weight_count") != inputs * outputs:
            fail(f"model layer {index} declares an inconsistent weight_count")
        if layer.get("bias_count") != outputs:
            fail(f"model layer {index} declares an inconsistent bias_count")
        connections += inputs * outputs
        biases += outputs

    if model.get("connections") != connections:
        fail(
            f"the manifest says {model.get('connections')!r} connections, "
            f"the layers hold {connections}"
        )
    if model.get("biases") != biases:
        fail(f"the manifest says {model.get('biases')!r} biases, the layers hold {biases}")

    sizes = [layers[0]["in"]] + [layer["out"] for layer in layers]
    architecture = model.get("architecture")
    if not isinstance(architecture, list) or [
        entry.get("size") for entry in architecture
    ] != sizes:
        fail("model.architecture does not match the layer sizes")

    activations = (manifest.get("activations") or {}).get("sizes")
    if not isinstance(activations, list) or activations != sizes[1:]:
        fail("activations.sizes does not match the computed layer sizes")
    if layout["activations"]["count"] != sum(sizes[1:]):
        fail("the record's activation count does not match the layer sizes")
    if layout["probabilities"]["count"] != sizes[-1]:
        fail("the record's probability count does not match the output layer size")

    images = manifest.get("images") or {}
    width = positive_int(images.get("width"), "image width")
    height = positive_int(images.get("height"), "image height")
    if layout["image"]["count"] != width * height:
        fail("the record's image count does not match the declared image size")

    return layers, connections, biases


def check_checkpoints(manifest: dict, record_size: int) -> tuple[list[int], dict, dict]:
    """Checks the checkpoint schedule; returns steps, training and checkpoints."""
    training = manifest.get("training") or {}
    checkpoints = manifest.get("checkpoints") or {}

    steps = training.get("recorded_steps")
    if not isinstance(steps, list) or not steps:
        fail("the manifest does not list the recorded steps")
    if any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in steps):
        fail("the recorded steps are not all non-negative integers")

    cadence = positive_int(training.get("snapshot_cadence"), "snapshot cadence")
    count = len(steps)
    if steps[0] != 0:
        fail(f"the first recorded step is {steps[0]}, expected 0")
    for position in range(1, count):
        delta = steps[position] - steps[position - 1]
        if delta <= 0:
            fail(f"recorded step {steps[position]} does not follow step {steps[position - 1]}")
        if delta > cadence:
            fail(
                f"recorded steps jump from {steps[position - 1]} to {steps[position]}, "
                f"more than the declared cadence of {cadence}"
            )
        if delta != cadence and position != count - 1:
            fail(
                f"recorded steps are not every {cadence} steps: "
                f"{steps[position - 1]} -> {steps[position]}"
            )

    if checkpoints.get("count") != count:
        fail(f"checkpoints.count is {checkpoints.get('count')!r}, the step list holds {count}")
    if training.get("checkpoint_count") != count:
        fail("training.checkpoint_count disagrees with the recorded step list")
    if checkpoints.get("first_step") != steps[0] or training.get("first_step") != steps[0]:
        fail("the manifest's first step disagrees with the recorded step list")
    if checkpoints.get("last_step") != steps[-1] or training.get("last_step") != steps[-1]:
        fail("the manifest's last step disagrees with the recorded step list")
    if training.get("steps_recorded") is not None and training["steps_recorded"] < steps[-1]:
        fail("training.steps_recorded is earlier than the last recorded step")
    if (manifest.get("model") or {}).get("total_steps") != steps[-1]:
        fail("model.total_steps disagrees with the last recorded step")
    if checkpoints.get("record_size") != record_size:
        fail("checkpoints.record_size disagrees with the record layout")
    if checkpoints.get("total_bytes") != count * record_size:
        fail("checkpoints.total_bytes does not match the checkpoint count and record size")
    return steps, training, checkpoints


def check_chunks(
    recording: Path, manifest: dict, record_size: int, steps: list[int]
) -> tuple[list[dict], int]:
    """Checks every chunk's declared name, count, size and step range."""
    checkpoints = manifest.get("checkpoints") or {}
    chunks_info = (manifest.get("format") or {}).get("chunks") or {}
    chunk_size = positive_int(chunks_info.get("chunk_size_records"), "chunk size")
    naming = chunks_info.get("naming")
    directory = chunks_info.get("directory")
    if not isinstance(naming, str) or "{index" not in naming:
        fail("the manifest does not name its chunk files with an {index...} pattern")
    if directory != CHUNKS_DIRECTORY:
        fail(f"the manifest's chunk directory is {directory!r}, expected {CHUNKS_DIRECTORY!r}")

    entries = checkpoints.get("chunks")
    expected_count = (len(steps) + chunk_size - 1) // chunk_size
    if not isinstance(entries, list) or len(entries) != expected_count:
        declared = len(entries) if isinstance(entries, list) else "no"
        fail(
            f"the manifest lists {declared} chunks, "
            f"{expected_count} are needed for {len(steps)} checkpoints"
        )

    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail("the manifest chunk entries are not objects")
        try:
            expected_file = naming.format(index=position)
        except (KeyError, IndexError, ValueError) as error:
            fail(f"the chunk naming pattern cannot be expanded: {error}")
        if Path(expected_file).parent.as_posix() != CHUNKS_DIRECTORY:
            fail(f"chunk file {expected_file!r} is not inside {CHUNKS_DIRECTORY}/")
        expected_records = min(chunk_size, len(steps) - position * chunk_size)
        first = steps[position * chunk_size]
        last = steps[position * chunk_size + expected_records - 1]

        if entry.get("index") != position:
            fail(f"chunk {position} is declared with index {entry.get('index')!r}")
        if entry.get("file") != expected_file:
            fail(
                f"chunk {position} is declared as {entry.get('file')!r}, "
                f"expected {expected_file!r}"
            )
        if entry.get("count") != expected_records:
            fail(
                f"chunk {position} is declared with {entry.get('count')!r} records, "
                f"expected {expected_records}"
            )
        if entry.get("first_step") != first or entry.get("last_step") != last:
            fail(
                f"chunk {position} declares steps {entry.get('first_step')!r}.."
                f"{entry.get('last_step')!r}, expected {first}..{last}"
            )
        if entry.get("bytes") != expected_records * record_size:
            fail(f"chunk {position} declares inconsistent bytes")

        path = recording / expected_file
        if not path.is_file():
            fail(f"{path} is missing")
        actual = path.stat().st_size
        if actual != entry["bytes"]:
            fail(f"{path} is {actual} bytes, the manifest says {entry['bytes']}")

    return entries, chunk_size


def check_initial_files(
    recording: Path, manifest: dict, connections: int, biases: int
) -> tuple[str, str]:
    """Checks the Step 0 weights and biases files against the layer table."""
    full = manifest.get("full") or {}
    weights_file = require_file_name(full.get("weights_file"), "the initial weights file")
    biases_file = require_file_name(full.get("biases_file"), "the biases file")
    if full.get("connection_count") != connections:
        fail("full.connection_count disagrees with the layer table")
    if full.get("bias_count") != biases:
        fail("full.bias_count disagrees with the layer table")
    if full.get("later_full_weights_recorded", False):
        fail(
            "this recording contains later full-network weights, but the published page "
            "only draws the Step 0 ones; refusing to publish a misleading site"
        )

    for name in (weights_file, biases_file):
        path = recording / name
        if not path.is_file():
            fail(f"{path} is missing")
    weights_bytes = (recording / weights_file).stat().st_size
    if weights_bytes != connections * 4:
        fail(f"{weights_file} is {weights_bytes} bytes, expected {connections * 4}")
    bias_bytes = (recording / biases_file).stat().st_size
    if bias_bytes != biases * 4:
        fail(f"{biases_file} is {bias_bytes} bytes, expected {biases * 4}")
    return weights_file, biases_file


def check_partial(
    recording: Path, manifest: dict, layout: dict[str, dict], layers: list[dict]
) -> int:
    """Checks the fixed Partial selection: count, layer ranges and index values."""
    partial = manifest.get("partial") or {}
    count = positive_int(partial.get("count"), "Partial weight count")
    top_k = positive_int(partial.get("top_k"), "Partial top-k")
    indices_file = require_file_name(partial.get("indices_file"), "the Partial indices file")
    if layout["partial_weights"]["count"] != count:
        fail("the record's Partial weight count disagrees with partial.count")

    path = recording / indices_file
    if not path.is_file():
        fail(f"{path} is missing")
    data = path.read_bytes()
    if len(data) != count * 2:
        fail(f"{indices_file} is {len(data)} bytes, expected {count * 2} uint16 indices")

    partial_layers = partial.get("layers")
    if not isinstance(partial_layers, list) or len(partial_layers) != len(layers):
        fail("partial.layers does not describe every model layer")
    cursor = 0
    for layer, model_layer in zip(partial_layers, layers):
        index = model_layer["index"]
        if layer.get("index") != index:
            fail(f"partial layer {index} is declared with index {layer.get('index')!r}")
        if layer.get("in") != model_layer["in"] or layer.get("out") != model_layer["out"]:
            fail(f"partial layer {index} does not match the model layer shape")
        if layer.get("k") != top_k:
            fail(f"partial layer {index} uses k = {layer.get('k')!r}, expected {top_k}")
        if layer.get("visible_connections") != model_layer["out"] * top_k:
            fail(f"partial layer {index} declares inconsistent visible_connections")

        for output in range(model_layer["out"]):
            row = [
                struct.unpack_from("<H", data, (cursor + k) * 2)[0] for k in range(top_k)
            ]
            if any(value >= model_layer["in"] for value in row):
                fail(f"partial layer {index} output {output} names an input outside the layer")
            if len(set(row)) != top_k:
                fail(f"partial layer {index} output {output} repeats an input index")
            cursor += top_k

    if cursor != count:
        fail(f"{indices_file} describes {cursor} indices, partial.count says {count}")
    return count


def check_colour(manifest: dict) -> list[str]:
    """The renderer and the page must use the fixed ±0.5 scale the manifest records."""
    colour = manifest.get("colour") or {}
    if colour.get("min") != -0.5 or colour.get("max") != 0.5:
        fail("the manifest's colour scale is not the fixed ±0.5 scale this page draws")
    stops = colour.get("stops")
    expected = (
        ("-0.5", "#b91c1c"),
        ("-0.25", "#fdba74"),
        ("0.0", "#cbd5e1"),
        ("0.25", "#93c5fd"),
        ("0.5", "#6d28d9"),
    )
    if not isinstance(stops, list) or len(stops) != len(expected):
        fail("the manifest does not list the five colour stops the page shows")
    for stop, (weight, value) in zip(stops, expected):
        if str(stop.get("weight")) != weight or stop.get("colour") != value:
            fail(
                f"colour stop {stop.get('weight')!r} is {stop.get('colour')!r}, "
                f"expected {value}"
            )
    return [stop["colour"] for stop in stops]


def check_initial_checksum(
    recording: Path,
    manifest: dict,
    layers: list[dict],
    weights_file: str,
    biases_file: str,
) -> str | None:
    """Re-computes the manifest's initial-weight sha256 in the recorder's order."""
    expected = (manifest.get("model") or {}).get("initial_weights_sha256")
    if not expected:
        return None

    weights = (recording / weights_file).read_bytes()
    biases = (recording / biases_file).read_bytes()
    digest = hashlib.sha256()
    weight_offset = 0
    bias_offset = 0
    for layer in layers:
        weight_bytes = layer["weight_count"] * 4
        bias_bytes = layer["bias_count"] * 4
        digest.update(weights[weight_offset : weight_offset + weight_bytes])
        weight_offset += weight_bytes
        digest.update(biases[bias_offset : bias_offset + bias_bytes])
        bias_offset += bias_bytes
    if weight_offset != len(weights) or bias_offset != len(biases):
        fail("the initial weights and biases do not match the layer table")
    actual = digest.hexdigest()
    if actual != expected:
        fail(f"the initial weights hash to {actual}, the manifest records {expected}")
    return actual


def check_records(
    recording: Path, entries: list[dict], record_size: int, steps: list[int], layout: dict
) -> None:
    """Reads every record's own step field, so the timeline cannot point at nothing."""
    step_offset = layout["step"]["offset"]
    cursor = 0
    for entry in entries:
        data = (recording / entry["file"]).read_bytes()
        if len(data) != entry["bytes"]:
            fail(f"{entry['file']} changed size while the build was running")
        for record in range(entry["count"]):
            recorded = struct.unpack_from("<I", data, record * record_size + step_offset)[0]
            if recorded != steps[cursor + record]:
                fail(
                    f"{entry['file']} record {record} stores step {recorded}, "
                    f"the manifest says {steps[cursor + record]}"
                )
        cursor += entry["count"]
    if cursor != len(steps):
        fail(f"the chunks hold {cursor} records, the manifest says {len(steps)}")


def build_page(source_text: str, run_name: str) -> str:
    """The replay page with its two documented changes for a published copy."""
    body, body_count = BODY_PATTERN.subn(
        f'<body class="replay" data-recording="./{DATA_DIRECTORY}/" '
        f'data-run-name="{html.escape(run_name, quote=True)}">',
        source_text,
    )
    if body_count != 1:
        fail(
            "frontend/replay.html no longer has exactly one "
            '<body class="replay" data-recording="..."> tag'
        )

    page, note_count = LIVE_MODE_PATTERN.subn(
        "The live training page is not part of this published site.",
        body,
    )
    if note_count != 1:
        fail("frontend/replay.html no longer has exactly one sentence about the live page")
    if 'href="index.html"' in page:
        fail("the built page still links to index.html, which is the page itself")
    return page


def check_page_references(page_path: Path, output: Path) -> list[str]:
    """Every href/src in the built page must be a relative path that exists."""
    text = page_path.read_text(encoding="utf-8")
    references = PAGE_REFERENCE_PATTERN.findall(text)
    if not references:
        fail(f"{PAGE_OUTPUT} does not reference any local file")
    for reference in references:
        if reference.startswith(("http://", "https://", "//", "/", "data:", "#")):
            fail(f"{PAGE_OUTPUT} refers to {reference!r}; published paths must be relative")
        target = (page_path.parent / reference).resolve()
        try:
            target.relative_to(output)
        except ValueError:
            fail(f"{PAGE_OUTPUT} refers to {reference!r}, which is outside the site")
        if not target.is_file():
            fail(f"{PAGE_OUTPUT} refers to {reference!r}, which does not exist")
    return references


def prepare_output(output: Path) -> Path:
    """Empties the output directory, refusing any directory that holds sources."""
    output = output.resolve()
    if output == REPO_ROOT.resolve():
        fail("refusing to rebuild the repository root")
    protected = [
        FRONTEND.resolve(),
        (REPO_ROOT / "training").resolve(),
        (REPO_ROOT / "backend").resolve(),
        (REPO_ROOT / "recorder").resolve(),
        DEFAULT_RECORDING.resolve(),
    ]
    for source in protected:
        if output == source or source in output.parents:
            fail(f"refusing to rebuild {output}: it holds source files")
    if output.exists() and not output.is_dir():
        fail(f"{output} exists and is not a directory")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def build(recording: Path, output: Path) -> dict:
    """Validates `recording` and writes the publishable site to `output`."""
    recording = recording.resolve()
    if not recording.is_dir():
        fail(f"the recording directory {recording} does not exist")

    manifest = read_json(recording / "manifest.json")
    layout, record_size = record_layout(manifest)
    layers, connections, biases = check_model(manifest, layout)
    steps, training, _ = check_checkpoints(manifest, record_size)
    entries, chunk_size = check_chunks(recording, manifest, record_size, steps)
    weights_file, biases_file = check_initial_files(recording, manifest, connections, biases)
    partial_count = check_partial(recording, manifest, layout, layers)
    colour_stops = check_colour(manifest)
    checksum = check_initial_checksum(recording, manifest, layers, weights_file, biases_file)
    check_records(recording, entries, record_size, steps, layout)

    output = prepare_output(output)

    for name in ASSETS:
        source = FRONTEND / name
        if not source.is_file():
            fail(f"{source} is missing")
        shutil.copy2(source, output / name)

    page_source = FRONTEND / PAGE_SOURCE
    if not page_source.is_file():
        fail(f"{page_source} is missing")
    page_path = output / PAGE_OUTPUT
    page_path.write_text(
        build_page(page_source.read_text(encoding="utf-8"), recording.name),
        encoding="utf-8",
    )
    page_stat = page_source.stat()
    os.utime(page_path, ns=(page_stat.st_atime_ns, page_stat.st_mtime_ns))

    # GitHub Pages runs Jekyll by default; this site is plain static files.
    (output / NOJEKYLL).write_bytes(b"")

    data_directory = output / DATA_DIRECTORY
    data_directory.mkdir()
    partial_indices_file = require_file_name(
        (manifest.get("partial") or {}).get("indices_file"), "the Partial indices file"
    )
    for name in ("manifest.json", weights_file, biases_file, partial_indices_file):
        shutil.copy2(recording / name, data_directory / name)

    chunks_directory = data_directory / CHUNKS_DIRECTORY
    chunks_directory.mkdir()
    for entry in entries:
        shutil.copy2(recording / entry["file"], chunks_directory / Path(entry["file"]).name)

    references = check_page_references(page_path, output)
    files = sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    )
    return {
        "recording": recording.name,
        "checkpoints": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "cadence": training["snapshot_cadence"],
        "chunks": len(entries),
        "chunk_size": chunk_size,
        "record_size": record_size,
        "connections": connections,
        "biases": biases,
        "partial": partial_count,
        "checksum": checksum,
        "colours": colour_stops,
        "references": references,
        "files": files,
        "bytes": sum((output / name).stat().st_size for name in files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the static GitHub Pages site into docs/ (Milestone 5C)."
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=DEFAULT_RECORDING,
        help="recording directory to publish (default: recordings/mnist-epoch1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="site directory to rebuild (default: docs)",
    )
    arguments = parser.parse_args(argv)

    try:
        summary = build(arguments.recording, arguments.output)
    except BuildError as error:
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        return 1

    output = arguments.output.resolve()
    print(f"Static site built: {output}")
    print(f"  recording   {summary['recording']}")
    print(
        f"  checkpoints {summary['checkpoints']:,} "
        f"(step {summary['first_step']:,} … {summary['last_step']:,}, "
        f"every {summary['cadence']:,} steps)"
    )
    print(
        f"  chunks      {summary['chunks']:,} × {summary['chunk_size']:,} records "
        f"({summary['record_size']:,} bytes each)"
    )
    print(
        f"  network     {summary['connections']:,} connections, "
        f"{summary['biases']:,} biases, {summary['partial']:,} Partial weights"
    )
    print(f"  checksum    {summary['checksum'] or 'not declared in the manifest'}")
    print(f"  colours     {' '.join(summary['colours'])}")
    print(f"  page refs   {', '.join(summary['references'])}")
    print(f"  files       {len(summary['files'])} ({summary['bytes'] / 1024 / 1024:.2f} MiB)")
    print("Serve it with a static server and open docs/index.html, for example:")
    print("  ./venv/bin/python -m http.server 8200 --bind 127.0.0.1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
