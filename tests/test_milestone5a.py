"""Milestone 5A verification: the standalone training recorder.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone5a.py

Checks, in order:
  1. the documented command records 100 optimiser steps into 11 checkpoints at
     steps 0, 10, 20, ..., 100, and writes the manifest last
  2. every file has the documented size: 5,688-byte records, 109,184 initial
     weights, 202 biases and 1,010 uint16 connection indices
  3. the manifest describes its own data: architecture, record offsets and
     dtypes, chunk layout, cadence, colour scale and capture convention
  4. the 1,010 Partial indices are the top-5 incoming |weight| of the initial
     model, are written once, and every checkpoint stores values at those indices
  5. the recorded numbers are the real PyTorch values: the initial full weights
     match the model, Step 0 carries no sample, and for every checkpoint the
     image, label, prediction, probabilities, loss and activations are the
     pre-update forward pass while the Partial weights are the post-update
     parameters
  6. the final step is recorded even when it is not a multiple of the cadence
  7. no file in training/ changes, and the training maths is untouched
  8. recordings/ is ignored by Git and README.md documents how to record

A short 100-step run is used; plan.md records the full 60,000-step run as a
manual verification because it takes minutes.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKS = 0
FAILURES: list[str] = []

# What training/training.py really defines.
INPUT_SIZE = 784
LAYER_SIZES = (128, 64, 10)
LAYERS = [(LAYER_SIZES[0], INPUT_SIZE), (LAYER_SIZES[1], LAYER_SIZES[0]), (LAYER_SIZES[2], LAYER_SIZES[1])]  # (out, in), in file order

TRAIN_SIZE = 60_000
PARTIAL_TOP_K = 5
PARTIAL_COUNT = 1_010
CONNECTION_COUNT = 109_184
BIAS_COUNT = 202

# The documented record layout (plan.md section 7.4).
RECORD_SIZE = 5_688
FIELD_LAYOUT = [
    ("step", 0, 4, "uint32", 1),
    ("sample_position", 4, 4, "uint32", 1),
    ("true_label", 8, 1, "uint8", 1),
    ("predicted_label", 9, 1, "uint8", 1),
    ("flags", 10, 1, "uint8", 1),
    ("reserved", 11, 1, "uint8", 1),
    ("image", 12, 784, "uint8", 784),
    ("probabilities", 796, 40, "float32", 10),
    ("loss", 836, 4, "float32", 1),
    ("activations", 840, 808, "float32", 202),
    ("partial_weights", 1648, 4_040, "float32", 1_010),
]

SEED = 20261104
STEPS = 100
CADENCE = 10
COLOR_STOPS = ["#b91c1c", "#fdba74", "#cbd5e1", "#93c5fd", "#6d28d9"]


def check(condition: bool, label: str, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}{'  -> ' + str(detail) if detail != '' else ''}")
        FAILURES.append(label)


def decode_record(blob: bytes) -> dict:
    """One checkpoint record, decoded the way the replay will decode it."""
    if len(blob) != RECORD_SIZE:
        raise ValueError(f"record is {len(blob)} bytes, expected {RECORD_SIZE}")
    step, sample_position = struct.unpack_from("<II", blob, 0)
    return {
        "step": step,
        "sample_position": sample_position,
        "true_label": blob[8],
        "predicted_label": blob[9],
        "flags": blob[10],
        "reserved": blob[11],
        "image": np.frombuffer(blob[12:796], dtype=np.uint8),
        "probabilities": np.frombuffer(blob[796:836], dtype="<f4"),
        "loss": struct.unpack_from("<f", blob, 836)[0],
        "activations": np.frombuffer(blob[840:1_648], dtype="<f4"),
        "partial_weights": np.frombuffer(blob[1_648:5_688], dtype="<f4"),
    }


class Recording:
    """One recording directory, read through its manifest only."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.chunk_bytes = [
            (self.root / entry["file"]).read_bytes()
            for entry in self.manifest["checkpoints"]["chunks"]
        ]
        self.blob = b"".join(self.chunk_bytes)
        self.indices = np.frombuffer((self.root / "partial-indices.bin").read_bytes(), dtype="<u2")

    @property
    def records(self) -> list[dict]:
        return [
            decode_record(self.blob[index * RECORD_SIZE : (index + 1) * RECORD_SIZE])
            for index in range(len(self.blob) // RECORD_SIZE)
        ]

    def record_bytes(self, index: int) -> bytes:
        return self.blob[index * RECORD_SIZE : (index + 1) * RECORD_SIZE]


def run_recorder(output: Path, steps: int | None = None, seed: int | None = None) -> subprocess.CompletedProcess:
    """The documented command, run exactly as the README shows it."""
    command = [
        sys.executable,
        "-m",
        "recorder.record_training",
        "--output",
        str(output),
    ]
    if steps is not None:
        command += ["--steps", str(steps)]
    if seed is not None:
        command += ["--seed", str(seed)]
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )



def check_files(recording: Recording) -> None:
    root = recording.root
    check((root / "model.bin").stat().st_size == CONNECTION_COUNT * 4, "model.bin is 109,184 float32 initial weights", (root / "model.bin").stat().st_size)
    check((root / "model-biases.bin").stat().st_size == BIAS_COUNT * 4, "model-biases.bin is 202 float32 initial biases", (root / "model-biases.bin").stat().st_size)
    check((root / "partial-indices.bin").stat().st_size == PARTIAL_COUNT * 2, "partial-indices.bin is 1,010 uint16 indices", (root / "partial-indices.bin").stat().st_size)

    names = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    chunk_names = {name for name in names if re.fullmatch(r"chunks/chunk-\d{4}\.bin", name)}
    documented = {"manifest.json", "model.bin", "model-biases.bin", "partial-indices.bin"}
    check(names - chunk_names == documented, "the recording holds exactly the documented files", sorted(names - chunk_names))
    check(len(chunk_names) == len(recording.manifest["checkpoints"]["chunks"]), "every chunk file is listed in the manifest", sorted(chunk_names))
    check(all(len(data) % RECORD_SIZE == 0 for data in recording.chunk_bytes), "every chunk is a whole number of 5,688-byte records")


def check_chunks(recording: Recording) -> None:
    checkpoints = recording.manifest["checkpoints"]
    chunks = checkpoints["chunks"]
    check(len(chunks) == len(recording.chunk_bytes), "manifest chunk count matches the chunk files", (len(chunks), len(recording.chunk_bytes)))

    total = 0
    for entry, data in zip(chunks, recording.chunk_bytes):
        index = entry["index"]
        check(entry["file"] == f"chunks/chunk-{index:04d}.bin", f"chunk {index} has the documented name", entry["file"])
        check(len(data) == entry["count"] * RECORD_SIZE, f"chunk {index} is {entry['count']} records x {RECORD_SIZE} bytes", len(data))
        check(entry["bytes"] == len(data), f"chunk {index} byte count matches the manifest", entry["bytes"])
        steps = [
            decode_record(data[offset * RECORD_SIZE : (offset + 1) * RECORD_SIZE])["step"]
            for offset in range(entry["count"])
        ]
        check(steps[0] == entry["first_step"] and steps[-1] == entry["last_step"], f"chunk {index} spans steps {entry['first_step']}-{entry['last_step']}", (steps[0], steps[-1]))
        check(all(a < b for a, b in zip(steps, steps[1:])), f"chunk {index} steps strictly increase")
        total += entry["count"]

    check(total == checkpoints["count"] == len(recording.records), "the chunks hold the declared checkpoint count", (total, checkpoints["count"]))
    check(checkpoints["total_bytes"] == total * RECORD_SIZE, "the manifest's checkpoint byte total is count x record size", checkpoints["total_bytes"])
    check(checkpoints["record_size"] == RECORD_SIZE, "the manifest's checkpoint record size is 5,688", checkpoints["record_size"])
    check(recording.manifest["format"]["chunks"]["chunk_size_records"] == 500, "chunks hold 500 records by default", recording.manifest["format"]["chunks"]["chunk_size_records"])


def check_manifest(recording: Recording) -> None:
    manifest = recording.manifest
    sections = {"format", "model", "training", "checkpoints", "partial", "full", "activations", "images", "capture", "colour", "notes"}
    check(set(manifest) == sections, "the manifest has the documented sections", sorted(manifest))

    fmt = manifest["format"]
    check(fmt["name"] == "mlp-visualiser-recording" and fmt["version"] == 1, "the manifest names and versions the format", (fmt["name"], fmt["version"]))
    check(fmt["byte_order"] == "little-endian", "the recorded float32 values are little-endian", fmt["byte_order"])
    check(fmt["record_size"] == RECORD_SIZE == fmt["record"]["size_bytes"], "the format states the 5,688-byte stride", fmt["record_size"])
    fields = [(field["name"], field["offset"], field["bytes"], field["dtype"], field["count"]) for field in fmt["record"]["fields"]]
    check(fields == FIELD_LAYOUT, "record fields match the documented offsets, dtypes and counts", fields)
    floats = [field for field in fmt["record"]["fields"] if field["dtype"] == "float32"]
    check(all(field["offset"] % 4 == 0 for field in floats), "every float32 field starts on a 4-byte boundary", [(f["name"], f["offset"]) for f in floats])
    check(fmt["record"]["flags"]["has_sample_bit"] == 0, "the manifest documents the has_sample flag bit", fmt["record"]["flags"])

    model = manifest["model"]
    sizes = [layer["size"] for layer in model["architecture"]]
    check(sizes == [INPUT_SIZE, *LAYER_SIZES], "the manifest records the 784-128-64-10 architecture", sizes)
    shapes = [(layer["in"], layer["out"]) for layer in model["layers"]]
    check(shapes == [(784, 128), (128, 64), (64, 10)], "the manifest records every layer's shape", shapes)
    check(model["connections"] == CONNECTION_COUNT and model["biases"] == BIAS_COUNT, "the manifest counts all 109,184 weights and 202 biases", (model["connections"], model["biases"]))
    check(model["batch_size"] == 1, "the manifest records batch size 1", model["batch_size"])
    check(model["total_steps"] == TRAIN_SIZE, "the manifest records one epoch of 60,000 steps", model["total_steps"])
    check(model["optimiser"] == "torch.optim.Adam" and model["learning_rate"] == 0.001, "the manifest records the unchanged Adam(0.001) optimiser", (model["optimiser"], model["learning_rate"]))
    check(model["loss"] == "nn.CrossEntropyLoss", "the manifest records the unchanged loss", model["loss"])
    check(re.fullmatch(r"[0-9a-f]{64}", model["initial_weights_sha256"]) is not None, "the manifest stores the initial weight checksum", model["initial_weights_sha256"])

    training = manifest["training"]
    check(training["recorded_steps"] == list(range(0, STEPS + 1, CADENCE)), "the manifest lists steps 0, 10, ..., 100", training["recorded_steps"])
    check(training["snapshot_cadence"] == CADENCE, "the manifest records the 10-step cadence", training["snapshot_cadence"])
    span = (training["first_step"], training["last_step"], training["checkpoint_count"])
    check(span == (0, STEPS, 11), "the manifest's step range matches the run", span)
    check(training["steps_recorded"] == STEPS, "the manifest records the number of completed steps", training["steps_recorded"])
    check(training["complete_epoch"] is False, "a 100-step run is marked as not a complete epoch", training["complete_epoch"])
    check(training["seed"] == SEED, "the manifest records the seed it was given", training["seed"])
    check(training["duration_seconds"] > 0, "the manifest records the recording duration", training["duration_seconds"])
    environment = {key: training[key] for key in ("python", "pytorch", "numpy", "platform")}
    check(all(environment.values()), "the manifest records the environment it was produced in", environment)

    partial = manifest["partial"]
    check(partial["top_k"] == PARTIAL_TOP_K and partial["count"] == PARTIAL_COUNT, "the manifest records the top-5, 1,010-connection selection", (partial["top_k"], partial["count"]))
    check(partial["indices_file"] == "partial-indices.bin" and partial["indices_dtype"] == "uint16", "the manifest documents the index file and dtype", (partial["indices_file"], partial["indices_dtype"]))
    visible = [layer["visible_connections"] for layer in partial["layers"]]
    check(visible == [640, 320, 50], "the manifest counts 640 + 320 + 50 visible connections", visible)
    check("fixed" in partial["selection_rule"], "the manifest states the indices are fixed for the recording", partial["selection_rule"])

    full = manifest["full"]
    check(full["weights_file"] == "model.bin" and full["connection_count"] == CONNECTION_COUNT, "the manifest points at the one initial full-weight file", (full["weights_file"], full["connection_count"]))
    check(full["biases_file"] == "model-biases.bin" and full["bias_count"] == BIAS_COUNT, "the manifest points at the initial biases", (full["biases_file"], full["bias_count"]))
    check(full["later_full_weights_recorded"] is False, "the manifest states that no later full weights exist", full["later_full_weights_recorded"])

    activations = manifest["activations"]
    check(activations["sizes"] == list(LAYER_SIZES) and activations["count"] == BIAS_COUNT, "the manifest describes the 128 + 64 + 10 activations", activations["sizes"])
    check(activations["capture"] == "pre-update", "the manifest states activations are pre-update", activations["capture"])
    check(manifest["images"] == {"width": 28, "height": 28, "dtype": "uint8", "range": [0, 255], "layout": "row-major"}, "the manifest describes the 28 x 28 uint8 images", manifest["images"])

    capture = manifest["capture"]
    check(capture["prediction"] == "pre-update" and capture["probabilities"] == "pre-update", "the manifest states the prediction is pre-update", (capture["prediction"], capture["probabilities"]))
    check(capture["loss"] == "pre-update", "the manifest states the loss is pre-update", capture["loss"])
    check(capture["partial_weights"] == "post-update", "the manifest states the Partial weights are post-update", capture["partial_weights"])
    check("no sample" in capture["step_0"], "the manifest states Step 0 has no sample", capture["step_0"])

    colour = manifest["colour"]
    check((colour["min"], colour["max"]) == (-0.5, 0.5), "the manifest records the fixed -0.5 .. 0.5 colour scale", (colour["min"], colour["max"]))
    stops = [stop["colour"] for stop in colour["stops"]]
    check(stops == COLOR_STOPS, "the manifest records the frontend's five colour stops", stops)
    weights = [stop["weight"] for stop in colour["stops"]]
    check(weights == [-0.5, -0.25, 0.0, 0.25, 0.5], "the colour stops sit at the frontend's weights", weights)

    notes = manifest["notes"]
    command = notes["command"]
    check("recorder.record_training" in command and f"--steps {STEPS}" in command, "the manifest records the command that produced it", command)
    check(len(notes["limitations"]) >= 4, "the manifest lists its limitations", notes["limitations"])


def check_records(recording: Recording) -> None:
    records = recording.records
    steps = [record["step"] for record in records]
    check(steps == list(range(0, STEPS + 1, CADENCE)), "the records are steps 0, 10, ..., 100 in order", steps)

    initial = records[0]
    check((initial["flags"] & 1) == 0, "Step 0 is flagged as having no sample", initial["flags"])
    check(initial["sample_position"] == 0, "Step 0 stores no sample position", initial["sample_position"])
    check(initial["true_label"] == 0 and initial["predicted_label"] == 0, "Step 0 stores no label or prediction", (initial["true_label"], initial["predicted_label"]))
    check(not initial["image"].any(), "Step 0 stores a blank image")
    check(not initial["probabilities"].any(), "Step 0 stores zero probabilities")
    check(initial["loss"] == 0.0, "Step 0 stores no loss", initial["loss"])
    check(not initial["activations"].any(), "Step 0 stores zero activations")

    later = records[1:]
    check(all(record["flags"] & 1 for record in later), "every later record is flagged as having a sample", [r["flags"] for r in later])
    check(all(record["reserved"] == 0 for record in later), "the reserved byte stays zero in every record")
    check(all(record["true_label"] < 10 for record in later), "every record stores a digit label", [r["true_label"] for r in later])
    positions = [record["sample_position"] for record in later]
    check(all(position == (record["step"] - 1) % TRAIN_SIZE + 1 for record, position in zip(later, positions)), "every record stores its 1-based sample position", positions)
    check(all(record["image"].any() for record in later), "every record stores a non-blank image")
    check(len(set(positions)) == len(positions), "the shuffled loader gives each checkpoint a different sample", positions)

    sums = [float(record["probabilities"].sum()) for record in later]
    check(all(abs(total - 1.0) < 1e-5 for total in sums), "every record's probabilities sum to 1", sums)
    check(all(record["probabilities"].min() >= 0 and record["probabilities"].max() <= 1 for record in later), "every recorded probability is in [0, 1]")
    check(all(record["predicted_label"] == int(record["probabilities"].argmax()) for record in later), "every prediction is the record's most probable digit", [(r["predicted_label"], int(r["probabilities"].argmax())) for r in later])
    losses = [record["loss"] for record in later]
    check(all(loss >= 0 for loss in losses), "every recorded loss is non-negative", losses)
    check(all((record["activations"][:192] >= 0).all() for record in later), "the 192 hidden activations are ReLU outputs", [float(r["activations"][:192].min()) for r in later])
    check(all(np.allclose(record["activations"][192:], record["probabilities"], atol=1e-6) for record in later), "the 10 output activations equal the probabilities")

    check(float(np.mean(losses[-3:])) < float(np.mean(losses[:3])), "100 seeded Adam steps lower the mean loss", (float(np.mean(losses[:3])), float(np.mean(losses[-3:]))))
    check(not np.array_equal(records[0]["partial_weights"], records[-1]["partial_weights"]), "the Partial weights change between Step 0 and Step 100")


def flat_parameters(module) -> tuple[np.ndarray, np.ndarray]:
    """The live model's weights and biases as flat little-endian float32 arrays."""
    from backend.visualiser import linear_layers

    layers = linear_layers(module)
    weights = np.concatenate([np.ascontiguousarray(layer.weight.detach().cpu().numpy(), dtype="<f4").reshape(-1) for layer in layers])
    biases = np.concatenate([np.ascontiguousarray(layer.bias.detach().cpu().numpy(), dtype="<f4").reshape(-1) for layer in layers])
    return weights, biases


def expected_partial(weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The 1,010 Partial values, read out of a flat [out][in] weight array.

    Independent of the recorder: it only uses the documented layer shapes, the
    documented layer-major index order and ordinary numpy indexing.
    """
    values = []
    weight_offset = 0
    index_offset = 0
    for rows, columns in LAYERS:
        block = weights[weight_offset : weight_offset + rows * columns].reshape(rows, columns)
        chosen = indices[index_offset : index_offset + rows * PARTIAL_TOP_K].reshape(rows, PARTIAL_TOP_K)
        values.append(block[np.arange(rows)[:, None], chosen].reshape(-1))
        weight_offset += rows * columns
        index_offset += rows * PARTIAL_TOP_K
    return np.concatenate(values)


def forward_from_weights(weights: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Softmax probabilities for one uint8 image, using a flat weight array."""
    values = pixels.astype(np.float64) / 255.0
    offset = 0
    for position, (rows, columns) in enumerate(LAYERS):
        block = weights[offset : offset + rows * columns].reshape(rows, columns)
        offset += rows * columns
        values = block @ values
        if position < len(LAYERS) - 1:
            values = np.maximum(values, 0.0)
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def check_against_pytorch(recording: Recording) -> None:
    """Re-record with the same seed and compare every value with PyTorch's own."""
    from backend.visualiser import load_training_module, linear_layers, select_partial_indices, weights_checksum

    from recorder import record_training

    torch.manual_seed(SEED)
    module = load_training_module()
    initial_weights, initial_biases = flat_parameters(module)
    selection = select_partial_indices(module)
    initial_indices = np.concatenate([np.asarray(chosen, dtype="<u2").reshape(-1) for chosen in selection])

    check(len(initial_indices) == PARTIAL_COUNT, "select_partial_indices() returns 1,010 indices", len(initial_indices))
    check(np.array_equal(recording.indices, initial_indices), "partial-indices.bin equals select_partial_indices() on the initial weights")
    check(
        recording.manifest["model"]["initial_weights_sha256"] == weights_checksum(linear_layers(module)),
        "the manifest's checksum is the initial model's checksum",
        recording.manifest["model"]["initial_weights_sha256"],
    )

    with tempfile.TemporaryDirectory(prefix="mlp-recorder-") as directory:
        reproduced_root = Path(directory) / "reproduced"
        observed: list[dict] = []

        def observe(step: int, record_bytes: bytes, result: dict | None) -> None:
            weights, _ = flat_parameters(module)
            observed.append(
                {
                    "step": step,
                    "result": None
                    if result is None
                    else {
                        key: result[key].detach().clone()
                        for key in ("images", "labels", "A1", "A2", "Y", "loss")
                    },
                    "weights": weights,
                }
            )

        summary = record_training.record(
            reproduced_root,
            total_steps=STEPS,
            module=module,
            command="tests/test_milestone5a.py",
            progress=lambda message: None,
            on_checkpoint=observe,
        )
        reproduced = Recording(reproduced_root)

        check(summary["checkpoints"] == 11, "re-recording observed 11 checkpoints", summary["checkpoints"])
        check(len(observed) == 11, "the observation hook saw every checkpoint", len(observed))
        check(reproduced.blob == recording.blob, "re-recording with the same seed reproduces every record byte for byte")
        check((reproduced_root / "model.bin").read_bytes() == (recording.root / "model.bin").read_bytes(), "both runs start from the same seeded initial weights")

        recorded_full = np.frombuffer((recording.root / "model.bin").read_bytes(), dtype="<f4")
        recorded_biases = np.frombuffer((recording.root / "model-biases.bin").read_bytes(), dtype="<f4")
        check(np.array_equal(recorded_full, initial_weights), "model.bin holds exactly the live model's initial weights", float(np.abs(recorded_full - initial_weights).max()))
        check(np.array_equal(recorded_biases, initial_biases), "model-biases.bin holds exactly the live model's initial biases", float(np.abs(recorded_biases - initial_biases).max()))

        records = reproduced.records
        initial_partial = expected_partial(initial_weights, initial_indices)
        check(np.array_equal(records[0]["partial_weights"], initial_partial), "Step 0 stores the initial Partial weights", float(np.abs(records[0]["partial_weights"] - initial_partial).max()))
        check(observed[0]["result"] is None and expected_partial(observed[0]["weights"], initial_indices).tobytes() == initial_partial.tobytes(), "Step 0 is observed before any optimiser step")

        differences = []
        for index in range(1, len(observed)):
            observation = observed[index]
            record = records[index]
            step = observation["step"]
            result = observation["result"]

            image = np.rint(result["images"][0].view(-1).numpy().astype(np.float64) * 255.0).astype(np.uint8)
            probabilities = torch.softmax(result["Y"], dim=1).view(-1).numpy()
            activations = np.concatenate([result["A1"].view(-1).numpy(), result["A2"].view(-1).numpy(), probabilities])
            values = expected_partial(observation["weights"], initial_indices)

            check(record["step"] == step, f"Step {step} record carries its step number", record["step"])
            check(np.array_equal(record["image"], image), f"Step {step} image is the sample train_step() trained", int(np.count_nonzero(record["image"] != image)))
            check(record["true_label"] == int(result["labels"][0]), f"Step {step} label is the sample's label", (record["true_label"], int(result["labels"][0])))
            check(record["predicted_label"] == int(probabilities.argmax()), f"Step {step} prediction is the forward pass's argmax", (record["predicted_label"], int(probabilities.argmax())))
            check(np.allclose(record["probabilities"], probabilities, atol=1e-6), f"Step {step} probabilities are the forward pass's softmax", float(np.abs(record["probabilities"] - probabilities).max()))
            check(abs(record["loss"] - float(result["loss"])) <= 1e-6 * max(1.0, abs(float(result["loss"]))), f"Step {step} loss is that step's cross-entropy", (record["loss"], float(result["loss"])))
            check(np.allclose(record["activations"], activations, atol=1e-6), f"Step {step} activations are A1, A2 and the softmax", float(np.abs(record["activations"] - activations).max()))
            check(np.array_equal(record["partial_weights"], values), f"Step {step} Partial weights are the post-update parameters", float(np.abs(record["partial_weights"] - values).max()))

            post_update = forward_from_weights(observation["weights"], record["image"])
            differences.append(float(np.abs(post_update - record["probabilities"]).max()))

        check(min(differences) > 1e-6, "the recorded probabilities are pre-update: the post-update weights produce different ones", min(differences))

        final_selection = select_partial_indices(module)
        changed = [
            int(np.count_nonzero(np.sort(a, axis=1) != np.sort(b, axis=1)))
            for a, b in zip(selection, final_selection)
        ]
        check(sum(changed) > 0, "later weights would select other connections, yet the recording keeps the initial 1,010", changed)

        short_root = Path(directory) / "short"
        record_training.record(
            short_root,
            total_steps=25,
            module=module,
            command="tests/test_milestone5a.py",
            progress=lambda message: None,
        )
        short = Recording(short_root)
        steps = [record["step"] for record in short.records]
        check(steps == [0, 10, 20, 25], "the final step is recorded even when it is off the cadence", steps)
        check(short.manifest["training"]["last_step"] == 25 and short.manifest["training"]["complete_epoch"] is False, "an off-cadence run reports its real final step", (short.manifest["training"]["last_step"], short.manifest["training"]["complete_epoch"]))


def check_output_guards() -> None:
    with tempfile.TemporaryDirectory(prefix="mlp-recorder-") as directory:
        forbidden = Path("training") / "recorder-should-refuse"
        result = run_recorder(forbidden, steps=1, seed=SEED)
        check(result.returncode != 0, "the recorder refuses to write inside training/", result.stderr.strip())
        check(not (REPO_ROOT / forbidden).exists(), "the refused output directory is not created")
        check((REPO_ROOT / "training" / "training.py").is_file(), "training/training.py is still there")

        existing = Path(directory) / "not-empty"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep", encoding="utf-8")
        result = run_recorder(existing, steps=1, seed=SEED)
        check(result.returncode != 0, "the recorder refuses a non-empty output directory", result.stderr.strip())
        check(not (existing / "manifest.json").exists(), "a refused output directory gains no manifest")
        check((existing / "keep.txt").read_text(encoding="utf-8") == "keep", "a refused output directory is left alone")

        missing_parent = Path(directory) / "nothing" / "here" / "recording"
        result = run_recorder(missing_parent, steps=1, seed=SEED)
        check(result.returncode == 0 and (missing_parent / "manifest.json").is_file(), "the recorder creates its output directory", result.stderr.strip())


def check_source() -> None:
    source = (REPO_ROOT / "recorder" / "record_training.py").read_text(encoding="utf-8")
    check(not any(name in source for name in ("backend.server", "FastAPI", "uvicorn")), "the recorder never imports the server")
    check("train_step" in source and "load_training_module" in source, "the recorder reuses training/training.py's train_step()")
    check("nn.Linear" in source, "the recorder refers to the existing nn.Linear layers")
    constructs = re.search(r"(nn\.Linear|nn\.CrossEntropyLoss|torch\.optim\.Adam)\s*\(", source)
    check(constructs is None, "the recorder never constructs a model, loss or optimiser of its own", constructs.group(0) if constructs else "")
    check("FORBIDDEN_OUTPUTS" in source, "the recorder keeps its forbidden-output list")


def check_repository() -> None:
    status = subprocess.run(["git", "status", "--porcelain", "--", "training", "backend"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    check(status.stdout.strip() == "", "training/ and backend/ are unchanged", status.stdout.strip())

    ignored = subprocess.run(["git", "check-ignore", "-q", "recordings/example/manifest.json"], cwd=str(REPO_ROOT))
    check(ignored.returncode == 0, "recordings/ is ignored by Git")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    check("-m recorder.record_training" in readme, "README.md documents the recording command")
    check("tests/test_milestone5a.py" in readme, "README.md documents the recorder test")


def main() -> int:
    started = time.perf_counter()
    print(f"Repository: {REPO_ROOT}")

    with tempfile.TemporaryDirectory(prefix="mlp-recorder-") as directory:
        output = Path(directory) / "recording"
        result = run_recorder(output, steps=STEPS, seed=SEED)
        check(result.returncode == 0, "recorder.record_training exits successfully", (result.stderr or result.stdout).strip()[-2_000:])
        check("Checkpoints: 11" in result.stdout, "the recorder reports 11 checkpoints", result.stdout.strip()[-500:])
        check((output / "manifest.json").is_file(), "the run wrote manifest.json")

        if (output / "manifest.json").is_file():
            recording = Recording(output)
            check_files(recording)
            check_chunks(recording)
            check_manifest(recording)
            check_records(recording)
            check_against_pytorch(recording)

    check_output_guards()
    check_source()
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
