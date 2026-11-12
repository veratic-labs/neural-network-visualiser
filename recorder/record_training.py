"""Records one real training run of the existing PyTorch MLP for offline replay.

Run it from the repository root with the project virtualenv:

    ./venv/bin/python -m recorder.record_training --output recordings/mnist-epoch1

The network, the dataset, the loss function, the optimiser, the DataLoader and
the optimiser update all come from ``training/training.py``, imported through
``recorder.model_access.load_training_module()``. This module only decides
*when* to look at the network and *how* to write what it sees: it never builds a
second model, never re-implements the forward/backward pass, never changes a
hyperparameter and never calls ``train()`` (which would write weights into
``training/``). It binds no port and imports nothing from the retired Live
Training backend, which no longer exists.

Capture convention (plan.md section 7.3), identical to the live Milestone 4
frame that the retired backend produced before its helpers moved to
``recorder/model_access.py``:

* the image, true label, predicted digit, softmax probabilities, loss and
  activations of checkpoint ``k`` are read from the forward pass of step ``k``,
  i.e. **before** that step's ``optimiser.step()``;
* the 1,010 Partial weights of checkpoint ``k`` are read **after** that step's
  ``optimiser.step()``, so they are the parameters the step produced.

Step 0 is the initial model: it has the initial Partial weights, the
``has_sample`` flag is 0 and its image, probabilities, loss and activation bytes
are zero.

Binary layout (plan.md section 7.4) — every checkpoint record has the same
fixed stride, so a record is read by arithmetic and never by parsing JSON:

======  ======  ==========================================  ==================
Offset  Bytes   Field                                       Type
======  ======  ==========================================  ==================
0       4       completed optimiser step                    uint32
4       4       sample position in the epoch (1-based)      uint32
8       1       true label (0-9)                            uint8
9       1       predicted digit                             uint8
10      1       flags (bit 0 = has sample)                  uint8
11      1       reserved (0)                                uint8
12      784     image, row-major 28 x 28                    uint8
796     40      softmax probabilities, classes 0-9          float32 x 10
836     4       cross-entropy loss                          float32
840     808     activations A1 (128), A2 (64), output (10)  float32 x 202
1,648   4,040   Partial weights over the 1,010 indices     float32 x 1,010
======  ======  ==========================================  ==================
        5,688   record size in bytes
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence

import numpy as np
import torch

from recorder.model_access import (
    PARTIAL_TOP_K,
    describe_architecture,
    encode_pixels,
    linear_layers,
    load_training_module,
    select_partial_indices,
    tensor_bytes,
    training_totals,
    visible_weight_layers,
    weights_checksum,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path("recordings") / "mnist-epoch1"

FORMAT_NAME = "mlp-visualiser-recording"
FORMAT_VERSION = 1

# One checkpoint every 10 completed optimiser steps, plus Step 0 and the final
# step (plan.md sections 3.2 and 7.2).
SNAPSHOT_CADENCE = 10

# Chunk files hold a whole number of fixed-stride records; the size is recorded
# in the manifest so the replay never has to guess a file's contents.
DEFAULT_CHUNK_SIZE = 500

MANIFEST_FILE = "manifest.json"
MODEL_FILE = "model.bin"
BIAS_FILE = "model-biases.bin"
INDICES_FILE = "partial-indices.bin"
CHUNKS_DIR = "chunks"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT

FLAG_HAS_SAMPLE = 1

# The one fixed diverging weight scale, copied from frontend/network.js
# (MIN_DISPLAY_WEIGHT / MAX_DISPLAY_WEIGHT and WEIGHT_COLOUR_STOPS). Recordings
# describe the scale so a viewer can confirm the drawing uses it unchanged.
COLOUR_MIN = -0.5
COLOUR_MAX = 0.5
COLOUR_STOPS = (
    (-0.5, "#b91c1c"),
    (-0.25, "#fdba74"),
    (0.0, "#cbd5e1"),
    (0.25, "#93c5fd"),
    (0.5, "#6d28d9"),
)

# Directories the recorder must never write into (plan.md section 7.2).
FORBIDDEN_OUTPUTS = ("training", "frontend", "docs")

PROGRESS_INTERVAL = 1_000


def chunk_name(index: int) -> str:
    """The name of chunk file ``index`` inside the chunks directory."""
    return f"chunk-{index:04d}.bin"



def build_layout(layers: Sequence[torch.nn.Linear], selection: Sequence[np.ndarray]) -> dict:
    """The checkpoint record layout for this network and this selection.

    The field order and dtypes are the documented format (plan.md section 7.4).
    Endianness is fixed by writing raw little-endian bytes, so the layout is the
    same on any host. For 784 -> 128 -> 64 -> 10 with k = 5 this produces the
    documented 5,688-byte record.
    """
    hidden_sizes = [int(layer.out_features) for layer in layers[:-1]]
    output_size = int(layers[-1].out_features)
    activation_counts = hidden_sizes + [output_size]
    activation_count = sum(activation_counts)
    partial_count = sum(int(order.size) for order in selection)

    fields: list[dict] = []
    offset = 0

    def add(name: str, size: int, dtype: str, count: int = 1) -> None:
        nonlocal offset
        fields.append(
            {"name": name, "offset": offset, "bytes": size, "dtype": dtype, "count": count}
        )
        offset += size

    add("step", 4, "uint32")
    add("sample_position", 4, "uint32")
    add("true_label", 1, "uint8")
    add("predicted_label", 1, "uint8")
    add("flags", 1, "uint8")
    add("reserved", 1, "uint8")
    add("image", IMAGE_BYTES, "uint8", IMAGE_BYTES)
    add("probabilities", output_size * 4, "float32", output_size)
    add("loss", 4, "float32")
    add("activations", activation_count * 4, "float32", activation_count)
    add("partial_weights", partial_count * 4, "float32", partial_count)

    return {
        "fields": fields,
        "offsets": {field["name"]: field["offset"] for field in fields},
        "record_size": offset,
        "activation_counts": activation_counts,
        "activation_count": activation_count,
        "partial_count": partial_count,
        "output_size": output_size,
    }


def _float32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def image_pixels(image: torch.Tensor) -> np.ndarray:
    """A ``ToTensor()`` 28 x 28 image as the uint8 bytes the dataset stores.

    ``encode_pixels()`` multiplies the 0..1 tensor back by 255 and rounds, which
    restores exactly the pixels ``ToTensor()`` divided, so a recorded image is
    the dataset's own bytes.
    """
    return np.frombuffer(base64.b64decode(encode_pixels(image)), dtype=np.uint8)


def partial_weight_values(module: ModuleType, selection: Sequence[np.ndarray]) -> np.ndarray:
    """The current weights of the fixed Partial connections, layer-major.

    ``visible_weight_layers()`` returns them in the same order (and with the
    same indices) for every checkpoint, because the indices come from
    ``selection`` and are never recomputed.
    """
    arrays = [
        np.frombuffer(base64.b64decode(layer["values"]), dtype="<f4")
        for layer in visible_weight_layers(module, selection)
    ]
    return np.concatenate(arrays)


def encode_record(
    layout: dict,
    *,
    step: int,
    partial_weights: np.ndarray,
    sample_position: int = 0,
    true_label: int = 0,
    predicted_label: int = 0,
    image: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    loss: float = 0.0,
    activations: np.ndarray | None = None,
) -> bytes:
    """One fixed-stride checkpoint record.

    ``image`` is ``None`` only for Step 0, whose sample fields stay zero and
    whose flags field has the ``has_sample`` bit clear.
    """
    offsets = layout["offsets"]
    record = bytearray(layout["record_size"])
    has_sample = image is not None

    if has_sample and (probabilities is None or activations is None):
        raise ValueError("a record with an image also needs probabilities and activations")
    if not has_sample and (probabilities is not None or activations is not None):
        raise ValueError("a record without an image cannot carry probabilities or activations")

    partial_weights = np.ascontiguousarray(partial_weights, dtype="<f4").reshape(-1)
    if partial_weights.size != layout["partial_count"]:
        raise ValueError(
            f"expected {layout['partial_count']} Partial weights, got {partial_weights.size}"
        )

    struct.pack_into("<II", record, offsets["step"], int(step), int(sample_position))
    record[offsets["true_label"]] = int(true_label) & 0xFF
    record[offsets["predicted_label"]] = int(predicted_label) & 0xFF
    record[offsets["flags"]] = FLAG_HAS_SAMPLE if has_sample else 0
    record[offsets["reserved"]] = 0

    if has_sample:
        pixels = np.ascontiguousarray(image, dtype=np.uint8).reshape(-1)
        if pixels.size != IMAGE_BYTES:
            raise ValueError(f"expected {IMAGE_BYTES} pixels, got {pixels.size}")

        probabilities = np.ascontiguousarray(probabilities, dtype="<f4").reshape(-1)
        if probabilities.size != layout["output_size"]:
            raise ValueError(
                f"expected {layout['output_size']} probabilities, got {probabilities.size}"
            )

        activations = np.ascontiguousarray(activations, dtype="<f4").reshape(-1)
        if activations.size != layout["activation_count"]:
            raise ValueError(
                f"expected {layout['activation_count']} activations, got {activations.size}"
            )

        start = offsets["image"]
        record[start : start + IMAGE_BYTES] = pixels.tobytes()
        start = offsets["probabilities"]
        record[start : start + probabilities.nbytes] = _float32_bytes(probabilities)
        struct.pack_into("<f", record, offsets["loss"], float(loss))
        start = offsets["activations"]
        record[start : start + activations.nbytes] = _float32_bytes(activations)

    start = offsets["partial_weights"]
    record[start : start + partial_weights.nbytes] = _float32_bytes(partial_weights)
    return bytes(record)



def build_checkpoint(
    layout: dict,
    module: ModuleType,
    result: dict,
    *,
    step: int,
    sample_position: int,
    selection: Sequence[np.ndarray],
) -> bytes:
    """One completed optimiser step of ``train_step()`` as a checkpoint record.

    ``result`` is what ``train_step()`` returned: its ``X``, ``A1``, ``A2``,
    ``Y`` and ``loss`` were produced before that step's ``optimiser.step()``.
    The Partial weights are read from the layers here, i.e. after the update.
    """
    with torch.no_grad():
        probabilities = torch.softmax(result["Y"], dim=1)
        predicted_label = int(torch.argmax(probabilities, dim=1))

    activation_arrays = [
        np.frombuffer(tensor_bytes(tensor.view(-1)), dtype="<f4")
        for tensor in (result["A1"], result["A2"], probabilities)
    ]

    return encode_record(
        layout,
        step=step,
        partial_weights=partial_weight_values(module, selection),
        sample_position=sample_position,
        true_label=int(result["labels"][0]),
        predicted_label=predicted_label,
        image=image_pixels(result["images"][0]),
        probabilities=np.frombuffer(tensor_bytes(probabilities.view(-1)), dtype="<f4"),
        loss=float(result["loss"].detach()),
        activations=np.concatenate(activation_arrays),
    )


class RecordingWriter:
    """Writes checkpoint records in order, rotating chunk files.

    Chunk files are created on demand and hold at most ``chunk_size`` records,
    so a complete run is a handful of files that can be fetched one at a time.
    Every record has the same stride, which is why the manifest can address a
    checkpoint with arithmetic alone.
    """

    def __init__(self, chunks_dir: Path, *, record_size: int, chunk_size: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk size must be at least 1 record")
        self.chunks_dir = chunks_dir
        self.record_size = record_size
        self.chunk_size = chunk_size
        self.chunks: list[dict] = []
        self.count = 0
        self._handle = None
        self._current: dict | None = None

    def write(self, step: int, record: bytes) -> None:
        if len(record) != self.record_size:
            raise ValueError(
                f"record is {len(record)} bytes, layout says {self.record_size}"
            )
        if self._handle is None or self._current["count"] == self.chunk_size:
            self._open_chunk(step)

        self._handle.write(record)
        self._current["last_step"] = int(step)
        self._current["count"] += 1
        self._current["bytes"] += len(record)
        self.count += 1

    def _open_chunk(self, first_step: int) -> None:
        self._close_chunk()
        index = len(self.chunks)
        path = self.chunks_dir / chunk_name(index)
        self._handle = path.open("wb")
        self._current = {
            "file": f"{self.chunks_dir.name}/{path.name}",
            "index": index,
            "first_step": int(first_step),
            "last_step": int(first_step),
            "count": 0,
            "bytes": 0,
        }
        self.chunks.append(self._current)

    def _close_chunk(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def close(self) -> None:
        self._close_chunk()



def build_manifest(
    *,
    module: ModuleType,
    layers: Sequence[torch.nn.Linear],
    selection: Sequence[np.ndarray],
    layout: dict,
    initial_checksum: str,
    recorded_steps: Sequence[int],
    total_steps: int,
    configured_total: int,
    epochs: int,
    train_size: int,
    chunk_size: int,
    writer: RecordingWriter,
    duration_seconds: float,
    command: str | None,
    seed: int | None,
) -> dict:
    """Everything the replay needs to describe its own data (plan.md 7.5)."""
    architecture = describe_architecture(layers)
    batch_size = int(getattr(module, "batch_size", 1))
    connection_count = sum(
        int(layer.out_features) * int(layer.in_features) for layer in layers
    )
    bias_count = sum(int(layer.out_features) for layer in layers)
    partial_layers = [
        {
            "index": index,
            "in": int(layer.in_features),
            "out": int(layer.out_features),
            "k": int(order.shape[1]),
            "visible_connections": int(order.size),
        }
        for index, (layer, order) in enumerate(zip(layers, selection))
    ]
    partial_count = sum(entry["visible_connections"] for entry in partial_layers)
    checkpoint_count = len(recorded_steps)

    optimiser = getattr(module, "optimiser", None)
    learning_rate = float(optimiser.param_groups[0]["lr"]) if optimiser else 0.001

    limitations = [
        "Full mode uses the Step 0 initial weights only; later full-network weights are not recorded.",
        "Partial mode stores the 1,010 fixed connections only; the selection is never recomputed.",
        "A recording is written in one pass: an interrupted run has no manifest and must be regenerated.",
    ]
    if seed is None:
        limitations.insert(
            0,
            "Initial weights are random because training/training.py sets no seed; "
            "each recording stores its own initial weights.",
        )
    else:
        limitations.insert(
            0,
            f"torch.manual_seed({seed}) was set before the model was created, so this "
            "recording is reproducible.",
        )

    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "format": {
            "name": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "byte_order": "little-endian",
            "record_size": layout["record_size"],
            "record_alignment": (
                "every float32 field starts on a 4-byte boundary; all arrays are row-major"
            ),
            "record": {
                "size_bytes": layout["record_size"],
                "fields": layout["fields"],
                "flags": {"has_sample_bit": 0},
                "note": "one record per checkpoint, always at the same fixed stride",
            },
            "chunks": {
                "directory": CHUNKS_DIR,
                "naming": f"{CHUNKS_DIR}/chunk-{{index:04d}}.bin",
                "chunk_size_records": chunk_size,
                "checkpoint_to_chunk": (
                    "chunk index = checkpoint_index // chunk_size_records"
                ),
                "checkpoint_to_offset": (
                    "offset = (checkpoint_index % chunk_size_records) * record_size"
                ),
            },
            "manifest": MANIFEST_FILE,
        },
        "model": {
            "architecture": architecture,
            "layers": [
                {
                    "index": index,
                    "in": int(layer.in_features),
                    "out": int(layer.out_features),
                    "weight_count": int(layer.out_features) * int(layer.in_features),
                    "bias_count": int(layer.out_features),
                }
                for index, layer in enumerate(layers)
            ],
            "input_size": int(layers[0].in_features),
            "activation": (
                "ReLU on both hidden layers; the output layer's logits feed nn.CrossEntropyLoss"
            ),
            "loss": "nn.CrossEntropyLoss",
            "optimiser": "torch.optim.Adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_samples": train_size,
            "total_steps": configured_total,
            "connections": connection_count,
            "biases": bias_count,
            "initial_weights_sha256": initial_checksum,
            "initial_weights_sha256_note": (
                "sha256 over every initial weight and bias byte in layer order "
                "(weights then biases per layer), as "
                "recorder.model_access.weights_checksum() computes it"
            ),
        },
        "training": {
            "recorded_at": captured_at,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "duration_seconds": round(float(duration_seconds), 3),
            "snapshot_cadence": SNAPSHOT_CADENCE,
            "first_step": int(recorded_steps[0]),
            "last_step": int(recorded_steps[-1]),
            "checkpoint_count": checkpoint_count,
            "recorded_steps": [int(step) for step in recorded_steps],
            "steps_recorded": int(total_steps),
            "complete_epoch": int(total_steps) == int(configured_total),
            "seed": seed,
            "recorder": "recorder/record_training.py",
        },
        "checkpoints": {
            "count": checkpoint_count,
            "first_step": int(recorded_steps[0]),
            "last_step": int(recorded_steps[-1]),
            "record_size": layout["record_size"],
            "total_bytes": checkpoint_count * layout["record_size"],
            "chunks": writer.chunks,
        },
        "partial": {
            "selection_rule": (
                f"the top {PARTIAL_TOP_K} incoming |weight| per receiving neuron of the "
                "initial model, fixed for the whole recording"
            ),
            "top_k": PARTIAL_TOP_K,
            "count": partial_count,
            "indices_file": INDICES_FILE,
            "indices_dtype": "uint16",
            "index_order": (
                "layer-major [out][k]: all k input indices of output neuron 0, then "
                "neuron 1, ...; layers in network order; k in select_partial_indices() order"
            ),
            "layers": partial_layers,
            "weights": (
                "every checkpoint stores the 1,010 weights at these indices, layer-major, "
                "in this same order"
            ),
        },
        "full": {
            "weights_file": MODEL_FILE,
            "connection_count": connection_count,
            "dtype": "float32",
            "shape_order": "row-major [out][in]",
            "layout": (
                "layer 1 [128][784], layer 2 [64][128], layer 3 [10][64], row-major "
                "[out][in], concatenated in that order"
            ),
            "biases_file": BIAS_FILE,
            "bias_count": bias_count,
            "later_full_weights_recorded": False,
            "note": (
                "The initial network is written once. Later full-network weights are not "
                "recorded, so Full mode keeps its colours for the whole replay."
            ),
        },
        "activations": {
            "order": ["A1", "A2", "output"],
            "sizes": layout["activation_counts"],
            "count": layout["activation_count"],
            "capture": "pre-update",
            "input": (
                "input activations are the 784 pixels divided by 255; the replay derives "
                "them from the uint8 image exactly as frontend/replay.js does"
            ),
        },
        "images": {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "dtype": "uint8",
            "range": [0, 255],
            "layout": "row-major",
        },
        "capture": {
            "image": "the MNIST sample this optimiser step trained",
            "true_label": "the label of that sample",
            "prediction": "pre-update",
            "probabilities": "pre-update",
            "activations": "pre-update",
            "loss": "pre-update",
            "partial_weights": "post-update",
            "step_0": (
                "the initial model before any optimiser step: no sample, prediction, "
                "activations or loss, and the initial Partial weights"
            ),
            "note": (
                "The image, prediction, probabilities, activations and loss come from the "
                "forward pass that this step backpropagated. The Partial weights were read "
                "from the layers after optimiser.step(), so they are the parameters that "
                "step produced."
            ),
        },
        "colour": {
            "min": COLOUR_MIN,
            "max": COLOUR_MAX,
            "stops": [
                {"weight": weight, "colour": colour} for weight, colour in COLOUR_STOPS
            ],
            "normalisation": (
                "none: one fixed scale for every layer, mode and checkpoint; magnitudes "
                "saturate at +/-0.5 and no per-layer or per-checkpoint rescaling is applied"
            ),
        },
        "notes": {
            "command": command or "record() called directly from Python (no command line)",
            "limitations": limitations,
        },
    }

def _prepare_output(output: Path) -> Path:
    """Create the recording directory, refusing anything that would be unsafe."""
    resolved = output.expanduser().resolve()
    for name in FORBIDDEN_OUTPUTS:
        forbidden = (REPO_ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(f"refusing to write a recording into {name}/: {resolved}")

    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"output path exists and is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise ValueError(
                f"output directory is not empty: {resolved}. "
                "Delete it or choose another --output."
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def record(
    output: Path | str,
    *,
    total_steps: int | None = None,
    seed: int | None = None,
    cadence: int = SNAPSHOT_CADENCE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    module: ModuleType | None = None,
    on_checkpoint: Callable[[int, bytes, dict | None], None] | None = None,
    log_every: int = PROGRESS_INTERVAL,
    command: str | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Train the existing model and write one complete recording.

    ``total_steps`` defaults to the configured length of one epoch (60,000 with
    ``batch_size = 1``). A smaller value only shortens the recording; it never
    changes the model, the optimiser, the batch size or the sample order, so a
    short run is a real prefix of a full one.

    ``module`` lets a caller supply the training module it already imported (the
    tests do this); otherwise the module is loaded here, after applying ``seed``
    so the initial weights are reproducible.

    ``on_checkpoint(step, record_bytes, result)`` is an observation hook for
    tests. It is called after each record is written; ``result`` is ``None`` for
    Step 0. It cannot change what is recorded.
    """
    if cadence < 1:
        raise ValueError("the snapshot cadence must be at least 1 step")
    if chunk_size < 1:
        raise ValueError("the chunk size must be at least 1 record")

    output = _prepare_output(Path(output))
    chunks_dir = output / CHUNKS_DIR
    chunks_dir.mkdir()

    if module is None:
        if seed is not None:
            torch.manual_seed(seed)
        module = load_training_module()
    elif seed is not None:
        raise ValueError("pass either module= or seed=, not both")

    layers = linear_layers(module)
    configured_total, epochs, train_size = training_totals(module)
    configured_steps = int(configured_total)

    total_steps = configured_steps if total_steps is None else int(total_steps)
    if total_steps < 1:
        raise ValueError("at least one optimiser step has to be recorded")
    if total_steps > configured_steps:
        raise ValueError(
            f"{total_steps:,} steps exceed the configured {configured_steps:,} "
            "(one epoch of training/training.py)"
        )

    # The Partial selection is made exactly once, on the initial weights, and
    # every checkpoint is written at these indices (plan.md section 7.2).
    selection = select_partial_indices(module)
    layout = build_layout(layers, selection)
    initial_checksum = weights_checksum(layers)

    (output / MODEL_FILE).write_bytes(
        b"".join(tensor_bytes(layer.weight) for layer in layers)
    )
    (output / BIAS_FILE).write_bytes(b"".join(tensor_bytes(layer.bias) for layer in layers))
    (output / INDICES_FILE).write_bytes(
        b"".join(np.ascontiguousarray(order, dtype="<u2").tobytes() for order in selection)
    )

    sizes = " -> ".join(str(layer["size"]) for layer in describe_architecture(layers))
    progress(f"Model:       {sizes} ({layout['partial_count']:,} Partial connections)")
    progress(f"Training:    {total_steps:,} optimiser steps, snapshot every {cadence}")
    progress(f"Output:      {output}")

    writer = RecordingWriter(
        chunks_dir, record_size=layout["record_size"], chunk_size=chunk_size
    )
    started = time.monotonic()
    try:
        # Step 0: the initial model, before any optimiser step.
        step_zero = encode_record(
            layout, step=0, partial_weights=partial_weight_values(module, selection)
        )
        writer.write(0, step_zero)
        if on_checkpoint is not None:
            on_checkpoint(0, step_zero, None)

        iterator = iter(module.train_loader)
        step = 0
        last_loss = 0.0
        while step < total_steps:
            try:
                images, labels = next(iterator)
            except StopIteration:
                raise RuntimeError(
                    "the training loader ran out of samples before the requested steps"
                ) from None

            result = module.train_step(images, labels)
            step += 1
            last_loss = float(result["loss"].detach())

            if step % cadence == 0 or step == total_steps:
                record_bytes = build_checkpoint(
                    layout,
                    module,
                    result,
                    step=step,
                    sample_position=(step - 1) % train_size + 1,
                    selection=selection,
                )
                writer.write(step, record_bytes)
                if on_checkpoint is not None:
                    on_checkpoint(step, record_bytes, result)

            if log_every and step % log_every == 0:
                progress(f"step {step:,}/{total_steps:,}  loss {last_loss:.6f}")
    finally:
        writer.close()

    duration = time.monotonic() - started

    recorded_steps = list(range(0, total_steps + 1, cadence))
    if recorded_steps[-1] != total_steps:
        recorded_steps.append(total_steps)
    if len(recorded_steps) != writer.count:
        raise RuntimeError(
            f"wrote {writer.count} checkpoints but the cadence implies {len(recorded_steps)}"
        )

    manifest = build_manifest(
        module=module,
        layers=layers,
        selection=selection,
        layout=layout,
        initial_checksum=initial_checksum,
        recorded_steps=recorded_steps,
        total_steps=total_steps,
        configured_total=configured_steps,
        epochs=epochs,
        train_size=train_size,
        chunk_size=chunk_size,
        writer=writer,
        duration_seconds=duration,
        command=command,
        seed=seed,
    )
    # The manifest is written last, so its presence means a complete recording.
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if len(recorded_steps) > 2:
        steps_text = f"{recorded_steps[0]}, {recorded_steps[1]}, ..., {recorded_steps[-1]}"
    else:
        steps_text = ", ".join(str(value) for value in recorded_steps)

    progress("")
    progress(f"Done:        {total_steps:,} optimiser steps in {duration:.1f} s")
    progress(f"Checkpoints: {writer.count:,} in {len(writer.chunks)} chunk file(s)")
    progress(f"Steps:       {steps_text}")
    progress(f"Output size: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MiB)")

    return {
        "output": output,
        "steps": total_steps,
        "checkpoints": writer.count,
        "chunks": len(writer.chunks),
        "checkpoint_bytes": writer.count * layout["record_size"],
        "bytes": total_bytes,
        "duration_seconds": duration,
    }


def command_line(argv: Sequence[str]) -> str:
    """The exact command the manifest records, with a repository-relative Python."""
    executable = Path(sys.executable)
    try:
        # venv/bin/python is a symlink to the system interpreter, so compare the
        # path as invoked rather than its resolved target.
        executable_text = "./" + str(executable.absolute().relative_to(REPO_ROOT))
    except ValueError:
        executable_text = str(executable)
    return " ".join([executable_text, "-m", "recorder.record_training", *argv])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="recorder.record_training",
        description=(
            "Train training/training.py once and record the run for the offline "
            "visualiser (plan.md milestone 5A)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            f"recording directory to create (default: {DEFAULT_OUTPUT}); "
            "it must not exist yet or must be empty"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "record only the first N optimiser steps, for verification; "
            "the default records one full epoch"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional torch.manual_seed() value, recorded in the manifest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"checkpoints per chunk file (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=PROGRESS_INTERVAL,
        help=f"print progress every N completed steps (default: {PROGRESS_INTERVAL})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    try:
        record(
            args.output,
            total_steps=args.steps,
            seed=args.seed,
            chunk_size=args.chunk_size,
            log_every=args.log_every,
            command=command_line(raw_argv),
        )
    except KeyboardInterrupt:
        print(
            "\nRecording interrupted. The output directory is incomplete and has no "
            "manifest; delete it and record again.",
            file=sys.stderr,
        )
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Recording failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

