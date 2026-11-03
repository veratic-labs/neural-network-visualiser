"""Reads the existing PyTorch model and turns it into a JSON-ready payload.

The network is **not** re-implemented here. `training/training.py` is imported
from its file path, so the backend sees exactly the layers the training script
builds (layer1, layer2, layer3) and their real initial weights. The training
loop in that file sits behind an ``if __name__ == "__main__":`` guard, so
importing the module defines the network without starting a training run.

Milestone 2 sends this payload once per WebSocket connection and never runs an
optimiser step, so the weights are the untrained ``nn.Linear`` values.

Milestone 3 adds `build_inference()`: one real MNIST sample from the existing
`train_data` dataset, pushed through the same forward pass `train()` uses, with
softmax probabilities and per-layer activations. It runs under `torch.no_grad()`
and never touches the optimiser, so the weights stay exactly as they were.

Milestone 4 adds `build_training_step()`, which serialises one *real* completed
optimiser step of ``train_step()`` in training/training.py. Everything that
describes the sample — pixels, prediction, softmax probabilities, activations
and loss — is read from the forward pass that produced that step's gradients,
i.e. **before** the optimiser update. The weights it carries are read **after**
``optimiser.step()``, so the colours the browser draws are the real post-update
parameters. Only the connections the Partial view draws are sent;
`select_partial_indices()` chooses them once from the model's initial weights and
a live frame never changes that set, so the same lines stay on screen while only
their values move. `build_weight_snapshot()` returns the full tensors when the
Full view needs them.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
TRAINING_FILE = TRAINING_DIR / "training.py"

# Weights travel as raw float32 bytes in base64. PyTorch stores little-endian
# float32 and browsers run on little-endian hardware, so the frontend can hand
# the bytes straight to a Float32Array.
DTYPE = "float32"
BYTE_ORDER = "little-endian"
SHAPE_ORDER = "row-major [out][in]"

# Sample images travel as the original MNIST bytes, one byte per pixel.
IMAGE_SIZE = 28
PIXEL_ENCODING = "uint8-base64"

# Mirrors PARTIAL_TOP_K in frontend/network.js: the reduced training view draws
# this many incoming connections per neuron. `select_partial_indices()` picks
# them once from the initial weights, so a live frame only has to carry those
# weights instead of all 109,184 of them — but always the same ones.
PARTIAL_TOP_K = 5


def load_training_module() -> ModuleType:
    """Import training/training.py without running the training loop."""
    if not TRAINING_FILE.is_file():
        raise FileNotFoundError(f"training script not found: {TRAINING_FILE}")

    # training.py does a flat `from load_data import train_data`, so the
    # training directory itself has to be importable.
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))

    spec = importlib.util.spec_from_file_location("mnist_mlp_training", TRAINING_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {TRAINING_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear_layers(module: ModuleType) -> list[nn.Linear]:
    """The network's linear layers, in the order training.py defines them."""
    layers = [value for value in vars(module).values() if isinstance(value, nn.Linear)]
    if not layers:
        raise RuntimeError(f"no nn.Linear layers found in {TRAINING_FILE}")

    for previous, current in zip(layers, layers[1:]):
        if previous.out_features != current.in_features:
            raise ValueError(
                f"layers are not chained: {previous.out_features} outputs "
                f"feeding {current.in_features} inputs"
            )
    return layers


def describe_architecture(layers: list[nn.Linear]) -> list[dict]:
    """Layer sizes taken from the live model; only the display names are derived."""
    architecture = [{"name": "Input", "size": layers[0].in_features}]
    last_index = len(layers) - 1
    for index, layer in enumerate(layers):
        name = "Output" if index == last_index else f"Hidden {index + 1}"
        architecture.append({"name": name, "size": layer.out_features})
    return architecture


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """A weight/bias tensor as raw row-major little-endian float32 bytes."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    return array.tobytes()


def encode_tensor(tensor: torch.Tensor) -> str:
    """The same bytes, base64 encoded for the JSON `model_info` message."""
    return base64.b64encode(tensor_bytes(tensor)).decode("ascii")


def weights_checksum(layers: list[nn.Linear]) -> str:
    """sha256 over every weight and bias byte, in layer order.

    Sent with `model_info` and with every inference snapshot, so the frontend
    (and the tests) can prove that a forward pass never changed a parameter.
    """
    digest = hashlib.sha256()
    for layer in layers:
        digest.update(tensor_bytes(layer.weight))
        digest.update(tensor_bytes(layer.bias))
    return digest.hexdigest()


def encode_layer_weights(layers: list[nn.Linear]) -> list[dict]:
    """Every weight and bias of every layer, as base64 float32."""
    encoded_layers = []
    for index, layer in enumerate(layers):
        encoded_layers.append(
            {
                "index": index,
                "in": layer.in_features,
                "out": layer.out_features,
                "weights": encode_tensor(layer.weight),
                "biases": encode_tensor(layer.bias),
                "weight_count": layer.out_features * layer.in_features,
                "bias_count": layer.out_features,
            }
        )
    return encoded_layers


def build_model_info(
    module: ModuleType,
    training: dict | None = None,
    selection: list[np.ndarray] | None = None,
) -> dict:
    """The `model_info` payload sent to every new WebSocket client.

    `training` is the training session's current status, so a browser that
    connects mid-run immediately knows whether to show Start, Pause or Resume.

    `selection` is the session's fixed Partial-view selection. Sending it here
    means the page and the server draw exactly the same connections for the whole
    run, instead of the browser guessing at a rule from the weights of the moment.
    Callers without a session get the selection derived from the current weights.
    """
    layers = linear_layers(module)
    architecture = describe_architecture(layers)
    if selection is None:
        selection = select_partial_indices(module)

    encoded_layers = encode_layer_weights(layers)
    connections = sum(layer["weight_count"] for layer in encoded_layers)
    parameters = connections + sum(layer["bias_count"] for layer in encoded_layers)

    partial = [
        {
            "index": index,
            "in": int(layer.in_features),
            "out": int(layer.out_features),
            "k": int(chosen.shape[1]),
            "indices": encode_indices(chosen),
            "visible_connections": int(chosen.size),
        }
        for index, (layer, chosen) in enumerate(zip(layers, selection))
    ]

    total_steps, epochs, train_size = training_totals(module)
    batch_size = getattr(module, "batch_size", None)
    step = int(training["step"]) if training else 0
    trained = step > 0

    return {
        "type": "model_info",
        "architecture": architecture,
        "layers": encoded_layers,
        "partial": {
            "top_k": PARTIAL_TOP_K,
            "count": sum(layer["visible_connections"] for layer in partial),
            "rule": (
                f"the top {PARTIAL_TOP_K} incoming |weight| per neuron of the "
                "initial model, fixed for the whole run"
            ),
            "selection": partial,
        },
        "totals": {
            "neurons": sum(layer["size"] for layer in architecture),
            "connections": connections,
            "parameters": parameters,
        },
        "batch_size": batch_size,
        "total_steps": total_steps,
        "epochs": epochs,
        "train_size": train_size,
        "dtype": DTYPE,
        "byte_order": BYTE_ORDER,
        "shape_order": SHAPE_ORDER,
        "weights_checksum": weights_checksum(layers),
        "weights_source": str(TRAINING_FILE.relative_to(REPO_ROOT)),
        "weights_state": "trained" if trained else "untrained",
        "training": training,
        "message": (
            f"Weights after {step:,} optimiser step(s) of training/training.py."
            if trained
            else (
                "Initial weights of the existing PyTorch model; no training steps "
                "have been run yet."
            )
        ),
    }


def build_weight_snapshot(module: ModuleType, step: int = 0) -> dict:
    """Every current weight and bias, for the Full view during/after training.

    Since Milestone 4 the model really changes, so the Full view asks for this
    snapshot when it is drawn instead of receiving all 109,184 weights with
    every training step.
    """
    layers = linear_layers(module)
    return {
        "type": "weights",
        "step": step,
        "layers": encode_layer_weights(layers),
        "dtype": DTYPE,
        "byte_order": BYTE_ORDER,
        "shape_order": SHAPE_ORDER,
        "weights_checksum": weights_checksum(layers),
        "weights_state": "trained" if step else "untrained",
        "message": f"All current weights after {step:,} optimiser step(s).",
    }


def configured_epochs(module: ModuleType) -> int:
    """How many passes over the training split training.py runs.

    training.py has no epoch setting: ``train()`` makes exactly one pass over
    ``train_loader``, so one epoch is the configured value.
    """
    return max(1, int(getattr(module, "epochs", 1)))


def training_totals(module: ModuleType) -> tuple[int, int, int]:
    """(total optimiser steps, epochs, training samples) for the real dataset.

    ``batch_size`` is 1, so every optimiser step consumes exactly one MNIST
    image and the total is the dataset length times the configured epochs.
    """
    train_size = len(sample_dataset(module))
    if train_size == 0:
        raise RuntimeError("the MNIST training split is empty")
    epochs = configured_epochs(module)
    return train_size * epochs, epochs, train_size


def snapshot_parameters(module: ModuleType) -> list[dict]:
    """Copies of every weight and bias, so a run can return to its start."""
    return [
        {name: value.detach().clone() for name, value in layer.state_dict().items()}
        for layer in linear_layers(module)
    ]


def restore_parameters(module: ModuleType, snapshot: list[dict]) -> None:
    """Puts the copied tensors back and forgets the optimiser's Adam state.

    ``load_state_dict`` copies into the existing tensors, so the optimiser keeps
    pointing at the same parameters while its moments are reset, exactly as a
    freshly built model would be.
    """
    for layer, saved in zip(linear_layers(module), snapshot):
        layer.load_state_dict({name: value.clone() for name, value in saved.items()})

    optimiser = getattr(module, "optimiser", None)
    if optimiser is not None:
        optimiser.state.clear()


def sample_dataset(module: ModuleType) -> object:
    """The MNIST training split that training/load_data.py already loads."""
    dataset = getattr(module, "train_data", None)
    if dataset is None:
        raise RuntimeError(f"{TRAINING_FILE} does not expose train_data")
    return dataset


def encode_pixels(image: torch.Tensor) -> str:
    """A 28 x 28 ``ToTensor()`` image in 0..1 as base64-encoded uint8 bytes.

    ``ToTensor()`` divides the MNIST bytes by 255, so multiplying back and
    rounding restores exactly the pixels stored in the dataset file.
    """
    values = image.detach().cpu().numpy().reshape(-1) * 255.0
    array = np.ascontiguousarray(np.rint(values).clip(0, 255), dtype=np.uint8)
    return base64.b64encode(array.tobytes()).decode("ascii")


def build_inference(module: ModuleType, index: int) -> dict:
    """One real forward pass through the existing network — no training.

    The computation is exactly the forward pass of ``train()`` in
    training/training.py (``layer1`` -> relu -> ``layer2`` -> relu -> ``layer3``)
    applied to one sample of the existing dataset, inside ``torch.no_grad()``
    and without any optimiser call, so no parameter can change.
    """
    dataset = sample_dataset(module)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("the MNIST training split is empty")

    sample_index = index % total
    image, label = dataset[sample_index]
    true_label = int(label)

    layers = linear_layers(module)
    if len(layers) != 3:
        raise RuntimeError(
            f"training/training.py is expected to chain 3 linear layers, found {len(layers)}"
        )
    layer1, layer2, layer3 = layers

    with torch.no_grad():
        # Same reshape as train(): batch size 1, flattened 28 x 28 image.
        X = image.view(-1, 784)
        A1 = torch.relu(layer1(X))
        A2 = torch.relu(layer2(A1))
        logits = layer3(A2)
        probabilities = torch.softmax(logits, dim=1)
        # The existing loss function from training.py, evaluated for this
        # sample. No backward pass is run.
        loss = module.loss_fn(logits, torch.tensor([true_label], dtype=torch.long))

    predicted_label = int(torch.argmax(probabilities, dim=1))
    width, height = image.shape[-1], image.shape[-2]

    return {
        "type": "inference",
        "sample": {
            "index": sample_index,
            "total": total,
            "width": width,
            "height": height,
            "pixels": encode_pixels(image),
        },
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "logits": encode_tensor(logits.view(-1)),
        "probabilities": encode_tensor(probabilities.view(-1)),
        # One array per network layer, so the frontend can tint every neuron:
        # normalised pixels, ReLU outputs of Hidden 1/2, softmax output.
        "activations": [
            encode_tensor(tensor.view(-1))
            for tensor in (X, A1, A2, probabilities)
        ],
        "activations_note": (
            "Input layer: normalised pixels; hidden layers: ReLU outputs; "
            "output layer: softmax probabilities."
        ),
        "loss": float(loss),
        "weights_checksum": weights_checksum(layers),
        "weights_state": "untrained",
        "encoding": {
            "pixels": PIXEL_ENCODING,
            "activations": DTYPE,
            "byte_order": BYTE_ORDER,
            "activations_shape_order": "row-major per layer, layer order",
        },
        "message": (
            "Forward pass only: no gradients were computed and no optimiser "
            "step was taken, so every weight is unchanged."
        ),
    }


def encode_indices(array: np.ndarray) -> str:
    """Connection indices as base64 uint16 (every layer size fits in 16 bits)."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<u2").tobytes()).decode("ascii")


def select_partial_indices(module: ModuleType, k: int = PARTIAL_TOP_K) -> list[np.ndarray]:
    """The Partial view's connections, chosen once from the model's weights.

    One integer array of shape ``[out][k]`` per layer: for every neuron, the
    ``k`` incoming connections with the largest |weight| at the moment this is
    called (strongest first, ties broken by ascending input index, so the result
    is deterministic). The training session calls it when the model is loaded and
    again after a reset, which restores those same weights and therefore derives
    the same indices.

    Nothing recomputes the selection while training runs: the indices are fixed
    for the whole run, so the browser keeps drawing the same 1,010 lines and only
    their values change.
    """
    selection = []
    for layer in linear_layers(module):
        weights = layer.weight.detach().cpu().numpy()
        limit = int(min(k, int(layer.in_features)))
        order = np.argsort(-np.abs(weights), axis=1, kind="stable")[:, :limit]
        selection.append(np.ascontiguousarray(order, dtype=np.int64))
    return selection


def visible_weight_layers(module: ModuleType, selection: list[np.ndarray]) -> list[dict]:
    """The current weights of the connections the Partial view draws.

    `selection` is what `select_partial_indices()` returned at load time. The
    indices are never recomputed, so each frame describes exactly the same 1,010
    connections; only the values read at those indices move during training. A
    connection that was strong at load keeps being sent even after other weights
    overtake it, which is what makes a fixed set of lines watchable.
    """
    layers = linear_layers(module)
    if len(selection) != len(layers):
        raise ValueError(
            f"selection covers {len(selection)} layers, model has {len(layers)}"
        )

    encoded = []
    for index, (layer, chosen) in enumerate(zip(layers, selection)):
        weights = layer.weight.detach().cpu().numpy()
        order = np.ascontiguousarray(chosen, dtype=np.int64)
        out_size, in_size = weights.shape
        if order.ndim != 2 or order.shape[0] != out_size:
            raise ValueError(
                f"layer {index}: selection {order.shape} does not match {weights.shape}"
            )
        if order.size and (int(order.min()) < 0 or int(order.max()) >= in_size):
            raise ValueError(f"layer {index}: selection has an out-of-range input index")

        values = np.take_along_axis(weights, order, axis=1)
        encoded.append(
            {
                "index": index,
                "in": int(in_size),
                "out": int(out_size),
                "k": int(order.shape[1]),
                "indices": encode_indices(order),
                "values": base64.b64encode(
                    np.ascontiguousarray(values, dtype="<f4").tobytes()
                ).decode("ascii"),
                "visible_connections": int(order.size),
            }
        )
    return encoded


def build_training_step(
    module: ModuleType,
    result: dict,
    *,
    run: int,
    step: int,
    total_steps: int,
    epochs: int,
    train_size: int,
    speed: str,
    frame_every: int,
    checksum_before: str | None,
    selection: list[np.ndarray],
) -> dict:
    """One completed optimiser step of training/training.py, ready for the page.

    `result` is what ``train_step()`` returned for this step. Its forward-pass
    tensors were produced *before* ``optimiser.step()`` ran, so the pixels,
    prediction, softmax probabilities, activations and loss in this payload all
    describe the model as it was when the step was taken. The weights below are
    read from the layers afterwards, i.e. the parameters the step produced.

    `selection` is the session's fixed Partial-view selection, so every frame
    carries the same 1,010 connections and only their values move.
    """
    layers = linear_layers(module)
    X, A1, A2, Y = result["X"], result["A1"], result["A2"], result["Y"]
    loss = result["loss"]
    image = result["images"][0]
    true_label = int(result["labels"][0])

    with torch.no_grad():
        probabilities = torch.softmax(Y, dim=1)
        predicted_label = int(torch.argmax(probabilities, dim=1))

    width, height = int(image.shape[-1]), int(image.shape[-2])

    return {
        "type": "training_step",
        "run": run,
        "step": step,
        "total_steps": total_steps,
        "epoch": min(epochs - 1, (step - 1) // train_size) if step > 0 else 0,
        "epochs": epochs,
        "speed": speed,
        "frame_every": frame_every,
        "sample": {
            # Position of this image in the current pass over the shuffled
            # DataLoader: one optimiser step trained exactly this one sample.
            "position": (step - 1) % train_size + 1,
            "total": train_size,
            "width": width,
            "height": height,
            "pixels": encode_pixels(image),
        },
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "logits": encode_tensor(Y.view(-1)),
        "probabilities": encode_tensor(probabilities.view(-1)),
        "activations": [
            encode_tensor(tensor.view(-1)) for tensor in (X, A1, A2, probabilities)
        ],
        "activations_note": (
            "Input layer: normalised pixels; hidden layers: ReLU outputs; "
            "output layer: softmax probabilities — all read before the update."
        ),
        "loss": float(loss.detach()),
        "capture": {
            "image": "the MNIST sample this optimiser step trained",
            "prediction": "before the optimiser update",
            "probabilities": "before the optimiser update",
            "activations": "before the optimiser update",
            "loss": "before the optimiser update",
            "weights": "after the optimiser update",
            "note": (
                "The image, prediction, probabilities, activations and loss come "
                "from the forward pass that this step backpropagated. The weights "
                "were read from the layers after optimiser.step(), so the drawn "
                "colours are the newest real parameters."
            ),
        },
        "weights": {
            "selection": (
                f"the same {PARTIAL_TOP_K} connections per neuron in every frame: "
                "the top-K incoming |weight| of the initial model, fixed for the run"
            ),
            "step": step,
            "checksum_before": checksum_before,
            "checksum_after": weights_checksum(layers),
            "layers": visible_weight_layers(module, selection),
        },
    }


def summarise(model_info: dict) -> str:
    """Human-readable model summary for the startup log."""
    sizes = " -> ".join(str(layer["size"]) for layer in model_info["architecture"])
    lines = [f"Model: {sizes}"]
    for layer in model_info["layers"]:
        lines.append(
            f"  layer {layer['index'] + 1}: {layer['in']} -> {layer['out']} "
            f"({layer['weight_count']:,} weights, {layer['bias_count']} biases)"
        )
    totals = model_info["totals"]
    lines.append(
        f"Totals: {totals['neurons']:,} neurons, {totals['connections']:,} "
        f"connections, {totals['parameters']:,} parameters"
    )
    lines.append(
        f"Weights: {model_info['weights_state']} from {model_info['weights_source']} "
        f"(sha256 {model_info['weights_checksum'][:16]})"
    )
    return "\n".join(lines)
