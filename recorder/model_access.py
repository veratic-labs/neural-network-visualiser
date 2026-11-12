"""Model and dataset helpers for the offline recorder.

These helpers read the existing PyTorch model and the existing MNIST dataset so
the recorder can serialise what it sees. The network itself is **not**
re-implemented here: `training/training.py` is imported from its file path, so
this module sees exactly the layers the training script builds (layer1, layer2,
layer3), their real initial weights and the training split it already loads. The
training loop in that file sits behind an ``if __name__ == "__main__":`` guard,
so importing the module defines the network without starting a training run.

The functions were moved here from ``backend/visualiser.py`` during Cleanup 2 so
the recorder no longer imports the retired backend package. Signatures and
behaviour are unchanged; the recorder writes the same bytes as before. Only the
helpers the recorder calls are present — the JSON/WebSocket payload builders
stay with the backend until Cleanup 3 removes it.

`select_partial_indices()` chooses the Partial view's connections once from the
model's initial weights, so a recording only has to describe those connections
and always the same ones. `visible_weight_layers()` reads their current values,
and `tensor_bytes()` produces the raw little-endian float32 bytes the binary
recording stores.
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

# Mirrors PARTIAL_TOP_K in frontend/network.js: the reduced training view draws
# this many incoming connections per neuron. `select_partial_indices()` picks
# them once from the initial weights, so a recording only has to carry those
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


def weights_checksum(layers: list[nn.Linear]) -> str:
    """sha256 over every weight and bias byte, in layer order.

    The recording's manifest carries this checksum, so a reader (and the tests)
    can prove which parameters the recording started from.
    """
    digest = hashlib.sha256()
    for layer in layers:
        digest.update(tensor_bytes(layer.weight))
        digest.update(tensor_bytes(layer.bias))
    return digest.hexdigest()


def configured_epochs(module: ModuleType) -> int:
    """How many passes over the training split training.py runs.

    training.py has no epoch setting: ``train()`` makes exactly one pass over
    ``train_loader``, so one epoch is the configured value.
    """
    return max(1, int(getattr(module, "epochs", 1)))


def sample_dataset(module: ModuleType) -> object:
    """The MNIST training split that training/load_data.py already loads."""
    dataset = getattr(module, "train_data", None)
    if dataset is None:
        raise RuntimeError(f"{TRAINING_FILE} does not expose train_data")
    return dataset


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


def encode_pixels(image: torch.Tensor) -> str:
    """A 28 x 28 ``ToTensor()`` image in 0..1 as base64-encoded uint8 bytes.

    ``ToTensor()`` divides the MNIST bytes by 255, so multiplying back and
    rounding restores exactly the pixels stored in the dataset file.
    """
    values = image.detach().cpu().numpy().reshape(-1) * 255.0
    array = np.ascontiguousarray(np.rint(values).clip(0, 255), dtype=np.uint8)
    return base64.b64encode(array.tobytes()).decode("ascii")


def encode_indices(array: np.ndarray) -> str:
    """Connection indices as base64 uint16 (every layer size fits in 16 bits)."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<u2").tobytes()).decode("ascii")


def select_partial_indices(module: ModuleType, k: int = PARTIAL_TOP_K) -> list[np.ndarray]:
    """The Partial view's connections, chosen once from the model's weights.

    One integer array of shape ``[out][k]`` per layer: for every neuron, the
    ``k`` incoming connections with the largest |weight| at the moment this is
    called (strongest first, ties broken by ascending input index, so the result
    is deterministic). The recorder calls it once, before training starts.

    Nothing recomputes the selection while training runs: the indices are fixed
    for the whole run, so the recording keeps describing the same 1,010
    connections and only their values change.
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

    `selection` is what `select_partial_indices()` returned before training
    started. The indices are never recomputed, so every checkpoint describes
    exactly the same 1,010 connections; only the values read at those indices
    move during training. A connection that was strong at load keeps being
    recorded even after other weights overtake it.
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
