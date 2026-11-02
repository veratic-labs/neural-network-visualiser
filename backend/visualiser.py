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


def build_model_info(module: ModuleType) -> dict:
    """The `model_info` payload sent to every new WebSocket client."""
    layers = linear_layers(module)
    architecture = describe_architecture(layers)

    encoded_layers = []
    connections = 0
    parameters = 0

    for index, layer in enumerate(layers):
        weight_bytes = tensor_bytes(layer.weight)
        bias_bytes = tensor_bytes(layer.bias)

        weight_count = layer.out_features * layer.in_features
        connections += weight_count
        parameters += weight_count + layer.out_features
        encoded_layers.append(
            {
                "index": index,
                "in": layer.in_features,
                "out": layer.out_features,
                "weights": base64.b64encode(weight_bytes).decode("ascii"),
                "biases": base64.b64encode(bias_bytes).decode("ascii"),
                "weight_count": weight_count,
                "bias_count": layer.out_features,
            }
        )

    total_steps = len(module.train_data) if hasattr(module, "train_data") else None
    batch_size = getattr(module, "batch_size", None)

    return {
        "type": "model_info",
        "architecture": architecture,
        "layers": encoded_layers,
        "totals": {
            "neurons": sum(layer["size"] for layer in architecture),
            "connections": connections,
            "parameters": parameters,
        },
        "batch_size": batch_size,
        "total_steps": total_steps,
        "dtype": DTYPE,
        "byte_order": BYTE_ORDER,
        "shape_order": SHAPE_ORDER,
        "weights_checksum": weights_checksum(layers),
        "weights_source": str(TRAINING_FILE.relative_to(REPO_ROOT)),
        "weights_state": "untrained",
        "message": (
            "Initial weights of the existing PyTorch model; no training steps "
            "have been run yet."
        ),
    }


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
