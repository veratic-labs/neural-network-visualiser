"""Milestone 3 verification: the page shows real inference data.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone3.py

Checks, in order:
  1. `build_inference()` takes its sample from the existing MNIST dataset
  2. its forward pass, softmax, prediction, loss and activations match an
     independent computation done here in the test
  3. a forward pass changes no weight and creates no gradient
  4. over the WebSocket, `model_info` is followed by a real sample, and
     `{"type": "next_sample"}` returns the next sample with the same weights
  5. the numbers on the wire can be reproduced from the transmitted pixels and
     the transmitted weights alone
  6. the frontend consumes those messages and no longer ships mock data

The script starts its own server on port 8124 and shuts it down afterwards.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = 8124
BASE_URL = f"http://127.0.0.1:{PORT}"
SOCKET_URL = f"ws://127.0.0.1:{PORT}/ws"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}{'  -> ' + detail if detail else ''}")
        FAILURES.append(label)


def decode_float32(value: str) -> np.ndarray:
    """Exactly what the browser does: base64 -> little-endian float32 values."""
    return np.frombuffer(base64.b64decode(value), dtype="<f4")


def decode_pixels(value: str) -> np.ndarray:
    """Exactly what the browser does: base64 -> one uint8 per pixel."""
    return np.frombuffer(base64.b64decode(value), dtype=np.uint8)


def training_module():
    from backend.visualiser import load_training_module

    return load_training_module()


def reference_forward(module, index: int):
    """The same forward pass, written independently of backend/visualiser.py."""
    image, label = module.train_data[index]
    with torch.no_grad():
        x = image.view(-1, 784)
        a1 = torch.relu(module.layer1(x))
        a2 = torch.relu(module.layer2(a1))
        logits = module.layer3(a2)
        probabilities = torch.softmax(logits, dim=1)
        loss = module.loss_fn(logits, torch.tensor([int(label)], dtype=torch.long))
    return {
        "image": image,
        "label": int(label),
        "x": x.numpy().reshape(-1),
        "a1": a1.numpy().reshape(-1),
        "a2": a2.numpy().reshape(-1),
        "logits": logits.numpy().reshape(-1),
        "probabilities": probabilities.numpy().reshape(-1),
        "loss": float(loss),
    }


def check_backend() -> tuple:
    """Parts 1-3: the payload holds real data and training state is untouched."""
    from backend.visualiser import build_inference, build_model_info, linear_layers

    module = training_module()
    layers = linear_layers(module)

    before = [layer.weight.detach().numpy().copy() for layer in layers]
    before_bias = [layer.bias.detach().numpy().copy() for layer in layers]
    checksum_before = build_model_info(module)["weights_checksum"]

    payload = build_inference(module, 0)
    expected = reference_forward(module, 0)

    check(payload["type"] == "inference", "inference payload has its own message type")
    check(
        payload["sample"]["index"] == 0 and payload["sample"]["total"] == len(module.train_data),
        "sample index and dataset size are reported",
        f"got {payload['sample']}",
    )
    check(
        payload["sample"]["width"] == 28 and payload["sample"]["height"] == 28,
        "sample image is 28 x 28",
    )

    pixels = decode_pixels(payload["sample"]["pixels"])
    dataset_bytes = np.rint(expected["image"].numpy().reshape(-1) * 255).astype(np.uint8)
    check(pixels.size == 784, "sample image has 784 pixels", str(pixels.size))
    check(
        np.array_equal(pixels, dataset_bytes),
        "sample pixels are the existing MNIST image from training/data/",
    )
    check(
        payload["true_label"] == expected["label"] == int(module.train_data[0][1]),
        "true label comes from the dataset",
        f"{payload['true_label']} != {expected['label']}",
    )

    probabilities = decode_float32(payload["probabilities"])
    logits = decode_float32(payload["logits"])
    check(
        np.allclose(logits, expected["logits"], atol=1e-5),
        "logits match the existing model's forward pass",
    )
    check(
        np.allclose(probabilities, expected["probabilities"], atol=1e-6),
        "probabilities are the softmax of those logits",
    )
    check(
        probabilities.size == 10
        and bool(np.all(probabilities >= 0))
        and abs(float(probabilities.sum()) - 1.0) < 1e-5,
        "10 non-negative probabilities that sum to 1",
        f"sum {float(probabilities.sum())}",
    )
    check(
        payload["predicted_label"] == int(np.argmax(probabilities)),
        "predicted digit is argmax(softmax)",
        f"{payload['predicted_label']}",
    )
    check(
        payload["correct"] == (payload["predicted_label"] == payload["true_label"]),
        "correct flag agrees with the two labels",
    )
    check(
        abs(payload["loss"] - expected["loss"]) < 1e-6,
        "loss is the model's cross-entropy for this sample",
        f"{payload['loss']} != {expected['loss']}",
    )

    activations = [decode_float32(values) for values in payload["activations"]]
    sizes = [values.size for values in activations]
    check(sizes == [784, 128, 64, 10], "one activation per neuron in every layer", str(sizes))
    check(
        np.allclose(activations[0], expected["x"], atol=1e-6),
        "input activations are the normalised pixels",
    )
    check(
        np.allclose(activations[1], expected["a1"], atol=1e-5),
        "hidden 1 activations are the ReLU outputs of layer1",
    )
    check(
        np.allclose(activations[2], expected["a2"], atol=1e-5),
        "hidden 2 activations are the ReLU outputs of layer2",
    )
    check(
        np.allclose(activations[3], probabilities, atol=1e-6),
        "output activations are the softmax probabilities",
    )
    check(
        bool(np.all(activations[1] >= 0)) and bool(np.all(activations[2] >= 0)),
        "hidden activations are non-negative (ReLU)",
    )

    after = [layer.weight.detach().numpy() for layer in layers]
    after_bias = [layer.bias.detach().numpy() for layer in layers]
    check(
        all(np.array_equal(a, b) for a, b in zip(before, after))
        and all(np.array_equal(a, b) for a, b in zip(before_bias, after_bias)),
        "the forward pass leaves every weight and bias unchanged",
    )
    check(
        payload["weights_checksum"] == checksum_before,
        "the payload's weight checksum equals the model_info checksum",
    )
    check(
        all(layer.weight.grad is None and layer.bias.grad is None for layer in layers),
        "no gradient was created (no backward pass)",
    )
    check(
        len(json.dumps(payload)) < 200_000,
        "inference payload is small",
        f"{len(json.dumps(payload)):,} bytes",
    )

    second = build_inference(module, 1)
    check(
        second["sample"]["index"] == 1
        and second["sample"]["pixels"] != payload["sample"]["pixels"],
        "a different index yields a different MNIST sample",
    )
    check(
        second["activations"] != payload["activations"],
        "a different sample yields different activations",
    )
    check(
        build_inference(module, len(module.train_data))["sample"]["index"] == 0,
        "sample index wraps back to the first image after the dataset end",
    )
    check(
        second["weights_checksum"] == payload["weights_checksum"],
        "sampling more images still does not change the weights",
    )

    return module, layers



def start_server() -> subprocess.Popen:
    """Part 4: launch the real server on a test port and wait until it answers."""
    environment = dict(os.environ, VISUALISER_HOST="127.0.0.1", VISUALISER_PORT=str(PORT))
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.server"],
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 90
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"server exited during startup:\n{output}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/", timeout=2) as response:
                if response.status == 200:
                    return process
        except OSError:
            time.sleep(0.25)

    process.kill()
    raise RuntimeError("server did not answer within 90 s")


def fetch(path: str):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=15) as response:
        return (
            response.status,
            response.headers.get("content-type", ""),
            response.read().decode("utf-8"),
        )


def check_http() -> None:
    status, content_type, body = fetch("/")
    check(
        status == 200 and "text/html" in content_type and "network-canvas" in body,
        "GET / serves the page",
        f"{status} {content_type}",
    )

    for element in (
        "btn-next-sample",
        "sample-canvas",
        "sample-index",
        "sample-true",
        "sample-predicted",
        "output-list",
        "metric-loss",
        "progress-bar",
        "mode-all",
        "mode-partial",
    ):
        check(f'id="{element}"' in body, f"index.html still provides #{element}")

    check("tag-mock" not in body, "no panel is labelled mock any more")

    try:
        fetch("/mock.js")
        removed = False
    except urllib.error.HTTPError as error:
        removed = error.code == 404
    check(removed, "the placeholder mock.js is no longer served")


def layers_from_model_info(info: dict) -> tuple[list, list]:
    """The transmitted weights as [out][in] matrices and [out] bias vectors."""
    weights = []
    biases = []
    for layer in info["layers"]:
        weights.append(decode_float32(layer["weights"]).reshape(layer["out"], layer["in"]))
        biases.append(decode_float32(layer["biases"]))
    return weights, biases


def reproduce_forward(weights: list, biases: list, pixels: np.ndarray) -> tuple:
    """The whole network, recomputed from the transmitted weights and pixels."""
    x = pixels.astype(np.float64) / 255.0
    a1 = np.maximum(x @ weights[0].astype(np.float64).T + biases[0], 0.0)
    a2 = np.maximum(a1 @ weights[1].astype(np.float64).T + biases[1], 0.0)
    logits = a2 @ weights[2].astype(np.float64).T + biases[2]
    shifted = np.exp(logits - logits.max())
    return a1, a2, logits, shifted / shifted.sum()


def read_three_messages():
    """model_info, the sample sent with it, and the reply to `next_sample`."""
    import websockets

    async def session():
        async with websockets.connect(SOCKET_URL, open_timeout=30) as socket:
            info = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            first = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            await socket.send(json.dumps({"type": "next_sample"}))
            second = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            return info, first, second

    return asyncio.run(session())


def check_websocket(module) -> None:
    """Part 5: the wire carries real, reproducible inference data."""
    info, first, second = read_three_messages()
    weights, biases = layers_from_model_info(info)

    check(info.get("type") == "model_info", "the socket still opens with model_info")
    check(
        first.get("type") == "inference",
        "the very next message is a real inference sample",
        repr(first.get("type")),
    )
    check(
        first["sample"]["index"] == 0,
        "the page opens on the first MNIST training sample",
        str(first["sample"]["index"]),
    )

    pixels = decode_pixels(first["sample"]["pixels"])
    image, label = module.train_data[0]
    expected_pixels = np.rint(image.numpy().reshape(-1) * 255).astype(np.uint8)
    check(
        np.array_equal(pixels, expected_pixels),
        "the transmitted image is the real MNIST sample, pixel for pixel",
    )
    check(
        first["true_label"] == int(label),
        "the transmitted true label matches the dataset",
        f"{first['true_label']} != {int(label)}",
    )

    a1, a2, logits, probabilities = reproduce_forward(weights, biases, pixels)
    sent_logits = decode_float32(first["logits"])
    sent_probabilities = decode_float32(first["probabilities"])
    check(
        np.allclose(sent_logits, logits, atol=1e-3),
        "logits follow from the transmitted weights and pixels",
        f"max diff {float(np.abs(sent_logits - logits).max()):.2e}",
    )
    check(
        np.allclose(sent_probabilities, probabilities, atol=1e-5),
        "probabilities are the softmax of those transmitted logits",
    )
    check(
        abs(float(sent_probabilities.sum()) - 1.0) < 1e-5,
        "transmitted probabilities sum to 1",
        f"{float(sent_probabilities.sum())}",
    )
    check(
        int(np.argmax(probabilities)) == first["predicted_label"],
        "the displayed prediction is argmax of the transmitted probabilities",
    )
    check(
        abs(first["loss"] + float(np.log(probabilities[first["true_label"]]))) < 2e-3,
        "the displayed loss is -log p(true class)",
        f"{first['loss']:.6f}",
    )

    activations = [decode_float32(values) for values in first["activations"]]
    check(
        [values.size for values in activations] == [784, 128, 64, 10],
        "activations cover all 986 neurons",
        str([values.size for values in activations]),
    )
    check(
        np.allclose(activations[1], a1, atol=1e-3)
        and np.allclose(activations[2], a2, atol=1e-3),
        "hidden activations are the real ReLU outputs of the transmitted weights",
    )
    check(
        np.allclose(activations[3], probabilities, atol=1e-5),
        "output activations are the transmitted probabilities",
    )
    check(
        first["weights_checksum"] == info["weights_checksum"],
        "the first sample left the weights untouched",
    )

    check(
        second["sample"]["index"] == 1,
        "next_sample returns the following MNIST sample",
        str(second["sample"]["index"]),
    )
    check(
        second["sample"]["pixels"] != first["sample"]["pixels"],
        "the image changes when the sample changes",
    )
    check(
        second["true_label"] == int(module.train_data[1][1]),
        "the second sample's label comes from the dataset",
        f"{second['true_label']} != {int(module.train_data[1][1])}",
    )
    check(
        second["activations"] != first["activations"]
        and second["probabilities"] != first["probabilities"],
        "the activations and probabilities change with the sample",
    )
    check(
        second["weights_checksum"] == info["weights_checksum"],
        "next_sample still leaves every weight unchanged",
    )


def check_frontend_sources() -> None:
    """Part 6: the frontend consumes the real sample instead of mock data."""
    main_js = (REPO_ROOT / "frontend" / "main.js").read_text()
    network_js = (REPO_ROOT / "frontend" / "network.js").read_text()
    html = (REPO_ROOT / "frontend" / "index.html").read_text()

    check('"next_sample"' in main_js, "main.js sends the next_sample command")
    check(
        '"inference"' in main_js and "handleInference" in main_js,
        "main.js handles inference messages",
    )
    check(
        "decodePixels" in main_js and 'from "./panels.js"' in main_js,
        "main.js draws the received MNIST pixels",
    )
    panels_js = (REPO_ROOT / "frontend" / "panels.js").read_text()
    check(
        "export function drawSampleImage(" in panels_js and "putImageData" in panels_js,
        "panels.js paints the 28 x 28 sample canvas",
    )
    check(
        "setActivations(" in main_js and "setActivations(" in network_js,
        "the activations reach the canvas renderer",
    )
    check(
        "ACTIVATION_COLOUR" in network_js
        and "POSITIVE_COLOUR" in network_js
        and "NEGATIVE_COLOUR" in network_js,
        "neurons keep a dedicated activation colour, weights keep theirs",
    )
    check(
        "PARTIAL_TOP_K" in network_js and "setMode" in network_js,
        "Full and Partial connection modes are untouched",
    )
    check(
        not (REPO_ROOT / "frontend" / "mock.js").exists()
        and "mock.js" not in main_js
        and "mock.js" not in html,
        "no mock data module is loaded any more",
    )


def main() -> int:
    module, _layers = check_backend()
    check_frontend_sources()

    process = start_server()
    try:
        check_http()
        check_websocket(module)
    finally:
        process.terminate()
        try:
            log = process.communicate(timeout=30)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            log = process.communicate()[0] or ""

    check("sent inference" in log, "the server logged sending inference samples")
    check("Trained:" not in log, "no training progress was printed")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1

    print("All Milestone 3 checks passed (real sample, softmax, activations, next_sample).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
