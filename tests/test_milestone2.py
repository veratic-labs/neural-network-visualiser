"""Milestone 2 verification: the real PyTorch model reaches the browser.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone2.py

Checks, in order:
  1. the backend payload matches the existing PyTorch model exactly
  2. the FastAPI server serves the frontend and answers on /ws
  3. a WebSocket client receives the same real weights and biases
  4. the frontend no longer generates mock connection weights

The script starts its own server on port 8123 and shuts it down afterwards.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = 8123
BASE_URL = f"http://127.0.0.1:{PORT}"
SOCKET_URL = f"ws://127.0.0.1:{PORT}/ws"

# Make `backend` importable no matter where the script is started from.
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


def training_module():
    from backend.visualiser import load_training_module

    return load_training_module()


def check_model_payload():
    """Part 1: the payload is derived from the real model, not hard-coded."""
    from backend.visualiser import build_model_info, linear_layers

    module = training_module()
    layers = linear_layers(module)
    info = build_model_info(module)

    shapes = [(layer.in_features, layer.out_features) for layer in layers]
    check(
        shapes == [(784, 128), (128, 64), (64, 10)],
        "existing model has 784->128->64->10 linear layers",
        f"got {shapes}",
    )
    check(
        callable(getattr(module, "train", None)),
        "training code is importable and exposes train()",
    )
    check(
        [layer["size"] for layer in info["architecture"]] == [784, 128, 64, 10],
        "layer sizes come from the nn.Linear modules",
    )
    check(
        [layer["name"] for layer in info["architecture"]]
        == ["Input", "Hidden 1", "Hidden 2", "Output"],
        "architecture is labelled Input/Hidden n/Output",
    )
    check(
        info["totals"] == {"neurons": 986, "connections": 109184, "parameters": 109386},
        "totals are 986 neurons / 109,184 connections / 109,386 parameters",
        f"got {info['totals']}",
    )
    check(
        info["weights_state"] == "untrained"
        and info["total_steps"] == 60000
        and info["batch_size"] == 1,
        "batch size and total steps come from training.py",
        f"got {info['weights_state']}/{info['total_steps']}/{info['batch_size']}",
    )

    weights_match = True
    biases_match = True
    within_bounds = True
    for layer_info, layer in zip(info["layers"], layers):
        weights = decode_float32(layer_info["weights"])
        expected = layer.weight.detach().numpy().reshape(-1)
        weights_match &= weights.size == expected.size and bool(np.array_equal(weights, expected))

        biases = decode_float32(layer_info["biases"])
        biases_match &= bool(np.array_equal(biases, layer.bias.detach().numpy().reshape(-1)))

        bound = 1.0 / (layer.in_features ** 0.5)
        within_bounds &= bool(np.abs(weights).max() <= bound + 1e-6)

    check(weights_match, "every encoded weight matches layer.weight exactly")
    check(biases_match, "every encoded bias matches layer.bias exactly")
    check(within_bounds, "weights are inside the nn.Linear bound +/-1/sqrt(fan_in)")
    check(
        len(base64.b64decode(info["layers"][0]["weights"])) == 100352 * 4,
        "weights travel as float32 bytes (4 bytes per connection)",
    )
    check(
        checksum_of(info) == info["weights_checksum"],
        "payload checksum matches the encoded weight bytes",
    )
    check(
        len(json.dumps(info)) < 2_000_000,
        "model_info payload stays under 2 MB",
        f"{len(json.dumps(info)):,} bytes",
    )

    return info, layers


def start_server() -> subprocess.Popen:
    """Part 2: launch the real server on a test port and wait until it answers."""
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
        "GET / serves index.html",
        f"{status} {content_type}",
    )

    for path, expected in (
        ("/style.css", "css"),
        ("/main.js", "javascript"),
        ("/network.js", "javascript"),
        ("/mock.js", "javascript"),
    ):
        status, content_type, _ = fetch(path)
        check(
            status == 200 and expected in content_type,
            f"GET {path} is served as {expected}",
            f"{status} {content_type}",
        )


def checksum_of(message: dict) -> str:
    """Recompute the server's fingerprint from the bytes that actually arrived."""
    digest = hashlib.sha256()
    for layer in message["layers"]:
        digest.update(base64.b64decode(layer["weights"]))
        digest.update(base64.b64decode(layer["biases"]))
    return digest.hexdigest()


def read_model_info():
    """One WebSocket session: the first message, plus anything that follows it."""
    import websockets

    async def session():
        async with websockets.connect(SOCKET_URL, open_timeout=30) as socket:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            try:
                extra = await asyncio.wait_for(socket.recv(), timeout=2)
            except asyncio.TimeoutError:
                extra = None
            return message, extra

    return asyncio.run(session())


def check_websocket(expected_info: dict, layers: list) -> None:
    """Part 3: the browser-facing channel carries the server's real weights."""
    message, extra = read_model_info()

    check(message.get("type") == "model_info", "first WebSocket message is model_info")
    check(
        [layer["size"] for layer in message["architecture"]] == [784, 128, 64, 10],
        "WebSocket reports the model's 784->128->64->10 layer sizes",
        f"got {[layer['size'] for layer in message['architecture']]}",
    )
    check(message["totals"] == expected_info["totals"], "WebSocket totals match the model")
    check(
        bool(message["weights_checksum"]) and checksum_of(message) == message["weights_checksum"],
        "received weight bytes match the server's sha256 checksum",
        f"{checksum_of(message)[:16]} != {message['weights_checksum'][:16]}",
    )

    received = [decode_float32(layer["weights"]) for layer in message["layers"]]
    bounds_ok = True
    sizes_ok = True
    for values, layer_info in zip(received, message["layers"]):
        sizes_ok &= values.size == layer_info["in"] * layer_info["out"]
        bounds_ok &= bool(np.abs(values).max() <= 1.0 / (layer_info["in"] ** 0.5) + 1e-6)
    check(sizes_ok, "each received matrix has in x out real weights")
    check(bounds_ok, "received weights are real nn.Linear initial values")

    local = layers[0].weight.detach().numpy().reshape(-1)
    check(
        not np.array_equal(received[0], local),
        "weights come from the server process, not from the test's own model copy",
    )

    second, _ = read_model_info()
    check(
        [layer["weights"] for layer in second["layers"]]
        == [layer["weights"] for layer in message["layers"]],
        "a second connection receives the same weights (no training between)",
    )

    check(
        extra is None,
        "socket stays open without further messages (nothing is training)",
        repr(extra)[:100],
    )


def check_frontend_sources() -> None:
    """Part 4: the frontend draws backend weights, not mock weights."""
    main_js = (REPO_ROOT / "frontend" / "main.js").read_text()
    mock_js = (REPO_ROOT / "frontend" / "mock.js").read_text()
    html = (REPO_ROOT / "frontend" / "index.html").read_text()

    check("createMockWeights" not in mock_js, "mock.js no longer generates weights")
    check(
        "weights" not in mock_js.split("export function createMockState")[1].lower(),
        "mock.js exports no weight data",
    )
    check("new WebSocket" in main_js, "main.js opens the WebSocket")
    check("decodeFloat32" in main_js, "main.js decodes the base64 float32 weights")
    check("view.setModel(" in main_js, "main.js passes the received model to the renderer")
    check(
        "status-badge" in html and "model-badge" in html,
        "index.html exposes the status and model badges",
    )


def main() -> int:
    info, layers = check_model_payload()
    check_frontend_sources()

    process = start_server()
    try:
        check_http()
        check_websocket(info, layers)
    finally:
        process.terminate()
        try:
            log = process.communicate(timeout=30)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            log = process.communicate()[0] or ""

    check("sent model_info" in log, "server logged sending model_info")
    check("Trained:" not in log, "no training progress was printed")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1

    print(f"All Milestone 2 checks passed ({len(info['layers'])} layers, "
          f"{info['totals']['connections']:,} real weights).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
