"""Milestone 4 verification: real-time training visualisation.

Run from the repository root with the project virtualenv:

    ./venv/bin/python tests/test_milestone4.py

Checks, in order:
  1. `train_step()` in training/training.py is one real optimiser step
  2. `build_training_step()` describes that step honestly: the loss, logits,
     probabilities and activations come from the forward pass that the step
     backpropagated, while the weights come from after `optimiser.step()`
  3. the Partial view's 1,010 connections are fixed: they are chosen once from
     the initial weights and the same indices travel with every frame, while only
     their values change (a weight another connection overtakes stays on screen)
  4. `restore_parameters()` puts the initial weights back and clears Adam
  5. `Outbox` keeps control payloads in order and replaces stale frames
  6. over the WebSocket, training never starts on its own, `start` really moves
     the weights one step at a time, progress and sample counts follow the real
     dataset, pause stops at a step boundary, reset restores the initial weights
     and fast mode still processes every sample while sending fewer frames
  7. the frontend decodes and shows those messages, and draws every weight on one
     fixed scale (MIN/MAX_DISPLAY_WEIGHT) in both Full and Partial mode
  8. the page uses a light theme and a fixed diverging weight scale (dark red for
     negative weights, grey at zero, dark purple for positive ones), with a legend
     painted from the renderer's own colour function

The script starts its own server on port 8125 and shuts it down afterwards.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = 8125
BASE_URL = f"http://127.0.0.1:{PORT}"
SOCKET_URL = f"ws://127.0.0.1:{PORT}/ws"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []

# What training/training.py really defines.
INPUT_SIZE = 784
LAYER_SIZES = (128, 64, 10)
TRAIN_SIZE = 60_000
EPOCHS = 1
PARTIAL_TOP_K = 5


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}{'  -> ' + detail if detail else ''}")
        FAILURES.append(label)


def decode_float32(value: str) -> np.ndarray:
    """Exactly what the browser does: base64 -> little-endian float32 values."""
    return np.frombuffer(base64.b64decode(value), dtype="<f4")


def decode_uint16(value: str) -> np.ndarray:
    """Exactly what the browser does for the partial-weight indices."""
    return np.frombuffer(base64.b64decode(value), dtype="<u2")


def decode_pixels(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype=np.uint8)


def frame_indices(message: dict) -> list[np.ndarray]:
    """The connection indices of one training frame, per layer as [out][k]."""
    return [
        decode_uint16(layer["indices"]).reshape(layer["out"], layer["k"])
        for layer in message["weights"]["layers"]
    ]


def same_indices(frames: list[dict], expected: list[np.ndarray]) -> bool:
    """True when every frame describes exactly the `expected` connection indices."""
    return all(
        np.array_equal(indices, wanted)
        for frame in frames
        for indices, wanted in zip(frame_indices(frame), expected)
    )


def training_module():
    from backend.visualiser import load_training_module

    return load_training_module()


def reference_forward(weights: list, biases: list, pixels: np.ndarray) -> tuple:
    """The whole network recomputed from given weights and raw pixel bytes."""
    x = pixels.astype(np.float64) / 255.0
    a1 = np.maximum(x @ weights[0].astype(np.float64).T + biases[0], 0.0)
    a2 = np.maximum(a1 @ weights[1].astype(np.float64).T + biases[1], 0.0)
    logits = a2 @ weights[2].astype(np.float64).T + biases[2]
    shifted = np.exp(logits - logits.max())
    return a1, a2, logits, shifted / shifted.sum()


def layer_arrays(layers) -> tuple[list, list]:
    """The live weights as [out][in] arrays and the biases as [out] arrays."""
    weights = [layer.weight.detach().cpu().numpy().copy() for layer in layers]
    biases = [layer.bias.detach().cpu().numpy().copy() for layer in layers]
    return weights, biases


def check_local_maths() -> None:
    """Parts 1-4: one real optimiser step, described honestly."""
    from backend.visualiser import (
        build_model_info,
        build_training_step,
        linear_layers,
        restore_parameters,
        select_partial_indices,
        snapshot_parameters,
        training_totals,
        visible_weight_layers,
        weights_checksum,
    )

    module = training_module()
    layers = linear_layers(module)
    total_steps, epochs, train_size = training_totals(module)

    check(
        (total_steps, epochs, train_size) == (TRAIN_SIZE * EPOCHS, EPOCHS, TRAIN_SIZE),
        "progress totals are the real dataset size times the configured epochs",
        f"{total_steps}, {epochs}, {train_size}",
    )
    check(
        [layer.in_features for layer in layers] == [INPUT_SIZE, 128, 64]
        and [layer.out_features for layer in layers] == [128, 64, 10],
        "the trained network is the existing 784-128-64-10 MLP",
    )

    initial = snapshot_parameters(module)
    before_weights, before_biases = layer_arrays(layers)
    before_checksum = weights_checksum(layers)

    # The Partial view's connections are chosen once, when the model is loaded:
    # every later frame must describe these very indices.
    selection = select_partial_indices(module)
    expected_selection = [
        np.argsort(-np.abs(weights), axis=1, kind="stable")[:, :PARTIAL_TOP_K]
        for weights in before_weights
    ]
    check(
        [chosen.shape for chosen in selection] == [(128, 5), (64, 5), (10, 5)]
        and all(
            np.array_equal(chosen, expected)
            for chosen, expected in zip(selection, expected_selection)
        ),
        "the partial selection is the load-time top-5 |weight| of every neuron",
        str([chosen.shape for chosen in selection]),
    )

    images, labels = next(iter(module.train_loader))
    result = module.train_step(images, labels)

    check(
        {"X", "A1", "A2", "Y", "loss", "images", "labels"} <= set(result),
        "train_step() returns the tensors of the step it took",
        str(sorted(result)),
    )
    after_checksum = weights_checksum(layers)
    check(
        after_checksum != before_checksum,
        "one train_step() really updates the weights",
        after_checksum,
    )

    frame = build_training_step(
        module,
        result,
        run=11,
        step=1,
        total_steps=total_steps,
        epochs=epochs,
        train_size=train_size,
        speed="slow",
        frame_every=1,
        checksum_before=before_checksum,
        selection=selection,
    )

    check(frame["type"] == "training_step", "the frame has its own message type")
    check(
        frame["run"] == 11 and frame["step"] == 1 and frame["epoch"] == 0,
        "the frame reports the run, step and epoch it belongs to",
        f"run {frame['run']}, step {frame['step']}, epoch {frame['epoch']}",
    )
    check(
        frame["total_steps"] == total_steps and frame["epochs"] == epochs,
        "the frame carries the real totals for the progress bar",
    )
    check(
        frame["sample"]["position"] == 1 and frame["sample"]["total"] == train_size,
        "the frame names the one sample this step trained",
        str(frame["sample"].get("position")),
    )

    pixels = decode_pixels(frame["sample"]["pixels"])
    trained_image = result["images"][0].detach().cpu().numpy().reshape(-1)
    check(
        np.array_equal(pixels, np.rint(trained_image * 255).astype(np.uint8)),
        "the frame's image is the MNIST sample the optimiser step trained",
    )

    # The numbers must describe the model as it was when the step was taken.
    a1, a2, logits, probabilities = reference_forward(before_weights, before_biases, pixels)
    check(
        np.allclose(decode_float32(frame["logits"]), logits, atol=1e-3),
        "the frame's logits come from the weights before the update",
        f"max diff {float(np.abs(decode_float32(frame['logits']) - logits).max()):.2e}",
    )
    check(
        np.allclose(decode_float32(frame["probabilities"]), probabilities, atol=1e-5),
        "the frame's probabilities are the softmax of those logits",
    )
    check(
        frame["predicted_label"] == int(np.argmax(probabilities))
        and frame["correct"] == (frame["predicted_label"] == frame["true_label"]),
        "the frame's prediction is the argmax before the update",
    )
    check(
        abs(frame["loss"] + float(np.log(probabilities[frame["true_label"]]))) < 2e-3,
        "the frame's loss is -log p(true class) of the pre-update model",
        f"{frame['loss']:.6f}",
    )
    check(
        frame["true_label"] == int(result["labels"][0]),
        "the frame's true label is the label the loss was computed against",
    )

    activations = [decode_float32(values) for values in frame["activations"]]
    check(
        [values.size for values in activations] == [INPUT_SIZE, *LAYER_SIZES],
        "activations cover all 986 neurons",
        str([values.size for values in activations]),
    )
    check(
        np.allclose(activations[1], a1, atol=1e-3) and np.allclose(activations[2], a2, atol=1e-3),
        "hidden activations are the real ReLU outputs before the update",
    )

    # The weights, in contrast, must be the ones the step produced.
    after_weights, _ = layer_arrays(layers)
    check(
        frame["weights"]["checksum_before"] == before_checksum
        and frame["weights"]["checksum_after"] == after_checksum,
        "the frame records the weights either side of the update",
    )

    selections_match = True
    visible = []
    for index, layer in enumerate(frame["weights"]["layers"]):
        indices = decode_uint16(layer["indices"]).reshape(layer["out"], layer["k"])
        values = decode_float32(layer["values"]).reshape(layer["out"], layer["k"])
        expected_values = np.take_along_axis(after_weights[index], selection[index], axis=1)
        selections_match = (
            selections_match
            and np.array_equal(indices, selection[index])
            and np.array_equal(values, expected_values)
        )
        visible.append(layer["visible_connections"])

    check(
        selections_match,
        "the frame sends the load-time selection with its post-update weights",
    )
    check(
        visible == [640, 320, 50] and sum(visible) == 1010,
        "one frame sends 1,010 real weights instead of all 109,184",
        str(visible),
    )
    check(
        all(layer["k"] == PARTIAL_TOP_K for layer in frame["weights"]["layers"]),
        "every neuron keeps the 5 connections chosen for it at load",
    )

    # A connection that was strong at load must keep being sent after other
    # weights overtake it: the selection is fixed, not re-ranked.
    chosen_row = selection[0][0]
    missing = int(np.setdiff1d(np.arange(INPUT_SIZE), chosen_row)[0])
    layers[0].weight.data[0, missing] = 10.0
    bumped = visible_weight_layers(module, selection)[0]
    bumped_indices = decode_uint16(bumped["indices"]).reshape(bumped["out"], bumped["k"])
    bumped_values = decode_float32(bumped["values"]).reshape(bumped["out"], bumped["k"])
    bumped_weights = layers[0].weight.detach().numpy()
    check(
        np.array_equal(bumped_indices, selection[0])
        and np.array_equal(
            bumped_values, np.take_along_axis(bumped_weights, selection[0], axis=1)
        )
        and missing not in set(bumped_indices[0].tolist()),
        "a weight that becomes the strongest is still not added to the fixed selection",
        f"input {missing}",
    )
    reranked = np.argsort(-np.abs(bumped_weights), axis=1, kind="stable")
    check(
        int(reranked[0][0]) == missing,
        "the re-ranking rule would have picked that connection, so the fixed set differs",
        str(reranked[0][:PARTIAL_TOP_K]),
    )

    restore_parameters(module, initial)
    check(
        weights_checksum(layers) == before_checksum,
        "restore_parameters() puts the initial weights back",
    )
    check(
        len(module.optimiser.state) == 0,
        "restore_parameters() also clears the Adam moments",
        str(len(module.optimiser.state)),
    )
    check(
        build_model_info(module)["weights_state"] == "untrained",
        "a restored model is reported as untrained again",
    )


def check_outbox() -> None:
    """Part 5: stale frames are replaced, control payloads keep their order."""
    from backend.trainer import Outbox

    async def session() -> tuple[list, dict | None]:
        outbox = Outbox(capacity=8)
        outbox.put({"type": "model_info"})
        for step in range(1, 6):
            outbox.put({"type": "training_step", "step": step})
        outbox.put({"type": "weights"})
        outbox.put({"type": "training_state"})

        delivered = [await outbox.get() for _ in range(4)]
        outbox.close()
        return delivered, await outbox.get()

    delivered, closed = asyncio.run(session())
    check(
        [payload["type"] for payload in delivered]
        == ["model_info", "training_step", "weights", "training_state"],
        "control payloads keep their order around a training frame",
        str([payload["type"] for payload in delivered]),
    )
    check(
        delivered[1]["step"] == 5,
        "an unread training frame is replaced by the newest one",
        str(delivered[1]["step"]),
    )
    check(closed is None, "a closed outbox ends the sender task")

    async def bounded() -> int:
        outbox = Outbox(capacity=4)
        for step in range(10):
            outbox.put({"type": "training_state", "step": step})
        delivered = 0
        while True:
            try:
                await asyncio.wait_for(outbox.get(), timeout=0.05)
            except asyncio.TimeoutError:
                return delivered
            delivered += 1

    check(asyncio.run(bounded()) == 4, "the outbox never grows past its capacity")


def check_session_is_idle() -> None:
    """Part 6a: a fresh session is idle until something starts it."""
    from concurrent.futures import ThreadPoolExecutor

    from backend.trainer import TrainingSession
    from backend.visualiser import select_partial_indices

    module = training_module()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-pytorch")
    try:
        session = TrainingSession(module, executor)
        status = session.status()

        check(
            session.state == "idle" and status["state"] == "idle" and status["step"] == 0,
            "creating a session does not start training",
            status["state"],
        )
        check(
            status["total_steps"] == TRAIN_SIZE * EPOCHS
            and status["epochs"] == EPOCHS
            and status["train_size"] == TRAIN_SIZE,
            "the session counts the real optimiser steps of one epoch",
            f"{status['total_steps']}/{status['epochs']}/{status['train_size']}",
        )
        check(
            status["speeds"] == ["slow", "normal", "fast"] and status["speed"] == "normal",
            "the three visualisation speeds are offered, normal by default",
            str(status["speeds"]),
        )
        check(
            status["frame_every"] == 1 and status["delay"] > 0,
            "normal speed draws every step but does not run flat out",
            f"{status['frame_every']} step(s), {status['delay']} s",
        )
        check(session.subscriber_count == 0, "a session starts with no subscribers")
        check(
            "not started" in status["message"],
            "the status explains that training has not started",
            status["message"],
        )

        shapes = [chosen.shape for chosen in session.selection]
        loaded = select_partial_indices(module)
        check(
            shapes == [(128, 5), (64, 5), (10, 5)]
            and sum(int(chosen.size) for chosen in session.selection) == 1010,
            "the session fixes the Partial view's 1,010 connections at load",
            str(shapes),
        )
        check(
            all(
                np.array_equal(chosen, expected)
                for chosen, expected in zip(session.selection, loaded)
            ),
            "the session's selection is the load-time top-5 |weight| of every neuron",
        )

        asyncio.run(session.reset())
        check(
            all(
                np.array_equal(chosen, expected)
                for chosen, expected in zip(session.selection, loaded)
            ),
            "a reset re-initialises exactly the same fixed selection",
        )
    finally:
        executor.shutdown(wait=True)


async def recv_message(socket, timeout: float = 30.0) -> dict:
    return json.loads(await asyncio.wait_for(socket.recv(), timeout=timeout))


async def next_of_type(socket, wanted: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no {wanted} message within {timeout} s")
        message = await recv_message(socket, remaining)
        if message.get("type") == wanted:
            return message


async def next_matching(
    socket, predicate, timeout: float = 30.0, describe: str = "message"
) -> dict:
    """The next message for which `predicate` is true."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no {describe} within {timeout} s")
        message = await recv_message(socket, remaining)
        if predicate(message):
            return message


async def next_state(socket, wanted: str, timeout: float = 30.0) -> dict:
    """The next `training_state` in the given state, ignoring other messages."""
    return await next_matching(
        socket,
        lambda message: message.get("type") == "training_state"
        and message.get("state") == wanted,
        timeout,
        f"{wanted} training_state",
    )


async def drain(socket, seconds: float) -> list[dict]:
    """Every message that arrives inside the window, then an empty tail."""
    collected = []
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return collected
        try:
            collected.append(await recv_message(socket, remaining))
        except asyncio.TimeoutError:
            return collected


def send(socket, payload: dict):
    return socket.send(json.dumps(payload))


async def wire_session() -> None:
    """Part 6b: the live server trains one real step at a time, on request."""
    import websockets

    async with websockets.connect(SOCKET_URL, open_timeout=30) as socket:
        info = await recv_message(socket)
        first = await recv_message(socket)
        check(
            info["type"] == "model_info" and first["type"] == "inference",
            "the socket still opens with model_info and one real sample",
            f"{info.get('type')}, {first.get('type')}",
        )
        check(info["batch_size"] == 1, "one optimiser step trains exactly one image")

        partial_layers = info.get("partial", {}).get("selection", [])
        fixed_indices = [
            decode_uint16(layer["indices"]).reshape(layer["out"], layer["k"])
            for layer in partial_layers
        ]
        check(
            info.get("partial", {}).get("count") == 1010
            and [layer["out"] for layer in partial_layers] == list(LAYER_SIZES)
            and [layer["k"] for layer in partial_layers] == [PARTIAL_TOP_K] * len(LAYER_SIZES),
            "model_info hands the page the fixed 1,010 connection indices",
            str([layer.get("visible_connections") for layer in partial_layers]),
        )

        run = info["training"]["run"]
        check(
            info["training"]["state"] == "idle" and info["training"]["step"] == 0,
            "the server starts idle, so nothing trains before the page asks",
            str(info["training"]),
        )

        idle = await drain(socket, 1.0)
        check(
            not [message for message in idle if message.get("type") == "training_step"],
            "no training frame is sent while idle",
            str([message.get("type") for message in idle]),
        )

        await send(socket, {"type": "request_weights"})
        snapshot = await next_of_type(socket, "weights")
        check(
            snapshot["step"] == 0 and snapshot["weights_checksum"] == info["weights_checksum"],
            "request_weights returns every current weight at step 0",
            str(snapshot.get("step")),
        )
        start_weights = [
            decode_float32(layer["weights"]).reshape(layer["out"], layer["in"]).astype(np.float64)
            for layer in snapshot["layers"]
        ]
        start_biases = [
            decode_float32(layer["biases"]).astype(np.float64) for layer in snapshot["layers"]
        ]
        check(
            all(
                np.array_equal(indices, expected)
                for indices, expected in zip(
                    fixed_indices,
                    [
                        np.argsort(-np.abs(weights), axis=1, kind="stable")[
                            :, :PARTIAL_TOP_K
                        ]
                        for weights in start_weights
                    ],
                )
            ),
            "those indices are the top-5 |weight| of the untrained model",
        )

        await send(socket, {"type": "set_speed", "speed": "slow"})
        speed_state = await next_of_type(socket, "training_state")
        check(
            speed_state["speed"] == "slow"
            and speed_state["frame_every"] == 1
            and speed_state["delay"] > 0,
            "slow speed draws every optimiser step",
            str(speed_state),
        )

        await send(socket, {"type": "start"})
        running = await next_state(socket, "running")
        check(
            running["run"] == run,
            "start is acknowledged with a running state",
            running["state"],
        )

        slow_frames = []
        while len(slow_frames) < 3:
            message = await recv_message(socket)
            if message.get("type") == "training_step":
                slow_frames.append(message)

        steps = [frame["step"] for frame in slow_frames]
        check(
            steps == [1, 2, 3],
            "each frame reports exactly one more completed optimiser step",
            str(steps),
        )
        check(
            all(frame["run"] == run for frame in slow_frames),
            "frames are tagged with the run they belong to",
        )
        check(
            all(frame["total_steps"] == TRAIN_SIZE for frame in slow_frames),
            "the frame's total steps match the real dataset",
        )
        check(
            slow_frames[0]["weights"]["checksum_before"] == info["weights_checksum"],
            "step 1 started from the weights the page was given",
        )
        check(
            all(
                later["weights"]["checksum_before"] == earlier["weights"]["checksum_after"]
                for earlier, later in zip(slow_frames, slow_frames[1:])
            ),
            "consecutive frames chain, so no optimiser step is skipped",
        )
        check(
            slow_frames[0]["weights"]["checksum_after"]
            != slow_frames[0]["weights"]["checksum_before"],
            "a frame's recorded checksums show a real parameter update",
        )
        check(
            same_indices(slow_frames, fixed_indices),
            "the same 1,010 connection indices are drawn in every frame",
        )
        first_values = [
            decode_float32(layer["values"]) for layer in slow_frames[0]["weights"]["layers"]
        ]
        last_values = [
            decode_float32(layer["values"]) for layer in slow_frames[-1]["weights"]["layers"]
        ]
        changed = [
            int(np.count_nonzero(a != b))
            for a, b in zip(first_values, last_values)
        ]
        check(
            all(count > 0 for count in changed)
            and sum(count for count in changed) > 100,
            "only the values at those fixed indices change as training runs",
            str(changed),
        )

        frame = slow_frames[0]
        pixels = decode_pixels(frame["sample"]["pixels"])
        a1, a2, logits, probabilities = reference_forward(start_weights, start_biases, pixels)
        check(
            np.allclose(decode_float32(frame["logits"]), logits, atol=1e-3),
            "the drawn logits are reproducible from the pre-update weights and pixels",
            f"max diff {float(np.abs(decode_float32(frame['logits']) - logits).max()):.2e}",
        )
        check(
            np.allclose(decode_float32(frame["probabilities"]), probabilities, atol=1e-5),
            "the drawn probabilities are those logits softmaxed",
        )
        check(
            abs(frame["loss"] + float(np.log(probabilities[frame["true_label"]]))) < 2e-3,
            "the drawn loss is the cross-entropy of the step that was trained",
            f"{frame['loss']:.6f}",
        )
        check(
            frame["sample"]["position"] == 1,
            "the frame names the sample at position 1 of the pass",
            str(frame["sample"].get("position")),
        )

        activations = [decode_float32(values) for values in frame["activations"]]
        check(
            [values.size for values in activations] == [INPUT_SIZE, *LAYER_SIZES],
            "the frame carries all 986 activations",
            str([values.size for values in activations]),
        )
        check(
            np.allclose(activations[1], a1, atol=1e-3)
            and np.allclose(activations[2], a2, atol=1e-3)
            and np.allclose(activations[3], probabilities, atol=1e-5),
            "the drawn activations are the real pre-update ReLU and softmax outputs",
        )

        partial = frame["weights"]["layers"]
        check(
            [layer["visible_connections"] for layer in partial] == [640, 320, 50],
            "the frame carries the Partial view's 1,010 connections",
            str([layer["visible_connections"] for layer in partial]),
        )
        values = np.concatenate([decode_float32(layer["values"]) for layer in partial])
        check(
            values.size == 1010 and bool(np.isfinite(values).all()),
            "the frame's weights are 1,010 real float32 values",
            str(values.size),
        )

        # Pause: the reported step must be a completed step, weights and all.
        await send(socket, {"type": "pause"})
        paused = await next_state(socket, "paused")
        stopped = await drain(socket, 1.5)
        late_frames = [message for message in stopped if message.get("type") == "training_step"]
        later_states = [
            message
            for message in stopped
            if message.get("type") == "training_state" and message.get("state") == "paused"
        ]
        paused = max([paused, *later_states], key=lambda message: message["step"])
        pause_step = paused["step"]
        check(pause_step >= 3, "pause keeps the steps that already ran", str(pause_step))
        check(
            all(message["step"] <= pause_step for message in late_frames),
            "no frame reports a step beyond the pause point",
            str([message["step"] for message in late_frames]),
        )

        await send(socket, {"type": "request_weights"})
        after = await next_of_type(socket, "weights")
        quiet = await drain(socket, 0.8)
        quiet_states = [
            message
            for message in quiet
            if message.get("type") == "training_state" and message.get("state") == "paused"
        ]
        if quiet_states:
            paused = max([paused, *quiet_states], key=lambda message: message["step"])
            pause_step = paused["step"]
        check(
            after["step"] == pause_step,
            "the paused weights belong to the pause step",
            f"{after['step']} != {pause_step}",
        )
        check(
            not [message for message in quiet if message.get("type") == "training_step"],
            "paused training sends no further frames",
            str([message.get("type") for message in quiet]),
        )

        newest = max(slow_frames + late_frames, key=lambda message: message["step"])
        current = [
            decode_float32(layer["weights"]).reshape(layer["out"], layer["in"]).astype(np.float64)
            for layer in after["layers"]
        ]
        drawn_matches = True
        for index, layer in enumerate(newest["weights"]["layers"]):
            indices = decode_uint16(layer["indices"]).reshape(layer["out"], layer["k"])
            drawn = decode_float32(layer["values"]).reshape(layer["out"], layer["k"])
            expected = np.take_along_axis(current[index], fixed_indices[index], axis=1)
            drawn_matches = (
                drawn_matches
                and np.array_equal(indices, fixed_indices[index])
                and np.array_equal(drawn, expected)
            )
        check(
            newest["step"] == after["step"] and drawn_matches,
            "the connections on screen are the real parameters after the last step",
            f"frame {newest['step']} vs snapshot {after['step']}",
        )


        # Reset: back to the weights the page was first given, as a new run.
        await send(socket, {"type": "reset"})
        reset_state = await next_state(socket, "idle")
        reset_weights = await next_of_type(socket, "weights")
        check(
            reset_state["step"] == 0,
            "reset reports zero progress again",
            f"{reset_state['state']} at step {reset_state['step']}",
        )
        check(
            reset_state["run"] == run + 1,
            "reset starts a new run so stale frames can be ignored",
            f"{reset_state['run']} != {run + 1}",
        )
        check(
            reset_weights["step"] == 0
            and reset_weights["weights_checksum"] == info["weights_checksum"],
            "reset restores the initial weights, not a fresh random model",
        )
        restored = [
            decode_float32(layer["weights"]).reshape(layer["out"], layer["in"])
            for layer in reset_weights["layers"]
        ]
        rederived = [
            np.argsort(-np.abs(weights), axis=1, kind="stable")[:, :PARTIAL_TOP_K]
            for weights in restored
        ]
        check(
            all(
                np.array_equal(again, first)
                for again, first in zip(rederived, fixed_indices)
            ),
            "after a reset the same 1,010 connections are selected again",
        )

        # Fast mode: every sample is still trained, fewer frames are drawn.
        await send(socket, {"type": "set_speed", "speed": "fast"})
        fast_state = await next_matching(
            socket,
            lambda message: message.get("type") == "training_state"
            and message.get("speed") == "fast",
            describe="fast training_state",
        )
        check(
            fast_state["frame_every"] > 1 and fast_state["delay"] == 0,
            "fast mode draws one frame per group of steps and runs flat out",
            str(fast_state["frame_every"]),
        )

        await send(socket, {"type": "start"})
        await next_state(socket, "running")
        collected = await drain(socket, 1.5)
        fast_frames = [message for message in collected if message.get("type") == "training_step"]
        check(bool(fast_frames), "fast mode still sends frames", str(len(fast_frames)))
        check(
            all(message["step"] % fast_state["frame_every"] == 0 for message in fast_frames),
            "fast-mode frames land on step boundaries",
            str([message["step"] for message in fast_frames][:6]),
        )
        highest = max((message["step"] for message in fast_frames), default=0)
        check(
            highest > len(fast_frames),
            "fast mode trains more steps than it draws",
            f"{highest} steps, {len(fast_frames)} frames",
        )
        check(
            all(message["run"] == reset_state["run"] for message in fast_frames),
            "fast-mode frames belong to the new run",
            str({message["run"] for message in fast_frames}),
        )
        check(
            all(
                later["weights"]["checksum_before"] != earlier["weights"]["checksum_after"]
                for earlier, later in zip(fast_frames, fast_frames[1:])
            ),
            "fast-mode frames jump over the steps that were not drawn",
        )
        check(
            same_indices(fast_frames, fixed_indices),
            "fast-mode frames keep drawing the same 1,010 connections",
        )

        await send(socket, {"type": "pause"})
        fast_paused = await next_state(socket, "paused")
        stopping = await drain(socket, 1.5)
        fast_states = [
            message
            for message in stopping
            if message.get("type") == "training_state" and message.get("state") == "paused"
        ]
        fast_paused = max([fast_paused, *fast_states], key=lambda message: message["step"])
        check(
            fast_paused["state"] == "paused",
            "fast training pauses as well",
            str(len(fast_states)),
        )
        fast_pause_step = fast_paused["step"]
        check(
            fast_pause_step >= highest,
            "training kept processing samples while frames were dropped",
            f"{fast_pause_step} vs {highest}",
        )
        trailing = await drain(socket, 0.8)
        check(
            not [message for message in trailing if message.get("type") == "training_step"],
            "a fast pause also stops the frames",
            str([message.get("type") for message in trailing]),
        )

        await send(socket, {"type": "reset"})
        final_state = await next_state(socket, "idle")
        final_weights = await next_of_type(socket, "weights")
        check(
            final_state["run"] == reset_state["run"] + 1 and final_state["step"] == 0,
            "a second reset starts another run from zero",
            str(final_state),
        )
        check(
            final_weights["weights_checksum"] == info["weights_checksum"],
            "a second reset restores the initial weights again",
        )


def check_frontend_sources() -> None:
    """Part 7: the page really asks for, decodes and draws these messages."""
    main_js = (REPO_ROOT / "frontend" / "main.js").read_text()
    network_js = (REPO_ROOT / "frontend" / "network.js").read_text()
    html = (REPO_ROOT / "frontend" / "index.html").read_text()

    check(
        '"training_step"' in main_js and "handleTrainingStep" in main_js,
        "main.js handles training frames",
    )
    check(
        "handleTrainingState" in main_js and "applyTrainingState" in main_js,
        "main.js follows the server's training state",
    )
    check(
        "decodeUint16" in main_js and "decodeVisibleWeights" in main_js,
        "main.js decodes the uint16 indices and float32 values",
    )
    check(
        "decodeSelection" in main_js
        and "setSelection" in network_js
        and "selection:" in main_js,
        "the page adopts the backend's fixed partial selection",
    )
    check(
        "MIN_DISPLAY_WEIGHT = -0.5" in network_js
        and "MAX_DISPLAY_WEIGHT = 0.5" in network_js
        and "Math.abs(weight) / MAX_DISPLAY_WEIGHT" in network_js,
        "network.js draws weights on one fixed scale from -0.5 to +0.5",
    )
    check(
        "layerMaxAbs" not in network_js and "layerAlphaScale" not in network_js,
        "no per-layer weight normalisation is left",
    )
    full_draw, partial_draw = (
        network_js.split(marker)[1].split(limit)[0]
        for marker, limit in (
            ("drawConnectionsFull(ctx)", "drawConnectionsPartial(ctx)"),
            ("drawConnectionsPartial(ctx)", "drawNeurons(ctx)"),
        )
    )
    check(
        all("connectionStyle(" in body for body in (full_draw, partial_draw)),
        "Full and Partial mode draw a weight through the same colour rule",
    )
    check(
        "activationMax = activations.map((values) => maxAbs(values))" in network_js
        and "maxAbs(weight)" not in network_js,
        "per-layer normalisation is left to activations only",
    )
    check(
        "strongest" not in main_js and "strongest" not in network_js,
        "the page no longer claims only the strongest connections are drawn",
    )
    check(
        "requestFullWeights" in main_js and '"request_weights"' in main_js,
        "main.js asks for the full weights instead of drawing partial ones full",
    )
    check(
        main_js.count("updateFrame(") >= 1 and "updateFrame(" in network_js,
        "a frame is drawn through network.js",
    )
    for command in ("start", "pause", "reset", "set_speed"):
        check(
            f'type: "{command}"' in main_js,
            f"main.js sends the {command} command",
        )
    check(
        "run !==" in main_js or "!== training.run" in main_js,
        "stale frames from an older run are ignored",
    )
    for element in (
        "btn-start",
        "btn-pause",
        "btn-reset",
        "speed-select",
        "controls-note",
        "progress-fill",
    ):
        check(
            f'"{element}"' in main_js and f'id="{element}"' in html,
            f"the {element} control exists",
        )
    check(
        ">Start<" in html and ">Reset<" in html and ">Pause<" in html,
        "the controls are no longer placeholders",
    )
    check(
        "placeholder" not in html.lower(),
        "the page no longer calls its controls placeholders",
    )


def luminance(colour: tuple[int, int, int]) -> float:
    """Relative luminance of an sRGB colour, to check depth is really depth."""
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in (channel / 255 for channel in colour)
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def check_light_theme_and_colour_scale() -> None:
    """Part 8: light theme, diverging weight scale and its legend."""
    network_js = (REPO_ROOT / "frontend" / "network.js").read_text()
    main_js = (REPO_ROOT / "frontend" / "main.js").read_text()
    html = (REPO_ROOT / "frontend" / "index.html").read_text()
    css = (REPO_ROOT / "frontend" / "style.css").read_text()

    colours = {
        name: tuple(int(value) for value in match)
        for name, *match in re.findall(
            r"const (\w+_COLOUR) = \{ r: (\d+), g: (\d+), b: (\d+) \};", network_js
        )
    }
    stops = re.search(r"const WEIGHT_COLOUR_STOPS = \[(.*?)\n\];", network_js, re.S)
    ordered = re.findall(r"\{ weight: ([^,]+), colour: (\w+) \}", stops.group(1) if stops else "")

    check(
        ordered
        == [
            ("MIN_DISPLAY_WEIGHT", "NEGATIVE_COLOUR"),
            ("MIN_DISPLAY_WEIGHT / 2", "NEGATIVE_WEAK_COLOUR"),
            ("0", "ZERO_COLOUR"),
            ("MAX_DISPLAY_WEIGHT / 2", "POSITIVE_WEAK_COLOUR"),
            ("MAX_DISPLAY_WEIGHT", "POSITIVE_COLOUR"),
        ],
        "the weight scale has five evenly spaced stops from -0.5 to +0.5",
        str(ordered),
    )

    expected = {
        "NEGATIVE_COLOUR": (185, 28, 28),
        "ZERO_COLOUR": (203, 213, 225),
        "POSITIVE_COLOUR": (109, 40, 217),
    }
    for name, wanted in expected.items():
        label = "#%02x%02x%02x" % wanted
        check(colours.get(name) == wanted, f"{name} is {label}", str(colours.get(name)))

    negative = colours.get("NEGATIVE_WEAK_COLOUR", (0, 0, 0))
    positive = colours.get("POSITIVE_WEAK_COLOUR", (0, 0, 0))
    check(
        negative[0] > negative[2] and min(negative) > 100,
        "weak negative weights are a light warm orange",
        str(negative),
    )
    check(
        positive[2] > positive[0] and min(positive) > 100,
        "weak positive weights are a light cool blue",
        str(positive),
    )
    check(
        luminance(expected["NEGATIVE_COLOUR"]) < luminance(negative)
        and luminance(expected["POSITIVE_COLOUR"]) < luminance(positive),
        "both ends of the scale are darker than the weak colours, so magnitude reads as depth",
    )

    check(
        "export function weightColour(weight)" in network_js
        and "Math.min(1, Math.abs(weight) / MAX_DISPLAY_WEIGHT)" in network_js,
        "every weight is clamped to that scale and coloured in exactly one place",
    )
    check(
        "const CONNECTION_OPACITY = 0.9" in network_js
        and "ALPHA_MIN" not in network_js
        and "ALPHA_GAMMA" not in network_js,
        "connections keep one constant opacity instead of a magnitude-driven alpha ramp",
    )

    check(
        re.search(r"--canvas-bg:\s*#f8fafc", css, re.I) is not None
        and "background: var(--canvas-bg)" in css,
        "the network canvas sits on the #f8fafc near-white background",
    )
    leftovers = [
        old
        for old in ("#080d16", "#060a12", "#0f1a2b", "#0b1524", "#1d2c45", "#e6edf7")
        if old in css.lower()
    ]
    check(not leftovers, "no dark-theme colours are left in style.css", str(leftovers))
    check(
        "--text: #1e293b" in css and "--muted: #64748b" in css,
        "body text and muted text are dark enough for the light background",
    )
    check(
        "rgba(30, 41, 59, 0.92)" in network_js and "rgba(71, 85, 105, 0.95)" in network_js,
        "captions and class labels are drawn in dark ink for the light canvas",
    )
    check(
        "grid-template-columns: minmax(0, 1fr) minmax(280px, 340px)" in css
        and "min-height: 520px" in css
        and "position: absolute;\n  inset: 0;" in css,
        "the page grid, panel sizes and canvas container keep their dimensions",
    )

    check(
        "weightColour(" in main_js
        and 'id="weight-scale-bar"' in html
        and "linear-gradient(90deg" in main_js,
        "main.js paints the legend from the renderer's own colour function",
    )
    legend = re.search(r'<ul class="legend">(.*?)</ul>', html, re.S)
    body = legend.group(1) if legend else ""
    positions = [body.find(f">{tick}<") for tick in ("-0.5", "-0.25", "0", "+0.25", "+0.5")]
    check(
        all(position >= 0 for position in positions)
        and positions == sorted(positions)
        and len(set(positions)) == 5,
        "the legend names -0.5, -0.25, 0, +0.25 and +0.5 in order",
        str(positions),
    )
    check(
        ".legend-scale" in css and ".scale-bar" in css and ".scale-ticks" in css,
        "the legend bar and its tick labels are styled",
    )
    check(
        "swatch-positive" not in html
        and "swatch-negative" not in html
        and "swatch-positive" not in css
        and "swatch-negative" not in css,
        "the old two-colour weight swatches are gone",
    )


def start_server() -> subprocess.Popen:
    """Launch the real server on a test port and wait until it answers."""
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


def main() -> int:
    check_local_maths()
    check_outbox()
    check_session_is_idle()
    check_frontend_sources()
    check_light_theme_and_colour_scale()

    process = start_server()
    try:
        asyncio.run(wire_session())
    finally:
        process.terminate()
        try:
            log = process.communicate(timeout=30)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            log = process.communicate()[0] or ""

    check("training started" in log, "the server logged the training loop starting")
    check(
        log.count("training reset") == 2,
        "the server logged both resets",
        str(log.count("training reset")),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1

    print("All Milestone 4 checks passed (real steps, honest frames, live controls).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

