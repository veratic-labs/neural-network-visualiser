"""FastAPI backend for the visualiser: serves the frontend, the model and training.

Start it from the repository root:

    ./venv/bin/python -m backend.server

then open http://localhost:8000.

Every connection starts with `model_info` (the current weights plus the training
status) and one real `inference` sample. The client then drives the run with:

    {"type": "start"}            one real optimiser step after another
    {"type": "pause"}            stop at the next step boundary, keep the weights
    {"type": "reset"}            back to the initial weights, progress 0
    {"type": "set_speed", "speed": "slow" | "normal" | "fast"}
    {"type": "request_weights"}  every current weight, for the Full view
    {"type": "next_sample"}      one forward pass, no optimiser step

Training never starts on its own: nothing happens until a client sends `start`.
The steps themselves come from ``train_step()`` in training/training.py and run
on a single worker thread, so the event loop stays responsive and no two PyTorch
computations ever overlap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .trainer import Outbox, TrainingSession
from .visualiser import (
    build_inference,
    build_model_info,
    build_weight_snapshot,
    load_training_module,
    summarise,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Bind to loopback: WSL2 forwards localhost to Windows, so the browser on the
# host can reach http://localhost:8000 without exposing the server.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("visualiser")

# The model is loaded once per process, before the first client connects.
TRAINING_MODULE = load_training_module()

# Exactly one worker: a training step and a forward pass can never run at the
# same time, and nothing PyTorch does can block the event loop.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch")
TRAINING = TrainingSession(TRAINING_MODULE, EXECUTOR)


async def in_executor(function: Callable[..., Any], *args: Any) -> Any:
    """Run PyTorch work on the single worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, function, *args)


def current_model_info() -> dict:
    """`model_info` for a client connecting right now (weights + training state)."""
    return build_model_info(TRAINING_MODULE, TRAINING.training_info(), TRAINING.selection)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("loaded %s", summarise(current_model_info()))
    yield
    await TRAINING.shutdown()
    EXECUTOR.shutdown(wait=True)


app = FastAPI(title="Neural Network Training Visualiser", lifespan=lifespan)


@app.websocket("/ws")
async def model_socket(websocket: WebSocket) -> None:
    """Send the real model, then serve commands and training frames."""
    await websocket.accept()
    logger.info("client connected: %s", websocket.client)

    outbox = Outbox()
    sample_index = 0

    # `model_info` first: it carries the current weights and whether training is
    # idle, running or paused, so a reloaded page shows the right controls and
    # progress without asking for anything. The inference sample is queued before
    # subscribing to training, so the first two messages keep that documented
    # order even when the page is reloaded mid-run; a coalesced frame that
    # happens to be missed meanwhile is replaced by the next one anyway.
    outbox.put(current_model_info())
    outbox.put(await in_executor(build_inference, TRAINING_MODULE, sample_index))
    TRAINING.subscribe(outbox)

    sender = asyncio.create_task(send_frames(websocket, outbox), name="ws-sender")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            command = json.loads(message.get("text") or "{}")
            kind = command.get("type")

            if kind == "next_sample":
                sample_index += 1
                outbox.put(await in_executor(build_inference, TRAINING_MODULE, sample_index))
            elif kind == "start":
                await TRAINING.start()
            elif kind == "pause":
                await TRAINING.pause()
            elif kind == "reset":
                await TRAINING.reset()
                await publish_weights()
            elif kind == "set_speed":
                try:
                    await TRAINING.set_speed(str(command.get("speed")))
                except ValueError as error:
                    outbox.put({"type": "error", "message": str(error)})
            elif kind == "request_weights":
                outbox.put(await snapshot_weights())
            else:
                logger.warning("ignoring client message: %s", kind)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        try:
            await sender
        except (asyncio.CancelledError, RuntimeError):
            pass
        TRAINING.unsubscribe(outbox)
        logger.info("client disconnected: %s", websocket.client)

        # Nobody is watching any more: stop at the next step boundary instead of
        # training unseen to the end of the epoch.
        if TRAINING.subscriber_count == 0:
            await TRAINING.pause_for_disconnect()


async def send_frames(websocket: WebSocket, outbox: Outbox) -> None:
    """Send this client everything its outbox holds, one payload at a time."""
    try:
        while True:
            payload = await outbox.get()
            if payload is None:
                return
            await websocket.send_json(payload)
            # Training frames are far too frequent to log; the payloads that
            # carry the model and its samples are logged after they are sent.
            if payload.get("type") in ("model_info", "inference"):
                logger.info("sent %s", payload["type"])
    except (WebSocketDisconnect, RuntimeError):
        # The socket closed under us; the connection handler does the cleanup.
        return


async def snapshot_weights() -> dict:
    """Every current weight and bias, for the Full view."""
    return await in_executor(build_weight_snapshot, TRAINING_MODULE, TRAINING.step)


async def publish_weights() -> None:
    """Broadcast a full weight snapshot (used after a reset)."""
    await TRAINING.broadcast(await snapshot_weights())


# Mounted last so the WebSocket route above keeps precedence. html=True serves
# index.html for "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main() -> None:
    host = os.environ.get("VISUALISER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("VISUALISER_PORT", str(DEFAULT_PORT)))
    logger.info("serving %s at http://%s:%d", FRONTEND_DIR, host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
