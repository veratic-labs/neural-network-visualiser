"""FastAPI backend for the visualiser: serves the frontend, the model and samples.

Start it from the repository root:

    ./venv/bin/python -m backend.server

then open http://localhost:8000.

Milestone 3 scope: on every WebSocket connection the server sends the existing
PyTorch model once as a `model_info` message, followed by an `inference` message
with one real MNIST sample, its softmax probabilities and the activations of
every layer. The client asks for another sample with `{"type": "next_sample"}`.
Only forward passes run, so no weight ever changes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .visualiser import build_inference, build_model_info, load_training_module, summarise

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Bind to loopback: WSL2 forwards localhost to Windows, so the browser on the
# host can reach http://localhost:8000 without exposing the server.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("visualiser")

# The model is loaded once per process, before the first client connects.
TRAINING_MODULE = load_training_module()
MODEL_INFO = build_model_info(TRAINING_MODULE)

app = FastAPI(title="Neural Network Training Visualiser")


@app.websocket("/ws")
async def model_socket(websocket: WebSocket) -> None:
    """Send the real model plus one real sample, then answer `next_sample`."""
    await websocket.accept()
    logger.info("client connected: %s", websocket.client)
    sample_index = 0

    try:
        await websocket.send_json(MODEL_INFO)
        logger.info(
            "sent model_info to %s (%d layers, %d connections)",
            websocket.client,
            len(MODEL_INFO["layers"]),
            MODEL_INFO["totals"]["connections"],
        )

        # The page opens with a real sample rather than a placeholder.
        await send_sample(websocket, sample_index)

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            command = json.loads(message.get("text") or "{}")
            if command.get("type") == "next_sample":
                sample_index += 1
                await send_sample(websocket, sample_index)
            else:
                logger.warning("ignoring client message: %s", command.get("type"))
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("client disconnected: %s", websocket.client)


async def send_sample(websocket: WebSocket, index: int) -> None:
    """Run one forward pass through the existing model and send the result."""
    payload = build_inference(TRAINING_MODULE, index)
    await websocket.send_json(payload)
    logger.info(
        "sent inference for sample %d/%d (true %d, predicted %d, loss %.4f, weights %s)",
        payload["sample"]["index"] + 1,
        payload["sample"]["total"],
        payload["true_label"],
        payload["predicted_label"],
        payload["loss"],
        payload["weights_checksum"][:16],
    )


# Mounted last so the WebSocket route above keeps precedence. html=True serves
# index.html for "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

logger.info("loaded %s", summarise(MODEL_INFO))


def main() -> None:
    host = os.environ.get("VISUALISER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("VISUALISER_PORT", str(DEFAULT_PORT)))
    logger.info("serving %s at http://%s:%d", FRONTEND_DIR, host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
