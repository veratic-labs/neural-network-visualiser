"""FastAPI backend for the visualiser: serves the frontend and the model.

Start it from the repository root:

    ./venv/bin/python -m backend.server

then open http://localhost:8000.

Milestone 2 scope: on every WebSocket connection the server sends the existing
PyTorch model once as a `model_info` message and then keeps the connection
open. Nothing is trained, so no weights change and no further messages follow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .visualiser import build_model_info, load_training_module, summarise

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
    """Send the real model to a new client, then keep the socket open."""
    await websocket.accept()
    logger.info("client connected: %s", websocket.client)

    try:
        await websocket.send_json(MODEL_INFO)
        logger.info(
            "sent model_info to %s (%d layers, %d connections)",
            websocket.client,
            len(MODEL_INFO["layers"]),
            MODEL_INFO["totals"]["connections"],
        )

        # Milestone 2 has no client-to-server commands; reading until the
        # disconnect keeps the channel open for the training milestones.
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("client disconnected: %s", websocket.client)


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
