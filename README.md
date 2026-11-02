# Neural Network Training Visualiser

Real-time visualiser for the existing PyTorch MNIST MLP in `training/`
(`784 → 128 → 64 → 10`, batch size 1). See `plan.md` for the full design and
milestones.

**Current state: Milestone 2 — the frontend draws the real, untrained PyTorch
weights sent by the FastAPI backend over a WebSocket.** No training runs yet, so
the weights never change; the sample image, prediction, loss and progress bar
are still placeholders labelled "mock".

## Run the application

Everything is served by the backend, so one process is enough. From the
repository root, using the project virtualenv:

```bash
./venv/bin/python -m backend.server
```

Then open <http://localhost:8000>.

The backend loads `training/training.py` (its network only — the training loop
stays behind a `__main__` guard), sends the model to the browser on connect and
serves `frontend/` as static files.

### WSL2 + Windows browser

* Uvicorn binds to `127.0.0.1`, and WSL2 forwards that to Windows, so
  <http://localhost:8000> works directly in Edge/Chrome.
* If localhost forwarding is unavailable, bind to all interfaces and use the WSL
  address (`hostname -I`):

  ```bash
  VISUALISER_HOST=0.0.0.0 VISUALISER_PORT=8000 ./venv/bin/python -m backend.server
  ```

  then open `http://<wsl-ip>:8000`. The binding is configurable through the
  `VISUALISER_HOST` / `VISUALISER_PORT` environment variables.

The page must be served over HTTP, not opened as a `file://` URL, because
`main.js` is loaded as an ES module and the WebSocket URL is derived from
`window.location`.

### Check that it works

```bash
./venv/bin/python tests/test_milestone2.py
```

The script starts its own server on port 8123, compares the transmitted weights
with the PyTorch tensors, and shuts the server down again.

## Files

| File | Purpose |
| --- | --- |
| `frontend/index.html` | Page layout: network view, sample / probability / metric panels, controls, progress bar |
| `frontend/style.css` | Styling and responsive layout |
| `frontend/network.js` | Canvas renderer for neurons and connections (full and partial modes) |
| `frontend/main.js` | WebSocket client, panel rendering and UI wiring |
| `frontend/mock.js` | Placeholder sample image, probabilities, loss and progress (no weights) |
| `backend/server.py` | FastAPI app: `/ws` WebSocket plus static frontend |
| `backend/visualiser.py` | Loads the existing PyTorch model and serialises it |
| `tests/test_milestone2.py` | End-to-end check: model → payload → HTTP/WebSocket → frontend |

`training/` is the existing PyTorch implementation. The only change made for
Milestone 2 is a `if __name__ == "__main__":` guard around its training loop so
the network can be imported without starting training.
