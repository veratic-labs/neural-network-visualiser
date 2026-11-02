# Neural Network Training Visualiser

Real-time visualiser for the existing PyTorch MNIST MLP in `training/`
(`784 → 128 → 64 → 10`, batch size 1). See `plan.md` for the full design and
milestones.

**Current state: Milestone 3 — the page shows real inference data.** The FastAPI
backend sends the real, untrained PyTorch weights plus one real MNIST training
sample over a WebSocket: the 28 × 28 image, its true label, the forward-pass
logits and softmax probabilities, the predicted digit, the cross-entropy loss of
that sample, and the activations of every layer (drawn as teal neuron
brightness). "Next sample" asks for the next image. No optimiser step ever runs,
so the weights are unchanged and their colours never move.

Training with live weight updates, the Start/Pause/Resume controls and the
progress bar are still Milestone 4.

## Run the application

Everything is served by the backend, so one process is enough. From the
repository root, using the project virtualenv:

```bash
./venv/bin/python -m backend.server
```

Then open <http://localhost:8000>.

The backend loads `training/training.py` (its network only — the training loop
stays behind a `__main__` guard), sends the model to the browser on connect,
followed by one `inference` message per sample, and serves `frontend/` as static
files. A forward pass runs inside `torch.no_grad()` and never touches the
optimiser, so the weights cannot change.

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
./venv/bin/python tests/test_milestone3.py
```

Each script starts its own server (ports 8123 and 8124), verifies the messages
the browser receives, and shuts the server down again. Milestone 2 checks that
the transmitted weights are the model's real tensors. Milestone 3 checks that
the sample is a real MNIST image, that the logits, softmax, prediction, loss and
activations are reproducible from the transmitted weights and pixels, and that
`next_sample` returns the next image without changing a single weight.

## Files

| File | Purpose |
| --- | --- |
| `frontend/index.html` | Page layout: network view, sample / probability / metric panels, Next sample button, controls, progress bar |
| `frontend/style.css` | Styling and responsive layout |
| `frontend/network.js` | Canvas renderer for neurons, connections (full and partial modes) and activations |
| `frontend/main.js` | WebSocket client, panel rendering and UI wiring |
| `backend/server.py` | FastAPI app: `/ws` WebSocket (`model_info`, `inference`, `next_sample`) plus static frontend |
| `backend/visualiser.py` | Loads the existing PyTorch model, serialises it and runs one-sample forward passes |
| `tests/test_milestone2.py` | Model → payload → HTTP/WebSocket → frontend checks |
| `tests/test_milestone3.py` | Real sample, softmax, activations and `next_sample` checks |

`training/` is the existing PyTorch implementation. The only change made to it
(for Milestone 2) is a `if __name__ == "__main__":` guard around its training
loop so the network can be imported without starting training.
