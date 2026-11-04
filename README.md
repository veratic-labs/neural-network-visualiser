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

## Record a training run (offline)

Milestone 5A adds a standalone recorder that trains the same network once and
writes the run to disk, so it can be replayed without retraining.

```bash
./venv/bin/python -m recorder.record_training --output recordings/mnist-epoch1
```

That runs one full epoch — 60,000 optimiser steps at batch size 1 — and records a
snapshot every 10 steps, from Step 0 (the initial model) to Step 60,000. On this
machine it takes about 50 seconds; budget a few minutes on a slower CPU. A short
run is enough to inspect the format:

```bash
./venv/bin/python -m recorder.record_training --output recordings/verify-100 --steps 100
```

`--steps N` records only the first N steps, `--seed N` calls `torch.manual_seed()`
before the model is created (the training script sets no seed, so the initial
weights are otherwise random), `--chunk-size N` sets the checkpoints per chunk
file (default 500) and `--log-every N` the progress interval. The output
directory must not exist yet or must be empty, and must not be inside
`training/`, `frontend/` or `docs/`.

A recording directory contains:

| File | Contents |
| --- | --- |
| `model.bin` | The 109,184 initial float32 weights, written once |
| `model-biases.bin` | The 202 initial float32 biases |
| `partial-indices.bin` | The 1,010 fixed Partial-mode connection indices (`uint16`) |
| `chunks/chunk-0000.bin` | Fixed 5,688-byte records, one per checkpoint |
| `manifest.json` | Architecture, recorded steps, layout, indices, colour scale, capture convention |

`manifest.json` is written last, so its presence is what marks a recording as
complete. Everything is little-endian and `recordings/` is ignored by Git.

Each record holds the MNIST sample, its true label, the predicted digit, the
softmax probabilities, the loss and the layer activations of the forward pass the
step trained on — all **pre-update** — plus the 1,010 Partial weights **after**
`optimiser.step()`. Step 0 has no sample: it is the initial model.

### Check the recorder

```bash
./venv/bin/python tests/test_milestone5a.py
```

It records a 100-step run, decodes every record and compares them with the values
PyTorch actually produced.

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
| `recorder/record_training.py` | Standalone recorder: trains one epoch and writes the binary recording plus manifest |
| `tests/test_milestone5a.py` | Recorder checks: record layout, sizes, manifest, fixed Partial indices, recorded values vs. PyTorch |

`training/` is the existing PyTorch implementation. The only change made to it
(for Milestone 2) is a `if __name__ == "__main__":` guard around its training
loop so the network can be imported without starting training.
