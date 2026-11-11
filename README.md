# Neural Network Training Visualiser

Real-time visualiser for the existing PyTorch MNIST MLP in `training/`
(`784 → 128 → 64 → 10`, batch size 1). See `plan.md` for the full design and
milestones.

**Current state: Milestone 5C — the replay is published as a static site.** The
live page (`index.html`) still trains and animates over its WebSocket, exactly as
in Milestone 4. A second page, `frontend/replay.html`, plays back a recording
written by the recorder: no backend, no socket, just the files in
`recordings/`. Play, Pause, Previous, Next, a speed selector and a timeline move
between the recorded checkpoints, and the same canvas renderer draws them. A
checkpoint is shown exactly as it was recorded — there is no interpolation or
smoothing between snapshots.

Full mode draws the 109,184 initial weights from `model.bin`; Partial mode draws
the 1,010 connections each checkpoint actually recorded, scattered back onto the
same fixed selection the live page uses. The recorded sample, probabilities, loss
and activations are shown for every checkpoint that has one.

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

## Replay a recording in the browser (offline)

`frontend/replay.html` plays a recording back without the backend. It only needs
a static file server that serves the repository root, so that the page can read
`recordings/` through `../recordings/…`:

```bash
./venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

Then open <http://localhost:8000/frontend/replay.html>. The page reads
`../recordings/mnist-epoch1/` — the path comes from the `data-recording`
attribute on the `<body>`, so a different recording only needs that attribute
changed (or a copy of the page).

Controls:

| Control | Effect |
| --- | --- |
| Play / Pause | Advances through the recorded checkpoints and stops when the last one is shown |
| Previous / Next | Moves exactly one checkpoint; a manual move pauses playback |
| Speed | Slow (1/s), Normal (20/s) or Fast (≈200/s, the whole recording in about 30 s); changing it while playing takes effect immediately |
| Timeline | Scrubs over checkpoint indices; the readout shows the recorded optimiser step |
| Full / Partial | Full draws `model.bin`, Partial draws the recorded 1,010 weights |

Step 0 is the initial model and has no sample, so the sample panel says so; every
later checkpoint shows its recorded image, prediction, probabilities, loss and
activations. Nothing is interpolated: each frame is one complete record. At the
Fast rate the page may skip visual frames when decoding cannot keep up, but the
checkpoint on screen is always the one the elapsed-time clock points at, never a
blend of two records.

Like the live page, it must be served over HTTP rather than opened as a `file://`
URL, because `replay.js` is an ES module and `fetch()` cannot read local files.

```bash
./venv/bin/python tests/test_milestone5b.py
```

This checks the page, the replay script, the shared modules, the colour scale, the
recording's layout and decoded checkpoints, and that `training/` and `backend/`
are untouched. It needs the recorder output, not a training run.

## Publish the replay as a static site (GitHub Pages)

Milestone 5C turns the replay into a self-contained site in `docs/`, which GitHub
Pages can serve without a backend:

```bash
./venv/bin/python scripts/build_pages.py
```

The build validates the recording (manifest, record layout, file sizes, step
list, Partial indices and the weight checksum), then writes:

| Path | Contents |
| --- | --- |
| `docs/index.html` | `frontend/replay.html` with `data-recording="./data/"` and a note replacing the live-page link |
| `docs/style.css`, `docs/network.js`, `docs/panels.js` | The shared frontend files, copied byte for byte |
| `docs/replay.js`, `docs/recording.js` | The replay wiring and the recording reader, copied byte for byte |
| `docs/data/manifest.json`, `docs/data/model.bin`, `docs/data/model-biases.bin`, `docs/data/partial-indices.bin` | The recording's manifest, initial weights, biases and Partial selection |
| `docs/data/chunks/chunk-0000.bin` … `chunk-0012.bin` | All 6,001 records, 500 per chunk (about 34 MiB in total) |

`.nojekyll` stops GitHub Pages from running Jekyll over the directory. Every URL
in the page and the scripts is relative, so the site works at a project-site path
such as `https://<user>.github.io/mlp-visualisation/`; the recording is read from
`./data/` because the `<body>` carries `data-recording="./data/"` and
`data-run-name="mnist-epoch1"`. Nothing outside `docs/` is fetched, so
`training/`, `backend/`, `recordings/` and the virtualenv never need publishing.

`--recording PATH` and `--output PATH` select another recording or another output
directory; the build refuses to write into the repository root, `frontend/`,
`training/`, `backend/`, `recorder/` or a recording.

To inspect the site locally, serve the directory itself (it also works under a
nested path, as on GitHub Pages):

```bash
./venv/bin/python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000/>. To publish, commit `docs/` and select
"Deploy from a branch" → `main` / `/docs` in the repository's Pages settings.

```bash
./venv/bin/python tests/test_milestone5c.py
```

The test rebuilds the site, checks that a second build is byte-identical, that
`docs/` contains only the page and the recording, that every copied file matches
its source, and that all 6,001 steps decode from `docs/data`. It then serves the
page from a nested path over HTTP and, when Playwright's Chromium is available,
drives it: recorded values against the binary data, seeking and Previous/Next,
Full vs. Partial mode, all three speeds, and a missing chunk. Without Playwright
the browser checks print `SKIP` and the rest still run.

## Files

| File | Purpose |
| --- | --- |
| `frontend/index.html` | Page layout: network view, sample / probability / metric panels, Next sample button, controls, progress bar |
| `frontend/style.css` | Styling and responsive layout |
| `frontend/network.js` | Canvas renderer for neurons, connections (full and partial modes) and activations |
| `frontend/panels.js` | Panel rendering shared by the live page and the replay page |
| `frontend/main.js` | WebSocket client, panel rendering and UI wiring |
| `frontend/recording.js` | Reads a recording from `recordings/`: manifest, initial model, Partial selection, checkpoints |
| `frontend/replay.html` | Offline replay page: sample, metrics, timeline, transport controls, progress bar |
| `frontend/replay.js` | Replay wiring: playback loop, seeking, mode switch, shared renderers |
| `backend/server.py` | FastAPI app: `/ws` WebSocket (`model_info`, `inference`, `next_sample`) plus static frontend |
| `backend/visualiser.py` | Loads the existing PyTorch model, serialises it and runs one-sample forward passes |
| `tests/test_milestone2.py` | Model → payload → HTTP/WebSocket → frontend checks |
| `tests/test_milestone3.py` | Real sample, softmax, activations and `next_sample` checks |
| `tests/test_milestone4.py` | Live training, Start/Pause/Resume and progress bar checks |
| `recorder/record_training.py` | Standalone recorder: trains one epoch and writes the binary recording plus manifest |
| `tests/test_milestone5a.py` | Recorder checks: record layout, sizes, manifest, fixed Partial indices, recorded values vs. PyTorch |
| `tests/test_milestone5b.py` | Replay checks: page and ids, offline-only wiring, shared modules, colour scale, decoded checkpoints |
| `scripts/build_pages.py` | Validates a recording and writes the publishable GitHub Pages site to `docs/` |
| `docs/` | The published site: replay page, its modules, `.nojekyll` and `data/` (recording, chunks) |
| `tests/test_milestone5c.py` | Static-site checks: reproducible build, published layout and sizes, HTTP from a nested path, browser replay |

`training/` is the existing PyTorch implementation. The only change made to it
(for Milestone 2) is a `if __name__ == "__main__":` guard around its training
loop so the network can be imported without starting training.
