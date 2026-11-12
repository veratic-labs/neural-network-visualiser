# Neural Network Training Visualiser

Real-time visualiser for the existing PyTorch MNIST MLP in `training/`
(`784 → 128 → 64 → 10`, batch size 1). See `plan.md` for the full design and
milestones.

**Current state: Cleanup 3 — the replay is published as a static site, and the
Live Training app is gone.** A recording written by the recorder is replayed by
`frontend/replay.html`: no server-side application, no socket, just the files in
`recordings/`. Play, Pause, Previous, Next, a speed selector and a timeline move
between the recorded checkpoints, and the same canvas renderer draws them. A
checkpoint is shown exactly as it was recorded — there is no interpolation or
smoothing between snapshots.

Full mode draws the 109,184 initial weights from `model.bin`; Partial mode draws
the 1,010 connections each checkpoint actually recorded, scattered back onto the
same fixed selection every frame shares. The recorded sample, probabilities, loss
and activations are shown for every checkpoint that has one.

## Run it

The project has two halves, and neither needs a Python server:

* **Record** a training run once, offline, into a directory of binary files —
  see [Record a training run](#record-a-training-run-offline).
* **Replay** that recording in the browser — from `frontend/replay.html` against
  `recordings/`, or from the published copy in `docs/`.

The replay page is static: it needs a plain file server and nothing else, which
is why the copy in `docs/` can be hosted on GitHub Pages. The sections below
cover each half, the tests that verify them and the recording format.

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

`frontend/replay.html` plays a recording back with a static file server and
nothing else. It only needs a server that serves the repository root, so that the
page can read `recordings/` through `../recordings/…`:

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

Chunks are read from `data/chunks/` as playback needs them: the checkpoint after the
one on screen is always fetched, and at Fast the chunk after the current one is
fetched as soon as playback enters it. A chunk stays on screen for 2.5 s at Fast but
25 s at Normal and 500 s at Slow, while a cold 2.8 MB chunk takes about 2 s over a
12 Mbit/s link, so only Fast (and a seek) can outrun its own downloads. Slow and
Normal therefore still fetch a chunk only when they reach it.

It must be served over HTTP rather than opened as a `file://` URL, because
`replay.js` is an ES module and `fetch()` cannot read local files.

```bash
./venv/bin/python tests/test_milestone5b.py
```

This checks the page, the replay script, the shared modules, the colour scale, the
recording's layout and decoded checkpoints, and that `training/` is untouched. It
needs the recorder output, not a training run.

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
| `docs/index.html` | `frontend/replay.html` with only its `data-recording` path changed to `./data/` |
| `docs/style.css`, `docs/network.js`, `docs/panels.js` | The shared frontend files, copied byte for byte |
| `docs/replay.js`, `docs/recording.js` | The replay wiring and the recording reader, copied byte for byte |
| `docs/data/manifest.json`, `docs/data/model.bin`, `docs/data/model-biases.bin`, `docs/data/partial-indices.bin` | The recording's manifest, initial weights, biases and Partial selection |
| `docs/data/chunks/chunk-0000.bin` … `chunk-0012.bin` | All 6,001 records, 500 per chunk (about 34 MiB in total) |

`.nojekyll` stops GitHub Pages from running Jekyll over the directory. Every URL
in the page and the scripts is relative, so the site works at a project-site path
such as `https://<user>.github.io/mlp-visualisation/`; the recording is read from
`./data/` because the `<body>` carries `data-recording="./data/"` and
`data-run-name="mnist-epoch1"`. Nothing outside `docs/` is fetched, so
`training/`, `recordings/` and the virtualenv never need publishing.

`--recording PATH` and `--output PATH` select another recording or another output
directory; the build refuses to write into the repository root, `frontend/`,
`training/`, `recorder/` or a recording.

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

## Measure playback smoothness

`scripts/measure_fast_playback.py` is a diagnostic, not a test: it serves `docs/`
over HTTP, throttles the browser's network and plays the recording twice in one
browser context (cold, then warm from the HTTP cache). For each run it reports
wall time, checkpoints actually drawn, every stall with its cause (network, main
thread or mixed), every chunk request with its lead time, animation-frame gaps
and long tasks.

```bash
./venv/bin/python scripts/measure_fast_playback.py
./venv/bin/python scripts/measure_fast_playback.py --mbps 12 --latency 60 --speed fast
./venv/bin/python scripts/measure_fast_playback.py --mbps 0 --speed normal --json run.json
```

`--mbps 0` disables throttling, `--speed` picks slow, normal or fast, `--stall-ms`
sets what counts as a stall, `--json PATH` writes the raw measurements and
`--site PATH` serves another directory instead of `docs/`. It needs Playwright
with Chromium (`./venv/bin/python -m pip install playwright`, then
`./venv/bin/python -m playwright install chromium`) and prints that instruction
when the module is missing. It reads `docs/` and the recording only; the sole
output file is the path given to `--json`.

## Files

| File | Purpose |
| --- | --- |
| `frontend/style.css` | Styling and responsive layout |
| `frontend/network.js` | Canvas renderer for neurons, connections (full and partial modes) and activations |
| `frontend/panels.js` | Panel rendering shared by the replay page's modules |
| `frontend/recording.js` | Reads a recording from `recordings/`: manifest, initial model, Partial selection, checkpoints |
| `frontend/replay.html` | Replay page: sample, metrics, timeline, transport controls, progress bar |
| `frontend/replay.js` | Replay wiring: playback loop, seeking, mode switch, shared renderers |
| `recorder/record_training.py` | Standalone recorder: trains one epoch and writes the binary recording plus manifest |
| `recorder/model_access.py` | Loads `training/training.py` without training, and encodes weights, activations and the fixed Partial selection |
| `tests/test_milestone5a.py` | Recorder checks: record layout, sizes, manifest, fixed Partial indices, recorded values vs. PyTorch |
| `tests/test_milestone5b.py` | Replay checks: page and ids, offline-only wiring, shared modules, colour scale, decoded checkpoints |
| `scripts/build_pages.py` | Validates a recording and writes the publishable GitHub Pages site to `docs/` |
| `scripts/measure_fast_playback.py` | Diagnostic: throttled browser playback of `docs/`, reporting stalls, chunk lead times and long tasks |
| `docs/` | The published site: replay page, its modules, `.nojekyll` and `data/` (recording, chunks) |
| `tests/test_milestone5c.py` | Static-site checks: reproducible build, published layout and sizes, HTTP from a nested path, browser replay |

`training/` is the existing PyTorch implementation. The only change made to it is
a `if __name__ == "__main__":` guard around its training loop so the network can
be imported without starting training.
