# Neural Network Training Visualiser — Implementation Plan

> **Document status.** Milestones 1–5C and Cleanup 1–3 are complete and approved. Cleanup 4 (final verification) has been performed; its results await user approval. **The final supported application is the offline Training Recorder plus the Static Replay Visualiser published to GitHub Pages.** Live Training, the FastAPI backend and the WebSocket interface have been removed. The Python runtime dependencies are `torch`, `torchvision` and `numpy` in `requirements.txt`; Playwright remains an optional development dependency for the browser tests and playback diagnostic.

## 1. Project goal

Build a web-based **neural network training visualiser** for the existing PyTorch MNIST multilayer perceptron (MLP). The project is intended for learning and observation: it should show how an individual handwritten-digit image travels through the network, what the network predicts, and how the weights change after training.

The existing PyTorch model, training data and training algorithm are the source of truth. Do not rewrite the model in JavaScript.

The project was delivered in two stages:

1. **Live stage — Milestones 1–4 (completed historical milestone):** trained in Python while streaming training state over a WebSocket to an HTML/CSS/JavaScript frontend. This mode is retired; its code was removed in Cleanup 3.
2. **Offline stage — Milestones 5A–5C (completed):** record one complete training run into compact binary files, then replay that recording in the frontend as a static website that needs no Python, FastAPI, PyTorch or WebSocket. This is the final supported application.

Recording once and replaying later makes the visualisation reproducible and freely seekable, and lets it be published as a free static site. The final project therefore consists of exactly three capabilities:

- **PyTorch training** — the existing model, dataset and training algorithm in `training/`, unchanged.
- **Offline training recorder** — `recorder/`, which trains once and writes a binary recording.
- **Static replay visualiser** — `frontend/` and the published `docs/` site, which replay a recording with no Python running.

Everything else in the repository exists only to support those three. GitHub Pages remains the public deployment target.

## 2. Project status and authorised scope

### 2.1 Milestone status

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | Static visualiser frontend with mock data | Completed — historical |
| 2 | Real PyTorch model streamed to the frontend | Completed — historical; removed in Cleanup 3 |
| 3 | Real MNIST inference — image, softmax, activations | Completed — historical; the replay keeps this display |
| 4 | Live training controls, real optimiser steps, fixed 1,010 Partial connections | Completed — historical; removed in Cleanup 3 |
| 5A | Training recorder | Completed |
| 5B | Offline replay visualiser | Completed |
| 5C | GitHub Pages deployment | Completed and deployed |
| Cleanup 1 | Project plan update (this document) | Completed and approved |
| Cleanup 2 | Shared helper migration | Completed and approved |
| Cleanup 3 | Legacy removal | Completed and approved |
| Cleanup 4 | Final verification | Verification complete — awaiting user approval |

- **The final supported application is the offline system:** PyTorch training in `training/`, the offline Training Recorder in `recorder/`, and the Static Replay Visualiser in `frontend/` published from `docs/` to GitHub Pages.
- **The requirement to preserve Live Training is retired.** Earlier revisions of this plan required `backend/server.py`, the WebSocket interface and the Live Training frontend to keep working. They no longer do; Cleanup 3 removed them.
- Milestones 1–5C are historical. They record what was built and approved; they are not to be re-implemented or re-scoped. The 5C deployment remains published on GitHub Pages.
- Cleanup 2, Cleanup 3 and Cleanup 4 are separate milestones, each with its own approval gate. Cleanup 2 and Cleanup 3 are approved; Cleanup 4 is complete when the user explicitly approves it.
- Nothing outside the currently authorised milestone may be implemented. During the Cleanup phase that also means no behaviour change and no removal beyond what the current cleanup milestone authorises.

### 2.2 Final architecture (implemented structure)

```text
training/               # Existing PyTorch model and MNIST loader
recorder/               # Training recorder and shared model-access utilities
recordings/             # Local recordings, ignored by Git
frontend/               # Static replay frontend only
docs/                   # GitHub Pages deployment files
tests/                  # Recorder, replay, and deployment tests
scripts/                # Static build and performance diagnostics
```

- **`training/`** — unchanged source of truth for the model, the MNIST loader, the loss, the optimiser and `train_step()`. The cleanup must not modify the training algorithm.
- **`recorder/`** — the standalone recorder plus the shared model-access helpers it needs (`recorder/model_access.py`). It runs without FastAPI or Uvicorn.
- **`recordings/`** — locally generated recordings, ignored by Git (already in `.gitignore`). The existing `recordings/mnist-epoch1/` recording is the reference recording.
- **`frontend/`** — the static replay page and its modules only: `replay.html`, `style.css`, `network.js`, `panels.js`, `recording.js` and `replay.js`. The Live Training page and its socket client were removed in Cleanup 3.
- **`docs/`** — the published static site: `index.html`, `style.css`, the replay modules, `.nojekyll` and `docs/data/`. It contains no Python.
- **`tests/`** — recorder, replay and deployment tests. Tests that exclusively exercised the retired Live Training application were retired in Cleanup 3.
- **`scripts/`** — `build_pages.py` (static build and validation) and `measure_fast_playback.py` (performance diagnostics).

Not part of the final architecture and removed in Cleanup 3: `backend/server.py`, `backend/trainer.py`, the rest of the `backend/` package, the WebSocket protocol, `frontend/index.html` and `frontend/main.js`.

### 2.3 Preservation rules (binding on every cleanup milestone)

The cleanup phase changes packaging, not behaviour. None of the following may change:

- the PyTorch training algorithm, the model architecture (`784 → 128 → 64 → 10`), the loss (`nn.CrossEntropyLoss`), the optimiser (`torch.optim.Adam`, learning rate `0.001`) or `batch_size = 1`;
- the existing 6,001 checkpoints in `recordings/mnist-epoch1/` — steps 0, 10, 20, …, 60,000 — or any of their recorded bytes;
- the binary recording format: record stride 5,688 bytes, field offsets and dtypes, manifest format name and version, and chunk naming (Sections 7.4–7.5);
- the fixed 1,010 Partial connections and the rule that selects them once from the initial weights;
- the ±0.5 diverging weight colour scale and its five colour stops;
- playback speed behaviour, page layout, controls, keyboard behaviour and chunk prefetching;
- the current GitHub Pages website content and URLs.

A cleanup milestone that would change any of these must stop and ask for approval instead of proceeding.

## 3. Confirmed design decisions

### 3.1 Original real-time decisions (Milestones 1–4) — historical, retired

- **Development environment:** WSL2 Ubuntu for Python, PyTorch and agent workflows; Windows for VS Code and browser. Serve the web UI locally (for example, `http://localhost:8000`).
- **Model:** use the existing MNIST MLP (currently expected to be `784 → 128 → 64 → 10`); verify the actual definitions in the supplied code rather than hard-coding assumptions.
- **Dataset:** use the existing MNIST data pipeline.
- **Batch size:** **1**. One training step processes one image and performs one optimiser update.
- **Visualisation:** show all neurons and all connections in an initial network overview **before training starts**. When training begins, automatically switch to a reduced set of connections while keeping the network structure visible. The viewer should be able to inspect individual neurons and their connections.
- **Weight encoding:** distinguish positive and negative weights with two contrasting colours; use opacity/intensity or line width to indicate magnitude. Include a legend. Avoid treating the sign of a weight as the sign of its activation.
- **Training information:** current 28 × 28 image, true label, predicted digit, all 10 class scores/probabilities, current loss, epoch, sample/step number, and network activations/weights.
- **Playback controls:** Start, Pause, Resume, and Stop/Reset as appropriate; speed selection including a **slow observation mode** and a **fast mode**.
- **Progress:** place a progress bar **below the main visualisation**, with completed steps / total steps and current epoch. Progress indicates actual training steps, not animation time.
- **Prioritise correctness and clarity over excessive animations.** Work in small verifiable increments.

These decisions are historical. The conventions that survive into the final replay — batch size 1, the all-connection overview, two-colour weight encoding on the fixed ±0.5 scale, the light theme with legends, and progress measured in real optimiser steps — are protected by the preservation rules in Section 2.3. Nothing in this section requires the Live Training application to be kept.

### 3.2 Offline architecture decisions (Milestones 5A–5C) — completed

- **Record once, replay anywhere.** A standalone Python recorder trains the existing model and writes visualisation data to disk. The browser never trains, never needs PyTorch and never needs a server.
- **Static delivery.** The published replay consists only of HTML, CSS, JavaScript and recorded data files, hosted on GitHub Pages.
- **One full epoch.** The recording covers all 60,000 MNIST training samples at `batch_size = 1` — 60,000 optimiser steps — matching the existing configuration in `training/training.py`.
- **Snapshot cadence.** Checkpoints are recorded every 10 completed optimiser steps, plus the initial state at Step 0 and the final state: 6,001 checkpoints for a complete run.
- **Fixed Partial selection.** The 1,010 Partial connections are chosen once, before training begins, from the initial weights: the top 5 incoming |weight| per receiving neuron. They are never reselected, in Python or in JavaScript.
- **Full weights recorded once.** The initial 109,184 connections are saved at Step 0. Full-mode colours stay static for the whole playback; updated full-network weights are not recorded.
- **Binary float32 data.** Numerical data is stored as little-endian float32 arrays (images as uint8) and described by a JSON manifest. Checkpoints are never serialised as per-frame JSON number blobs.
- **Real timeline.** The timeline represents actual recorded optimiser steps. There is no interpolation between checkpoints and no artificial animation frame numbering.
- **Unchanged visuals.** Reuse the existing layout, Canvas renderer, light theme, legends and the fixed ±0.5 diverging weight colour scale.
- **Publish from `docs/`.** Milestone 5C generates a self-contained `docs/` directory served by GitHub Pages from the `main` branch. The local Python environment, dataset and recorder stay out of the deployed site.

## 4. UX specification

Sections 4.1 and 4.3 describe the retired Live Training mode and are kept as a historical record. Section 4.2 describes the network display states, which the replay retains as its Full and Partial modes, and Section 4.4 describes the final replay UX.

### 4.1 Main page layout (live mode, Milestones 1–4) — historical, retired

1. **Header:** project title, model architecture, current status (Ready / Training / Paused / Finished / Error).
2. **Network view (central focus):** neurons arranged by layer with weighted connections. The initial view shows every connection. Once training starts, render only a defined subset of connections by default (see Section 4.2 for the display states and Section 7.8 for the connection modes).
3. **Current sample panel:** current MNIST image enlarged with crisp pixels, ground-truth label, predicted digit, and whether the prediction is correct. Make clear whether the displayed prediction was produced **before** or **after** the optimiser update (use one consistent convention; prefer forward-pass prediction before update).
4. **Output panel:** 10 bars labelled 0–9, showing `softmax(logits)` probabilities; optionally expose logits separately so they are not mistaken for probabilities.
5. **Training metrics:** current loss, epoch, step, and optionally a small historical loss chart.
6. **Control row and bottom progress bar:** Start / Pause / Resume, speed selection, and progress `completed / total` with a percentage. Keep the progress bar visible at all times.

### 4.2 Network display states

In the final replay these two states are the viewer-selectable **Full** and **Partial** modes (Section 7.8); selecting Start is replaced by selecting a mode, and no connection selection is recomputed. The original live requirements follow.

**State A — Ready / overview (before training):**

- Show every neuron and every weight connection for the model, preferably rendered with Canvas or WebGL rather than creating a separate SVG/DOM element for each of ~109,000 lines.
- At initialisation, obtain the real initial weights from PyTorch if possible; do not imply placeholder weights are real.
- For a `784 → 128 → 64 → 10` architecture, the full graph contains 986 neurons and 109,184 weight connections. Expect significant overlap: this overview illustrates scale and layer structure, not an individually legible set of lines.
- If drawing all lines creates performance problems, render the static overview once, cache it, and redraw only when needed; maintain the requirement to show all connections at least initially.

**State B — Training / focused view:**

- Automatically switch to partial connection rendering when Start is pressed.
- Begin with a deterministic subset of lines (for example, the strongest positive/negative weights per neuron or a fixed sampled subset). Document the actual selection rule; never imply that the displayed connections are the whole network.
- Keep the **full weight tensors** in Python even when rendering only a subset.
- Colour and intensity should update as real weights change. Show neuron activation using a separate visual encoding (such as fill brightness) with its own legend.
- Clicking or hovering a neuron should reveal its layer/index, activation, bias (where applicable), and relevant incoming/outgoing weights, with an option to focus on its connections.
- A **Show all connections** toggle during training is optional after the minimum viable version; it must not be required for the initial milestone.

### 4.3 Speed and timing (live mode, Milestone 4) — historical, retired

- Training progress should advance **one sample per optimiser step** (`batch_size=1`).
- **Slow/Observe:** after each step, wait for the frontend to acknowledge or permit the next step, or otherwise use a controlled delay. This ensures humans can observe each image/weight update without missing frames. The UI must remain responsive.
- **Normal:** moderate visualisation cadence.
- **Fast:** remove deliberate waits and reduce UI update frequency (e.g., display every Nth training step), **without changing the underlying optimiser steps or silently skipping training samples**. The progress bar must continue to represent actual completed steps.
- Choose practical defaults after testing. Do not make animation speed alter the learning rate or the mathematics of training.
- Pause should take effect at a safe boundary between training steps (not halfway through `loss.backward()` or `optimiser.step()`).

### 4.4 Replay UX (Milestone 5B) — final replay behaviour, completed

1. **Header:** same title and architecture summary, with a status badge that says **Replay** and shows the recorded step. It must never say "Training" or imply live training.
2. **Network view:** the same Canvas renderer and the same two modes. Full mode shows all 109,184 initial connections and stays static. Partial mode shows the fixed 1,010 recorded connections and recolours them at each selected checkpoint.
3. **Current sample panel:** the MNIST image of the selected recorded step, its true label, the predicted digit and whether it is correct. One caption states the capture convention: the prediction, probabilities, activations and loss come from the forward pass **before** that step's optimiser update, while the displayed Partial weights are the state **after** it. At Step 0 the panel states that it is the initial model and that no optimiser step has completed.
4. **Output panel:** the 10 softmax bars of the selected checkpoint.
5. **Training metrics:** the recorded loss of the selected checkpoint, its epoch, its real step, and the total recorded steps.
6. **Timeline:** a slider over the recorded checkpoints, labelled with real step numbers (0 … 60,000), not animation frames. Dragging the slider seeks directly to that checkpoint.
7. **Playback controls and progress:** Play, Pause, Previous, Next and a playback speed selector, with the existing bottom progress bar showing the selected step / total steps. Previous and Next move exactly one checkpoint (±10 steps).
8. **Data state:** a small indicator while recorded data is being fetched, and a clear error message if the manifest or a data file cannot be loaded. Playback must never quietly show a different checkpoint from the one selected.

## 5. Live backend design (Milestones 2–4) — historical, retired

This section is retained as a historical record of how the live mode worked. It is **not** a requirement to preserve `backend/`, FastAPI, Uvicorn or the WebSocket interface. The recorder's few dependencies on helpers in `backend/visualiser.py` were migrated to `recorder/model_access.py` in Cleanup 2; Cleanup 3 removed the backend entirely.

The project already contains working PyTorch training code, data-loading utilities, and the MNIST dataset in the `training/` directory.

The backend reused the existing implementation rather than creating a separate training system. This design was implemented for Milestones 2–4 and has since been retired.

Suggested project structure as originally planned:

```text
neural-network-visualiser/
├── AGENTS.md
├── plan.md
├── training/
│   ├── train.py          # Existing PyTorch training code
│   ├── load_data.py      # Existing data-loading implementation
│   └── data/             # Existing MNIST dataset
├── backend/
│   ├── server.py         # Local HTTP server + WebSocket endpoint
│   └── visualiser.py     # Extracts/serialises training snapshots
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── main.js           # Client state, controls and socket
│   └── network.js        # Neural-network drawing
└── README.md
```

The filenames inside `training/` are illustrative. The final files are
`training/training.py` (model, loss, optimiser and `train_step()`) and
`training/load_data.py`. The live files (`backend/server.py`,
`backend/trainer.py`, `backend/visualiser.py`, the Live Training frontend files)
and `tests/test_milestone2.py` … `tests/test_milestone4.py` were removed in
Cleanup 3; Section 2.2 lists the final layout.

Use FastAPI/WebSocket if compatible with the existing project; keeping the backend simple is more important than introducing a framework unnecessarily.

**Existing code requirements:**

- Inspect the entire `training/` directory before making training-related changes.
- Reuse the existing PyTorch model, training loop, and data-loading implementation.
- Do not create another neural network or duplicate the training logic.
- Do not move, rename, or delete existing files without explicit approval.
- Do not download or duplicate the MNIST dataset unnecessarily.
- Preserve the existing model architecture, loss function, optimiser, and training behaviour.
- Keep `batch_size = 1` for step-by-step visualisation.
- Make only the minimum changes necessary to expose training information and support external controls.
- Do not modify the training implementation during Milestone 1.

### 5.1 Training step sequence

For each `(image, label)` at batch size 1:

1. Acquire the current image/label using the existing data-loading implementation. Normalise and reshape exactly as the existing model expects.
2. Compute the forward pass using the existing model. Capture logits and intermediate layer activations.
3. Compute loss, backward pass, and `optimiser.step()` using the **existing** training logic.
4. Capture the **updated** layer weights and biases at the chosen observation cadence.
5. Publish a snapshot containing the sample image, true label, pre-update prediction/logits/probabilities, loss, activations, updated weight information, and progress metadata. Label the before/after timing consistently.
6. Respect Pause/Stop and speed control **at a safe step boundary**, then proceed to the next image.

For the existing MLP, intermediate activations can be captured through explicit variables or PyTorch forward hooks. Prefer the least intrusive approach compatible with the current implementation.

Detach and move tensors to CPU for serialisation. Do not transmit tensors requiring gradient tracking.

### 5.2 Communication

Use a WebSocket to send training events and receive control events.

**Server → browser**

- `model_info`: architecture, layer sizes, total expected steps, optional initial weights.
- `training_state`: current training status and errors.
- `step`: sample index, epoch, step count, image pixels, label, logits/probabilities, predicted label, loss, activation data, and selected connection weights or a weight snapshot reference.
- `progress`: completed/total training steps, especially when fast display skips individual visual frames.
- `finished`: final metrics and training status.

**Browser → server**

- `start`
- `pause`
- `resume`
- `stop`
- `set_speed`
- `focus_neuron` (optional)

Use JSON for metadata and small initial messages.

Avoid sending all 109,184 float32 weights as JSON on every step, since this introduces unnecessary serialisation, transfer, and rendering overhead.

For the first version, transmit only visible/subselected connection weights at each visual update, or throttle full snapshots and use binary transfer if necessary.

Keep full weights in the existing PyTorch model. Make connection identifiers stable so line colours update correctly.

As implemented, the server message types are `model_info`, `inference`, `training_state`, `training_step`, `weights` and `error`; weights, probabilities, activations and pixels travel as base64 of raw little-endian float32/uint8 bytes rather than as JSON number arrays. The implemented client commands are `start`, `pause`, `reset`, `set_speed`, `request_weights` and `next_sample`, so `resume`, `stop` and `focus_neuron` are covered by `start` after `pause`, by `reset`, and by the deferred neuron-inspection backlog in Section 8.

### 5.3 Correctness and stability

- Do not block the FastAPI event loop with the training loop. Run training in a worker/thread/process as appropriate and safely relay state to the WebSocket.
- Handle disconnects gracefully. Prefer pause-on-disconnect for the initial observation-oriented design.
- Ensure pause/resume and progress counters reflect actual completed optimiser steps.
- Never modify the optimiser or learning rate when only changing the visualisation speed.
- Keep all training computations within the existing PyTorch implementation.
- Ensure the visualisation backend does not accidentally trigger a second training process.
- Preserve compatibility with the existing dataset paths and loading functions.
- Any necessary modification to files inside `training/` must be minimal, explained, and limited to the currently authorised milestone.

### 5.4 Relationship to the offline pipeline — transition to the final architecture

- The live backend is **not** part of the published replay site; Milestone 5C never deploys `backend/`.
- The recorder (5A) reused model-loading and serialisation helpers from `backend/visualiser.py` rather than duplicating the model or the selection rule. Cleanup 2 moved the helpers it needs to `recorder/model_access.py`, so the recorder no longer imports the `backend` package.
- **Superseded requirement:** earlier revisions required `./venv/bin/python -m backend.server` to keep serving the live mode and `tests/test_milestone2.py`, `tests/test_milestone3.py` and `tests/test_milestone4.py` to keep passing. That requirement is retired. Those tests were retired in Cleanup 3.

## 6. Frontend design (the final replay frontend retains these requirements; the live wiring is retired)

- Start with vanilla HTML + CSS + JavaScript; no React dependency required.
- Use **Canvas** for network lines and neurons to avoid a huge number of DOM/SVG elements in the full overview. Use ordinary HTML for panels, controls and labels.
- **Historical (retired):** rendering and socket event handling were separated into `network.js` and `main.js`. The final frontend has no socket; `replay.js` supplies recorded data to the unchanged renderer.
- Draw all connections **once** in Full mode, then show the fixed subset in Partial mode.
- Distinguish weight colour/intensity from neuron activation colour/intensity; show clear legends.
- On each selected checkpoint, update image, predictions, loss, highlights, visible weight colours and progress. Prevent overlapping renders and stale state while seeking.
- During Playback, coalesce/redraw at an affordable cadence without skipping checkpoints.

### 6.1 Replay frontend (Milestone 5B) — final frontend design, completed

- **One renderer.** `frontend/network.js` provides `setModel()`, `setSelection()`, `setMode()`, `setWeights()`, `setActivations()` and `updateFrame()`. It is the only network renderer and is reused unchanged by the replay page.
- **Shared panel rendering.** The DOM/panel updates for image, labels, verdict, output bars, metrics and step/epoch live in `frontend/panels.js`. No second renderer and no second copy of the panel logic exist.
- **No socket in replay.** The published page must not open a WebSocket, must not call FastAPI endpoints, and must not fall back to mock data. Load failures are reported, never hidden.
- **Relative URLs only.** Every asset and data reference is relative (`./style.css`, `./replay.js`, `./data/manifest.json`, manifest entries relative to the data directory), so the site works under a GitHub Pages project path.
- **No interpolation and no local animation state.** The page renders exactly the checkpoint that is selected. Playback only advances the checkpoint index.
- **Explicit provenance.** The page states it is a replay of a recorded run, shows the recorded step, and labels pre-update versus post-update values as described in Section 7.3.

## 7. Offline recording and replay architecture (Milestones 5A–5C) — completed

### 7.1 Pipeline

```text
training/training.py            model, MNIST data, loss, Adam, train_step()
        │ imported unchanged
        ▼
recorder/record_training.py     standalone recorder (5A)
        │ writes binary checkpoints + manifest
        ▼
recordings/<run-name>/          manifest.json, *.bin, chunks/*.bin
        │ copied by the 5C build script
        ▼
docs/                           static site (frontend + data)
        │ plain HTTP, relative URLs
        ▼
browser                         network.js renderer + replay data source (5B)
```

Training happens exactly once per recording. Viewing the recorded run never invokes Python, never trains, and never changes a weight.

### 7.2 Recorder responsibilities (5A, completed)

- A standalone program in the project virtualenv, for example `./venv/bin/python -m recorder.record_training --output recordings/full-run`. The argument name and exact path are settled at the start of 5A; the recorder must be runnable without starting the web server.
- Imports the existing implementation through the shared model-access helpers and uses that module's `layer1`, `layer2`, `layer3`, `loss_fn`, `optimiser`, `train_loader`, `train_data`, `batch_size` and `train_step()`. No second model, no copied forward/backward pass, no changed hyperparameters. Cleanup 2 moved those helpers from `backend/visualiser.py` to `recorder/model_access.py`; their behaviour and signatures did not change.
- Drives the loop itself with `train_loader` + `train_step()` rather than calling `train()`, because `train()` writes `training/weights/mnist_mlp.pth` and offers no observation hooks. This keeps every artefact out of `training/`.
- Calls `select_partial_indices(module)` **once**, before the first optimiser step, on the initial weights. That exact selection (640 + 320 + 50 = 1,010 indices) is written once and used for every checkpoint. It is never recomputed during or after training.
- Records Step 0 before any optimiser step: the initial full weights, the initial values of the 1,010 selected connections, and no sample data.
- After each completed optimiser step whose number is divisible by 10, appends one checkpoint. After the loop it guarantees a final checkpoint at the last completed step, so a complete run ends at Step 60,000.
- Uses `training/training.py`'s existing configuration: `batch_size = 1`, one pass over the shuffled `train_loader`, 60,000 steps, `nn.CrossEntropyLoss`, `torch.optim.Adam(lr=0.001)`. It does not add epochs, batching or learning-rate changes.
- Prints progress (`step / total`, current loss) at a sensible interval, then the final step count, elapsed time, checkpoint count and output size, so a multi-minute CPU run is observable.
- Writes only complete recordings. A crashed or interrupted run is invalid and must be regenerated; resumability is not part of 5A.
- Reads tensors under `torch.no_grad()` / `.detach().cpu()` and writes raw little-endian bytes. It must not keep computation graphs alive.
- Does not write into `training/`, `frontend/` or `docs/`, and imports nothing from the retired backend.

### 7.3 Capture convention: pre-update versus post-update

Each checkpoint describes the completed optimiser step `k`:

| Recorded value | Moment it was read |
| --- | --- |
| MNIST image, true label | the sample that step `k` trained on (fixed for the step) |
| Predicted digit, softmax probabilities, loss | forward pass of step `k`, **before** `optimiser.step()` — **pre-update** |
| Activations A1, A2 and the output layer | the same forward pass — **pre-update** |
| The 1,010 Partial weights | after `optimiser.step()` of step `k` — **post-update** |

- This matches the convention of the approved, now retired Milestone 4 live frame in `backend/visualiser.py` (`build_training_step`), so the recorded bytes describe a step exactly as the live frame did.
- Field names, the manifest, the page caption and the tests must all state the convention. Do not label a pre-update prediction as the result of the displayed weights.
- Step 0 is different: it is the initial model before any optimiser step. Its record carries the initial Partial weights and is flagged as having no sample, prediction, activations or loss.

### 7.4 Recorded files and binary layout

```text
<output>/
├── manifest.json          # architecture, recorded steps, indices, layout, colour scale, capture notes
├── model.bin              # initial full weights, 109,184 × float32, written once
├── model-biases.bin       # initial 202 biases, once (supports neuron/bias details; optional but cheap)
├── partial-indices.bin    # the fixed 1,010 connection indices, uint16, written once
└── chunks/
    ├── chunk-0000.bin     # fixed-stride checkpoint records
    ├── chunk-0001.bin
    └── …
```

**`model.bin`** — layer 1 `[128][784]`, layer 2 `[64][128]`, layer 3 `[10][64]`, row-major `[out][in]`, concatenated in that order, total 100,352 + 8,192 + 640 = 109,184 float32 values. This is the initial state and the only full-weight file.

**`partial-indices.bin`** — per layer `[out][5]` uint16 input indices, layer-major, in the order `select_partial_indices()` returns them: 128 × 5 = 640, 64 × 5 = 320, 10 × 5 = 50, total 1,010 indices. The same file applies to every checkpoint.

**Checkpoint record** — fixed stride for every checkpoint, so any record is read by arithmetic, never by parsing JSON:

| Offset | Bytes | Field | Type |
| --- | --- | --- | --- |
| 0 | 4 | completed step | uint32 |
| 4 | 4 | sample position in the epoch (1-based) | uint32 |
| 8 | 1 | true label (0–9) | uint8 |
| 9 | 1 | predicted digit (argmax softmax) | uint8 |
| 10 | 1 | flags: bit 0 = has sample | uint8 |
| 11 | 1 | reserved (0) | uint8 |
| 12 | 784 | image, row-major 28 × 28 | uint8 (0–255) |
| 796 | 40 | softmax probabilities, classes 0–9 | float32 × 10 |
| 836 | 4 | cross-entropy loss | float32 |
| 840 | 808 | activations: A1 (128), A2 (64), output softmax (10) | float32 × 202 |
| 1,648 | 4,040 | Partial weights, layer-major over the 1,010 indices | float32 × 1,010 |
| **5,688** | | **record size (bytes)** | |

Notes:

- All floats are little-endian float32; all arrays are row-major. Offsets keep float32 fields 4-byte aligned.
- The 784 input activations are the image pixels divided by 255, so they are not stored twice; the frontend derives them from the uint8 image exactly as `main.js` does today.
- The output activations are the same 10 softmax values as the probabilities field, stored again because the renderer expects a four-layer activation list. The duplication costs 40 bytes per record.
- The Partial weights are stored in the fixed manifest order, not in a per-checkpoint order, so record `k` always describes the same 1,010 connections.
- Step 0 uses the `has sample = 0` flag; its image, probabilities, loss and activation bytes are zero.

### 7.5 Manifest contents

`manifest.json` is the only JSON the replay needs, and it must fully describe its own data:

- **`format`** — format name and version, byte order, record size in bytes, per-field offsets and dtypes, and the chunk naming scheme. A replay that does not recognise the version must refuse to run.
- **`model`** — layer sizes `784 → 128 → 64 → 10`, activation (`ReLU` on both hidden layers, logits into `CrossEntropyLoss`), loss (`nn.CrossEntropyLoss`), optimiser (`torch.optim.Adam`), learning rate (`0.001`), batch size (`1`), epochs (`1`), training samples (`60,000`), total optimiser steps (`60,000`), and the sha256 checksum of the initial weights so a recording can be matched to a model state.
- **`training`** — recording date, Python and PyTorch versions, wall-clock duration, the snapshot cadence (10), the recorded first and last step (0 and 60,000), and the exact ordered list of recorded steps (or the cadence rule plus the explicit final step).
- **`checkpoints`** — checkpoint count (6,001) and, per chunk file, the file name, first/last step, record count and byte size.
- **`partial`** — the selection rule ("top 5 incoming |weight| per receiving neuron of the initial model, fixed for the whole recording"), per-layer `in`/`out`/`k`, the total visible connection count (1,010), and the file holding the indices.
- **`full`** — the initial-weight file, its connection count (109,184) and layout, the bias file, and an explicit statement that later full-network weights are not recorded.
- **`activations`** — the per-layer sizes and order (A1 128, A2 64, output 10), that they are pre-update values, and that input activations come from the image.
- **`images`** — 28 × 28, uint8, 0–255, row-major.
- **`capture`** — the pre-update / post-update statement from Section 7.3, in machine-checkable field names and in prose for the UI.
- **`colour`** — the fixed scale bounds (`-0.5` to `+0.5`) and the five stops (`#b91c1c`, `#fdba74`, `#cbd5e1`, `#93c5fd`, `#6d28d9`), matching `frontend/network.js`, plus a note that no per-layer or per-checkpoint normalisation is applied.
- **`notes`** — the exact command used to produce the recording and any known limitation (for example "initial weights are random because the training code sets no seed; each recording stores its own").

If a seed is deliberately set to make recordings reproducible, it must be recorded in the manifest. Setting one is optional and must not change the training mathematics.

### 7.6 Size budget

| Item | Size |
| --- | --- |
| Initial full weights (`model.bin`) | 436,736 bytes (≈ 0.42 MiB) |
| Initial biases (`model-biases.bin`) | 808 bytes |
| Fixed connection indices | 2,020 bytes |
| One checkpoint record | 5,688 bytes |
| All 6,001 checkpoints | ≈ 34.1 MB |
| Complete recording | ≈ 34.6 MB |

- Chunk files are sized in whole records (for example 500 checkpoints ≈ 2.8 MB each, 13 files for a complete run). The chosen chunk size is recorded in the manifest, and the loader never guesses a file's contents.
- 34.6 MB is acceptable for a GitHub Pages project, but it must not be downloaded up front. The replay fetches the manifest, the initial model and indices once, then only the chunk it needs, caching a small window of recent chunks.
- If transfer size ever becomes a problem, the acceptable options are chunk-level on-demand loading first, then (only with explicit approval) pre-compressed chunk files or reduced numeric precision. Precision may not be silently changed.

### 7.7 Replay loading and playback (5B)

- **Startup:** fetch `manifest.json`, validate the format version, then load `model.bin`, `model-biases.bin` and `partial-indices.bin` once. Only after that does playback become available; the page shows a loading state until then.
- **Checkpoint access:** map a checkpoint to its chunk and byte offset from the manifest, fetch the chunk if it is not cached, then decode that single record in place. Seeking, Next, Previous and Play all use the same lookup, so they always show the same data for the same step.
- **Caching:** keep the most recently used chunks in memory (for example the current chunk plus its neighbours) and prefetch the next chunk during playback. Drop old chunks when they are no longer needed; never keep all 34 MB decoded.
- **Relative URLs:** every request is relative to the page (`./data/...`), never root-absolute, so the site works under `https://<user>.github.io/<repo>/`.
- **Timeline:** the slider's values are checkpoint indices; its labels and readouts show the real recorded step numbers (0, 10, 20, …, 60,000). Dragging it seeks; keyboard arrow support is expected because it is a range input.
- **Playback:** Play advances one checkpoint per tick at the chosen speed; Pause stops at the current checkpoint; Previous and Next move exactly one checkpoint. Speed changes only the dwell time per checkpoint. It never skips checkpoints, renumbers them, changes step labels or interpolates values.
- **Rendering:** reuse the existing `NetworkView` API. Each checkpoint update sets the partial weights, activations, image, probabilities, loss and step readouts through the renderer's documented functions.
- **Truthfulness:** the page shows "replay" and the recorded step, never a simulated "live" state. A missing or corrupt chunk produces an error message, not a stale or invented frame.

### 7.8 Full and Partial connection modes

**Full mode**

- Displays all 109,184 connections.
- Uses the initial weights recorded at Step 0 (`model.bin`).
- Keeps the full network's connections and colours static throughout playback.
- Does not record, request or load updated full-network weights for later steps.
- The UI states that Full mode shows the initial network, so a static diagram is not mistaken for a stalled update.

**Partial mode**

- Displays exactly the same fixed 1,010 connections for the whole playback, in the manifest's index order.
- Reads their recorded weights from the selected checkpoint and updates their colours when the checkpoint changes.
- Never reselects connections from current weight magnitude, in Python or in JavaScript. A connection that started in the top 5 stays on screen even after another weight overtakes it.

### 7.9 Colour scale and theme

- One fixed diverging scale for every layer, both modes and every checkpoint: `MIN_DISPLAY_WEIGHT = -0.5` to `MAX_DISPLAY_WEIGHT = 0.5` in `frontend/network.js`, dark red for strongly negative weights, light orange for weakly negative, grey at zero, light blue for weakly positive and dark purple for strongly positive.
- The scale is never normalised per layer, per checkpoint or per recording; magnitudes saturate at ±0.5 exactly as in the approved live version.
- Neuron activation keeps its separate teal encoding, and the light theme and legends are unchanged.
- Recorded values are real weights. No gain, clipping or smoothing is applied to the data; only the drawing saturates.
- The manifest repeats the bounds and stops so a viewer can confirm the data is drawn on the documented scale.

## 8. Implementation milestones for the coding agent

Implement **one milestone at a time**. Do not redesign the user's existing PyTorch model or introduce unrelated functionality.

Milestones 1–4 and 5A–5C below are historical and completed. The active roadmap is the Project Cleanup phase (Cleanup 1–4), documented after Milestone 5C in this section. Each cleanup milestone needs explicit approval before work starts, and the next must not begin until the previous is approved.

### Milestone 1 — Inspect existing project and build static UI (completed and approved)

- Read `train.py`/model/data pipeline to confirm actual architecture and dataset format.
- Build the page layout, neuron/connection renderer, legend, sample/output panels and bottom progress bar with **mock data clearly marked as mock**.
- Verify that the Ready-state renderer can draw **all** network connections without freezing the browser.

**Acceptance:** page loads locally, full overview is drawn before Start, and training view can switch to partial lines.

### Milestone 2 — Connect real model state (completed and approved)

- Start the Python server and establish the WebSocket.
- Send real architecture/initial weights from the loaded PyTorch model.
- Show the real initial network in Ready state; no training controls required yet.

**Acceptance:** displayed layer counts/weights come from the model rather than hard-coded mock values.

### Milestone 3 — Single-sample real-time training (completed and approved)

- Set `batch_size=1` and preserve the existing optimiser/loss.
- Add backend step snapshots and frontend live rendering for each image in Slow mode.
- Show true label, prediction/probabilities, activations, updated weights and loss.

**Acceptance:** each observed frame corresponds to one real completed optimiser step, and displayed numbers can be verified against PyTorch logs.

### Milestone 4 — Controls and progress (completed and approved)

- Implement Start, Pause, Resume, Stop (or Reset), speed selection, and bottom progress bar.
- Ensure Slow mode permits human observation of each frame; Fast mode speeds up training by reducing wait/visual refresh, not by skipping optimiser steps.
- Show epoch and actual completed/total steps accurately.

**Acceptance:** Pause is reliable, Resume continues without restarting, Fast completes all intended samples, and progress reaches 100% on successful completion.

### Milestone 5A — Training Recorder (completed)

Goal: create a standalone Python program that trains the existing PyTorch model and records the results for offline visualisation.

Requirements:

1. Reuse the existing model, dataset, loss function, optimiser, and training logic in `training/`.
2. Keep `batch_size = 1`.
3. Train all 60,000 MNIST samples for one epoch, unless the existing configuration specifies otherwise. (`training/training.py` performs exactly one pass over its shuffled loader, so the recording is 60,000 optimiser steps.)
4. Record a snapshot **every 10 completed optimiser steps**.
5. Include the initial state at Step 0 and the final state.
6. Each recorded snapshot should contain:
   - actual completed training step;
   - MNIST sample image and true label;
   - predicted digit and Softmax probabilities;
   - loss value;
   - intermediate neuron activations;
   - updated weights for the fixed 1,010 Partial-mode connections.
7. Select the 1,010 Partial connections once, before training begins, using the existing initial Top-5 incoming absolute weights per receiving neuron.
8. Preserve those exact connection indices throughout the entire recording.
9. Save the initial full network weights (109,184 connections) only once.
10. Store the recordings in a compact format suitable for loading in a browser, preferably binary float32 arrays for numerical data.
11. Include a manifest describing the model architecture, recorded steps, fixed connection indices, colour scale, and data format.
12. Preserve the existing training algorithm. Do not introduce a second implementation of the model.

The recorded predictions and activations must be clearly identified as pre-update or post-update values, consistently with the recorded weights (Section 7.3).

Implementation notes agreed here (details in Sections 7.2–7.6):

- The recorder lives in a new `recorder/` area (for example `recorder/record_training.py`) and runs from the project virtualenv. It reuses the model-access helpers now in `recorder/model_access.py` (moved from `backend/visualiser.py` in Cleanup 2) and `training/training.py`'s `train_step()`; it imports nothing from the retired FastAPI app and does not write into `training/`.
- Recordings are written to a configurable output directory (for example `recordings/<run-name>/`), which is added to `.gitignore`.
- The fixed Partial indices are written once, and every checkpoint stores only the 1,010 current values at those indices.
- A short "how to record" section is added to `README.md` in this milestone.

**Acceptance criteria:**

- A complete recording exists with 6,001 checkpoints at steps 0, 10, 20, …, 60,000 and the manifest lists them.
- `model.bin` contains exactly 109,184 float32 initial weights and is written once; no later full-weight file exists.
- The manifest reproduces the architecture, the 1,010 fixed indices, the colour bounds, the capture convention and the binary layout.
- The 1,010 indices equal `select_partial_indices()` computed on the initial weights, and no checkpoint uses a different index set.
- A sample of recorded checkpoints matches the values observed during the recording run (loss, probabilities, activations, Partial weights) and the training log.
- No file in `training/` is modified, and `training/training.py`'s training maths is unchanged.
- Automated checks are added (for example `tests/test_milestone5a.py`) covering manifest completeness, counts, cadence, record size, fixed indices and the pre/post-update values. A shortened run may be used for the automated check; the full 60,000-step recording is verified manually and its size/time reported.

### Milestone 5B — Offline Replay Visualiser (completed)

Goal: replay the recorded training process in the browser without requiring Python, FastAPI, or WebSocket.

Requirements:

1. Reuse the existing frontend layout and neural network rendering implementation.
2. Load recorded training snapshots from local static files.
3. Add a timeline slider that allows seeking freely between recorded checkpoints.
4. Support Play, Pause, Previous, Next, and playback speed controls.
5. Update the MNIST image, prediction probabilities, neuron activations, loss, and Partial weights when the selected checkpoint changes.
6. The timeline must represent actual recorded optimiser steps, not artificial animation frames.
7. Do not interpolate between checkpoints unless explicitly approved.
8. Preserve the existing fixed absolute weight colour scale of ±0.5 and the light theme.

Connection modes are defined in Section 7.8: Full mode shows the 109,184 initial connections, static; Partial mode shows the same fixed 1,010 connections with their recorded weights updated per checkpoint. Neither mode reselects connections.

Implementation notes agreed here (details in Sections 7.7 and 6.1):

- The replay entry point and data adapter are added to `frontend/`; the Canvas renderer is reused, not duplicated.
- Data is read from the recording using relative URLs and on-demand chunk loading, so it works both from a local static server and from GitHub Pages.
- The page is explicitly labelled as a replay and states the pre-update / post-update convention.

**Acceptance criteria:**

- The page works when served by a plain static HTTP server with the Python backend stopped.
- Seeking to a checkpoint shows exactly that checkpoint's recorded image, probabilities, activations, loss, step and Partial weights; repeatedly seeking back to the same checkpoint shows identical values.
- The timeline labels and the progress readout use real recorded steps, and Previous/Next move exactly one checkpoint.
- Play, Pause, Previous, Next and speed control work, and playback never interpolates or skips a selected checkpoint.
- Full mode draws all 109,184 initial connections once and keeps them visually static while the timeline moves; Partial mode updates only the fixed 1,010 recorded weights.
- Colours follow the fixed ±0.5 scale in both modes with the light theme and existing legends.
- The replay page opens no WebSocket and makes no request outside the published data files; failures are surfaced to the user.
- **Superseded by Cleanup 3:** Live mode no longer exists (`./venv/bin/python -m backend.server` was removed), and the Milestone 2–4 tests were retired. This clause is historical.
- Automated checks are added (for example `tests/test_milestone5b.py`) covering manifest and chunk consistency, record decoding, the fixed indices, absence of WebSocket usage in the replay path and relative asset references. Browser behaviour is verified manually and reported.

### Milestone 5C — GitHub Pages Deployment (completed)

Goal: publish the offline replay visualiser as a standalone static website.

Requirements:

1. Generate a `docs/` directory containing the required frontend files and replay data.
2. Ensure the deployed website works without Python, FastAPI, PyTorch, or WebSocket.
3. Use relative paths compatible with GitHub Pages project URLs.
4. Load recorded data efficiently, preferably in chunks or on demand.
5. Preserve the local Python training environment separately.
6. Exclude raw MNIST datasets, virtual environments, and unnecessary training files from the deployed website.
7. Document the build and deployment process.

Deployment design is in Section 9.

**Acceptance criteria:**

- `docs/` is self-contained and works when served from a subdirectory, mirroring a project Pages URL.
- All asset and data paths are relative; no request leaves `docs/`.
- No dataset, virtualenv, raw recordings, tests, backend code or caches are present under `docs/`.
- The manifest and chunk files in `docs/` match the recording exactly (counts and checksums verified by the build step).
- The published URL passes the 5B acceptance checks with no local Python running.
- `README.md` documented recording, building `docs/`, publishing via GitHub Pages (`main` → `/docs`) and re-running the live mode locally. This was the 5C state; Cleanup 4 refreshed the README for the final offline-only architecture.
- No commit or push is made without explicit user approval.

**Historical note.** These 5C criteria were met when the live mode still existed. The clause about re-running the live mode locally is superseded: the live mode is retired, and Cleanup 3 removed its code and documentation.

### Project Cleanup phase — overview

The cleanup phase makes the repository match the final architecture in Section 2.2. It changes packaging only: no training mathematics, no recording bytes, no replay behaviour and no published-site behaviour may change. The order is mandatory — the recorder must stop importing `backend` (Cleanup 2) before `backend` can be removed (Cleanup 3).

### Cleanup 1 — Project Plan Update (completed, documentation only)

Goal: record the approved final architecture and the cleanup roadmap before any file is moved or deleted.

Requirements:

1. Update `plan.md` to mark Milestones 1–5C as completed historical milestones.
2. State that the final supported application is the offline recorder plus the static replay site.
3. Retire the requirement to preserve Live Training, the FastAPI backend and the WebSocket interface.
4. Keep the historical milestone descriptions for reference, clearly labelled as historical.
5. Document the target directory structure and the preservation rules (Section 2).
6. Add Cleanup 2, Cleanup 3 and Cleanup 4 with explicit requirements and acceptance criteria.
7. Keep GitHub Pages as the public deployment target.

Restrictions: modify `plan.md` only; do not delete or move files; do not modify application code; do not install or uninstall Python packages; do not create Git commits; do not push; do not start Cleanup 2.

Acceptance criteria:

- `git status --porcelain` lists only `plan.md`.
- No remaining requirement obliges the repository to keep the Live Training application, `backend/server.py`, FastAPI, Uvicorn or the WebSocket interface working.
- Every recording and replay requirement remains in force: the 6,001 checkpoints, the fixed 1,010 Partial connections, the ±0.5 colour scale, playback controls, chunk prefetching and the GitHub Pages deployment.
- Cleanup 2–4 are documented with their verification steps.

### Cleanup 2 — Shared Helper Migration (completed)

Goal: let the recorder stop importing the `backend` package without changing a single recorded byte.

Requirements:

1. Create `recorder/model_access.py` and move the recorder-required helpers there from `backend/visualiser.py`:
   - `load_training_module`, `linear_layers`, `describe_architecture`, `tensor_bytes`, `weights_checksum`, `training_totals`;
   - `PARTIAL_TOP_K`, `select_partial_indices`, `visible_weight_layers`, `encode_pixels`;
   - the private helpers those functions call (for example `encode_indices`) and the module-level constants they use (`REPO_ROOT`, `TRAINING_DIR`, `TRAINING_FILE`). `recorder/record_training.py` keeps its own `REPO_ROOT`.
   Do not move anything the recorder does not use. The recorder already imports `encode_pixels` lazily inside one function (for pixels); that import must also point at `recorder.model_access`.
2. Keep every function's behaviour and signature identical. Do not touch `training/training.py` or the training algorithm, loss, optimiser, batch size or `train_step()`.
3. Do not change the binary recording format or any recorded manifest text: format name/version, record size 5,688 bytes, field offsets, dtypes, chunk naming and the manifest prose stay exactly as they are in this milestone.
4. Update `recorder/record_training.py` to import from `recorder.model_access` and remove every `backend` import. The recorder must not import `backend.server`, FastAPI or Uvicorn.
5. Update `tests/test_milestone5a.py` (and any other test importing the migrated names) to use `recorder.model_access`.
6. Leave the rest of `backend/` untouched; it was removed in Cleanup 3.

Verification:

1. Before the change, record a seeded reference run:
   `./venv/bin/python -m recorder.record_training --output /tmp/cleanup2-before --steps 100 --seed 1234`
2. After the change, record the same run to `/tmp/cleanup2-after`.
3. Compare the two directories. Every binary file (`model.bin`, `model-biases.bin`, `partial-indices.bin` and every chunk) must be byte-identical. `manifest.json` must be identical except the volatile fields `training.recorded_at`, `training.duration_seconds` and `notes.command` (which contains the output path); compare after removing exactly those fields.
4. Run `tests/test_milestone5a.py` and report the result.
5. Confirm `grep -R "from backend" recorder/` returns nothing.

Acceptance criteria:

- `recorder/` imports nothing from `backend/`.
- The seeded recording matches the pre-migration recording under the comparison above.
- The existing `recordings/mnist-epoch1/` recording is untouched and still validates.
- The training algorithm, the binary format and the recorded manifest prose are unchanged.
- No new dependency is added.

### Cleanup 3 — Legacy Removal (completed)

Goal: delete the retired Live Training application once the recorder no longer needs it.

Requirements:

1. Remove `backend/server.py` and `backend/trainer.py`.
2. Remove the rest of the `backend/` package (`backend/__init__.py`, `backend/visualiser.py`) once nothing imports it. If something still needs a migrated helper, that is a Cleanup 2 defect to fix there rather than a reason to keep the package.
3. Remove the Live Training frontend: `frontend/index.html` and `frontend/main.js`.
4. Keep everything the replay needs: `frontend/replay.html`, `frontend/style.css`, `frontend/network.js`, `frontend/panels.js`, `frontend/recording.js`, `frontend/replay.js`, `docs/` and `recorder/`.
5. Retire the tests that exist only to verify the removed application: `tests/test_milestone2.py`, `tests/test_milestone3.py` and `tests/test_milestone4.py`. Trim rather than delete `tests/test_milestone5b.py`: remove only its Live Training checks (`main.js`/`index.html` assertions and the live-page clauses) and keep every replay check. Update `tests/test_milestone5c.py` where it asserts the two documented page edits, the live-page note and the unchanged `backend`/live-page paths; keep its build, data, HTTP and browser checks. Also update `tests/test_milestone5a.py`: its "`training/` and `backend/` are unchanged" Git-status check must assert only that `training/` is untouched (the `backend/` deletion is intentional), while its "the recorder never imports the server" check stays.
6. Remove outdated links and references:
   - in `frontend/replay.html`, replace the sentence that links to the live page with the neutral published sentence `The live training page is not part of this published site.`, which is already what `docs/index.html` shows today;
   - in `scripts/build_pages.py`, remove the `LIVE_MODE_PATTERN` rewrite and its validation so the only page edit is the `data-recording` path, while still rejecting a live-page link;
   - in `README.md`, remove the description of `./venv/bin/python -m backend.server`, the WebSocket protocol and the live controls.
7. Do not remove any functionality needed by the Recorder or Replay, and keep the `.gitignore` entries for `recordings/`, `venv/` and `training/data/`.

Verification:

1. Searching the repository (excluding `venv/`, `training/data/`, `recordings/`, `.git/`) finds no live-mode imports, links or run instructions.
2. `./venv/bin/python -m recorder.record_training --output /tmp/cleanup3 --steps 100 --seed 1234` succeeds with `backend/` gone, and its binary files match the Cleanup 2 reference recording.
3. Serving `docs/` statically still loads the replay page with no console errors or failed requests.
4. All remaining tests run; report passes and any intentionally retired checks.

Acceptance criteria:

- `backend/` and the Live Training frontend files are gone.
- The recorder runs with no `backend` package present and produces the same bytes.
- The replay page and `docs/` still work, and no remaining file references the removed live mode.
- No functionality required by the recorder or the replay was removed.
- The existing recording is unchanged and still valid.

### Cleanup 4 — Final Verification (verification complete — awaiting user approval)

Goal: prove the cleaned project is the documented final architecture.

Requirements:

1. Update `README.md` to describe only the final application: recording, replaying locally, rebuilding and publishing `docs/`, running the remaining tests, and the preservation guarantees (6,001 checkpoints, 1,010 Partial connections, ±0.5 scale).
2. Create `requirements.txt` with only the Python runtime dependencies the final application imports: `torch`, `torchvision` and `numpy`. FastAPI, Uvicorn and WebSocket libraries are absent because nothing imports them. Test-only tools (for example Playwright) are documented in `README.md`, not listed as runtime requirements.
3. Review unused imports and stale documentation in `recorder/`, `scripts/`, `tests/`, `frontend/` and `README.md`.
4. Verify the recorder works without FastAPI or Uvicorn: run a short recording in an environment where they are unavailable (for example a clean virtualenv built from `requirements.txt`), or otherwise prove and report that no module imports them.
5. Verify `recordings/mnist-epoch1/` remains valid — 6,001 checkpoints at 0, 10, …, 60,000, 13 chunks, 5,688-byte records, fixed 1,010 indices and the manifest's initial-weight checksum. `scripts/build_pages.py` already performs these checks.
6. Rebuild `docs/` with `./venv/bin/python scripts/build_pages.py` and verify byte-for-byte equivalence for the deployed replay assets: every file under `docs/data/`, every JavaScript module, `style.css` and `.nojekyll` must be identical to the currently deployed copies. Because Cleanup 3 keeps the published page's wording unchanged, `docs/index.html` must also be byte-identical; if it is not, stop and report the difference before publishing.
7. Run all remaining tests (`tests/test_milestone5a.py`, `tests/test_milestone5b.py`, `tests/test_milestone5c.py`) including the browser checks.
8. Confirm GitHub Pages functionality is unchanged: initial network, timeline seeking, Play/Pause/Previous/Next, all playback speeds, Full and Partial modes, the ±0.5 colours, chunk loading and error handling.

Acceptance criteria:

- `requirements.txt` lists only `torch`, `torchvision` and `numpy`; a short recording runs from an environment built from it.
- No file imports FastAPI, Uvicorn or a WebSocket library.
- The existing recording validates and the rebuilt `docs/` replay assets, including `index.html`, are byte-identical.
- All remaining tests and browser checks pass; failures are reported, not hidden.
- The published GitHub Pages URL behaves exactly as before, and the repository contains only the architecture in Section 2.2.

### Deferred backlog — neuron inspection and usability (former Milestone 5)

- Add neuron focus/hover details and corresponding connections.
- Add a compact loss history chart if desired.
- Verify legibility, legends, resize behaviour and performance.

Status: not authorised and not part of Milestones 5A–5C or the Cleanup phase. This work needs a separate milestone and explicit approval; it must not be bundled into the replay or cleanup milestones.

## 9. Deployment design (Milestone 5C, completed — GitHub Pages remains the deployment target)

### 9.1 Published layout

```text
docs/
├── .nojekyll                # do not run Jekyll over the static site
├── index.html               # built from frontend/replay.html
├── style.css
├── network.js               # the only network renderer
├── panels.js                # panel rendering
├── recording.js             # binary recording decoder
├── replay.js                # replay data source, controls and timeline
└── data/
    ├── manifest.json
    ├── model.bin
    ├── model-biases.bin
    ├── partial-indices.bin
    └── chunks/
        ├── chunk-0000.bin
        └── …
```

`docs/` contains only what the replay needs and no Python. Its contents were settled in 5B/5C; nothing about Python, the recorder, tests or the dataset may be copied into it, and the cleanup phase must not change the deployed files.

### 9.2 Paths

- Every reference is relative: `./style.css`, `./replay.js`, `./data/manifest.json`, and manifest entries relative to the `data/` directory.
- No root-absolute paths (`/…`), no `file://` assumptions and no dependence on the repository name.
- The same files must work at `https://<user>.github.io/<repo>/` and when served locally from a subdirectory.

### 9.3 Build process

- A small build script (for example `scripts/build_pages.py`, run with the project virtualenv) copies the frontend files and one chosen recording into `docs/`, and verifies record counts, byte sizes and checksums against the manifest.
- The script refuses to copy the MNIST dataset, `venv/`, `recordings/` sources, `tests/`, `recorder/`, `__pycache__/`, editor metadata and Git files. The `backend/` exclusion stays as a safeguard after Cleanup 3 removed the package.
- The published `docs/` content is committed, so no build step runs in the browser and no Python is needed by visitors.
- Rebuilding for a new recording replaces the data files in place and must leave the manifest and chunks consistent. The build fails loudly rather than publishing a mismatched recording.

### 9.4 Exclusions

| Excluded | Reason |
| --- | --- |
| `training/data/` (raw/processed MNIST) | Large, and only needed to record or train locally |
| `venv/` | Machine-specific Python environment |
| `recordings/` | Local recorder output; only the published copy under `docs/data/` ships |
| `backend/` (removed in Cleanup 3), `tests/`, `recorder/` | Not required by a static site |
| `__pycache__/`, `*.py[cod]`, `*:Zone.Identifier` | Caches and platform noise |
| `.git/`, editor configuration | Never published |

### 9.5 Publishing steps (documented in `README.md` during 5C; refreshed in Cleanup 4)

1. Generate a recording with the recorder from 5A.
2. Run the build script to produce `docs/`.
3. Review the diff, then commit `docs/` — only with explicit user approval; no automatic commits or pushes.
4. On GitHub: **Settings → Pages → Deploy from a branch → `main` → `/docs`**.
5. Open the published URL and repeat the Milestone 5B acceptance checks there.

### 9.6 Verification

- Serve the repository root with a plain static server (for example `./venv/bin/python -m http.server 8100`) and open `http://localhost:8100/docs/` — a subpath, like a project Pages URL.
- Confirm the page loads with no Python server running, opens no WebSocket, and requests nothing outside `docs/`.
- Confirm timeline seeking, Play/Pause/Previous/Next, speed, Full-mode static colours, Partial-mode updates and the ±0.5 colour scale.
- Confirm a cold reload and a cached reload both work, and that a deliberately missing chunk produces a clear error instead of a wrong frame.
- Confirm the recorder still produces a valid recording and that the remaining tests pass. The earlier requirement that `./venv/bin/python -m backend.server` and the Milestone 2–4 tests keep working is retired (Section 2.1).

## 10. Performance risks and mitigations

The live-mode rows are retained as historical risk records. Only the replay rows describe the final application.

| Risk | Mitigation |
| --- | --- |
| Full initial overview draws ~109k lines | Use Canvas, render/cache the static overview once; reuse the existing cached bitmap in `network.js` |
| Batch size 1 requires many updates | The recording runs once without a browser; replay uses 6,001 checkpoints instead of 60,000 frames |
| Full weights over WebSocket every step are expensive | Live mode sends only visible connections; replay stores full weights once and 1,010 values per checkpoint |
| Training blocks the local web server | Live training runs on a single worker thread; the recorder runs as a separate offline process |
| Predictions and weights belong to different moments | Explicit pre-update / post-update labelling in the manifest, UI and tests (Section 7.3) |
| Browser disconnects mid-training | Live mode pauses safely; replay has no connection to lose |
| Dense 784-input layer is hard to inspect | Compact neuron layout and focus inspection; the renderer already handles scale |
| A 34 MB recording exceeds comfortable page weight | Never download it all: fetch the manifest, initial weights and indices once, then load chunks on demand with a small cache and prefetch |
| Seeking far along the timeline stalls on a chunk fetch | Show a loading state, keep recent chunks cached, prefetch the next chunk during playback |
| GitHub Pages file and repository limits | 13 chunk files of ≈ 2.8 MB are well under the 100 MB per-file limit; keep raw datasets and environments out of `docs/` |
| Rebuilding grows Git history with binary data | Only rebuild when the recording changes; if the repository becomes too large, reduce cadence or precision only with explicit approval |
| A recording and frontend drift out of sync | Manifest format version, field offsets and checksums; the replay refuses an unknown version and the build verifies counts |
| 60,000 CPU optimiser steps take time to record | Record once, log progress, report elapsed time and size; no need to repeat for viewing |
| Interrupted recording looks usable | Recordings are written only as complete runs; a partial output is invalid and is regenerated |

## 11. Plan evolution: live streaming versus offline replay

- Milestones 1–4 implemented the original Plan A (live WebSocket streaming) and were approved. That mode is now retired; its code was removed in Cleanup 3.
- Plan B (record first, replay later) was originally the fallback. It became the chosen architecture for the published visualiser, because replaying removes the browser/server coupling, makes every displayed frame reproducible and freely seekable, and allows free static hosting. It is now the only supported application.
- Consequences of the change:
  - the replay timeline has 6,001 checkpoints (every 10 optimiser steps), not one frame per training sample;
  - full-network weights are recorded only at Step 0, so Full mode is intentionally static;
  - the shown prediction/activations always come from the step's forward pass, while the Partial weights are that step's post-update state;
  - changing the model, the data pipeline or the cadence requires a new recording; the manifest and checksums make mismatches detectable.
- The cleanup phase completed the transition: Cleanup 1 recorded it, Cleanup 2 removed the recorder's dependency on `backend/`, Cleanup 3 deleted the live mode and its documentation, and Cleanup 4 verified the final project.

## 12. Out of scope for the current version

- Rewriting the trained model in JavaScript.
- Maintaining, reviving or extending the retired Live Training application, FastAPI server or WebSocket interface.
- Training inside the browser, or replaying through a Python server.
- Supporting arbitrary neural network architectures beyond the existing MNIST MLP.
- Distributed training, cloud deployment or account systems.
- Detailed gradient/backpropagation animations or optimiser internals unless explicitly requested later.
- Saving an enormous complete model snapshot for every single sample.
- Interpolation, tweening or synthetic frames between recorded checkpoints.
- Resumable recordings, multiple epochs, multiple recordings in one player, or per-step full-network weights.
- Recording gradients, Adam moments or other optimiser state.
- Git LFS, GitHub Actions deployment automation or server-side rendering, unless separately approved.

## 13. Definition of done

### 13.1 Version 1 — live visualiser (met at Milestone 4) — historical, superseded

The user can open a local webpage, see **all neurons and connections before training**, press Start to observe actual **batch-size-1 PyTorch training**, see the display switch to **partial connections**, watch each MNIST image and its predicted probabilities with neuron activations and changing weight colours, adjust between slow human-observable and fast modes, pause/resume safely, and watch a correct progress bar below the visualisation reach completion.

### 13.2 Version 2 — offline replay (Milestones 5A–5C) — completed

A visitor can open the GitHub Pages URL on a machine with no Python installed, see the initial network with all 109,184 connections, drag the timeline to any recorded step between 0 and 60,000, step forwards and backwards one checkpoint at a time, play or pause the recording, watch the recorded MNIST sample, softmax probabilities, neuron activations, loss and the changing 1,010 Partial connection weights, switch to Full mode and see the initial network remain static, and read from the page and the manifest which values are pre-update and which are post-update. The displayed numbers match the Python training log for the recorded steps.

This was completed when the user approved Milestones 5A, 5B and 5C in order; the site is deployed on GitHub Pages and the cleanup phase must not change it.

### 13.3 Final supported project (Cleanup 4 — awaiting approval)

The repository contains only the final architecture in Section 2.2. A developer can install `torch`, `torchvision` and `numpy` from `requirements.txt`, run the recorder to produce a recording, replay it from `frontend/replay.html` or the published `docs/` URL, and rebuild and publish `docs/` with `scripts/build_pages.py` — with no FastAPI, Uvicorn or WebSocket dependency anywhere.

The existing 6,001 checkpoints, the fixed 1,010 Partial connections, the ±0.5 colour scale, the playback behaviour and the published GitHub Pages site are unchanged from the approved 5C state.

Cleanup 2 and Cleanup 3 are approved. This final state is complete when the user explicitly approves Cleanup 4.
