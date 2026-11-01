# Neural Network Training Visualiser — Implementation Plan

## 1. Project goal

Build a local web-based **real-time neural network training visualiser** for the existing PyTorch MNIST multilayer perceptron (MLP). The project is intended for learning and observation: it should show how an individual handwritten-digit image travels through the network, what the network predicts, and how the weights change after training.

**Primary approach (Plan A):** train in Python while streaming training state to an HTML/CSS/JavaScript frontend. **Fallback (Plan B):** record training snapshots first and replay them later if real-time streaming proves too slow or difficult to coordinate.

The existing PyTorch model, training data, and training algorithm are the source of truth. Do not rewrite the model in JavaScript.

## 2. Confirmed design decisions

- **Development environment:** WSL2 Ubuntu for Python, PyTorch, backend and agent workflows; Windows for VS Code and browser. Serve the web UI locally (for example, `http://localhost:8000`).
- **Model:** use the existing MNIST MLP (currently expected to be `784 → 128 → 64 → 10`); verify the actual definitions in the supplied code rather than hard-coding assumptions.
- **Dataset:** use the existing MNIST data pipeline.
- **Batch size:** **1**. One training step processes one image and performs one optimiser update.
- **Visualisation:** show all neurons and all connections in an initial network overview **before training starts**. When training begins, automatically switch to a reduced set of connections while keeping the network structure visible. The viewer should be able to inspect individual neurons and their connections.
- **Weight encoding:** distinguish positive and negative weights with two contrasting colours; use opacity/intensity or line width to indicate magnitude. Include a legend. Avoid treating the sign of a weight as the sign of its activation.
- **Training information:** current 28 × 28 image, true label, predicted digit, all 10 class scores/probabilities, current loss, epoch, sample/step number, and network activations/weights.
- **Playback controls:** Start, Pause, Resume, and Stop/Reset as appropriate; speed selection including a **slow observation mode** and a **fast mode**.
- **Progress:** place a progress bar **below the main visualisation**, with completed steps / total steps and current epoch. Progress indicates actual training steps, not animation time.
- **Prioritise correctness and clarity over excessive animations.** Work in small verifiable increments.

## 3. UX specification

### 3.1 Main page layout

1. **Header:** project title, model architecture, current status (Ready / Training / Paused / Finished / Error).
2. **Network view (central focus):** neurons arranged by layer with weighted connections. The initial view shows every connection. Once training starts, render only a defined subset of connections by default (see Section 4).
3. **Current sample panel:** current MNIST image enlarged with crisp pixels, ground-truth label, predicted digit, and whether the prediction is correct. Make clear whether the displayed prediction was produced **before** or **after** the optimiser update (use one consistent convention; prefer forward-pass prediction before update).
4. **Output panel:** 10 bars labelled 0–9, showing `softmax(logits)` probabilities; optionally expose logits separately so they are not mistaken for probabilities.
5. **Training metrics:** current loss, epoch, step, and optionally a small historical loss chart.
6. **Control row and bottom progress bar:** Start / Pause / Resume, speed selection, and progress `completed / total` with a percentage. Keep the progress bar visible at all times.

### 3.2 Network display states

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

### 3.3 Speed and timing

- Training progress should advance **one sample per optimiser step** (`batch_size=1`).
- **Slow/Observe:** after each step, wait for the frontend to acknowledge or permit the next step, or otherwise use a controlled delay. This ensures humans can observe each image/weight update without missing frames. The UI must remain responsive.
- **Normal:** moderate visualisation cadence.
- **Fast:** remove deliberate waits and reduce UI update frequency (e.g., display every Nth training step), **without changing the underlying optimiser steps or silently skipping training samples**. The progress bar must continue to represent actual completed steps.
- Choose practical defaults after testing. Do not make animation speed alter the learning rate or mathematics of training.
- Pause should take effect at a safe boundary between training steps (not halfway through `loss.backward()` or `optimiser.step()`).

## 4. Backend design (Python)

The project already contains working PyTorch training code, data-loading utilities, and the MNIST dataset in the `training/` directory.

The backend must reuse the existing implementation rather than creating a separate training system.

Suggested project structure:

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

The filenames inside `training/` are illustrative. Inspect the actual files before implementing anything.

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

### 4.1 Training step sequence

For each `(image, label)` at batch size 1:

1. Acquire the current image/label using the existing data-loading implementation. Normalise and reshape exactly as the existing model expects.
2. Compute the forward pass using the existing model. Capture logits and intermediate layer activations.
3. Compute loss, backward pass, and `optimiser.step()` using the **existing** training logic.
4. Capture the **updated** layer weights and biases at the chosen observation cadence.
5. Publish a snapshot containing the sample image, true label, pre-update prediction/logits/probabilities, loss, activations, updated weight information, and progress metadata. Label the before/after timing consistently.
6. Respect Pause/Stop and speed control **at a safe step boundary**, then proceed to the next image.

For the existing MLP, intermediate activations can be captured through explicit variables or PyTorch forward hooks. Prefer the least intrusive approach compatible with the current implementation.

Detach and move tensors to CPU for serialisation. Do not transmit tensors requiring gradient tracking.

### 4.2 Communication

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

### 4.3 Correctness and stability

- Do not block the FastAPI event loop with the training loop. Run training in a worker/thread/process as appropriate and safely relay state to the WebSocket.
- Handle disconnects gracefully. Prefer pause-on-disconnect for the initial observation-oriented design.
- Ensure pause/resume and progress counters reflect actual completed optimiser steps.
- Never modify the optimiser or learning rate when only changing the visualisation speed.
- Keep all training computations within the existing PyTorch implementation.
- Ensure the visualisation backend does not accidentally trigger a second training process.
- Preserve compatibility with the existing dataset paths and loading functions.
- Any necessary modification to files inside `training/` must be minimal, explained, and limited to the currently authorised milestone.
## 5. Frontend design

- Start with vanilla HTML + CSS + JavaScript; no React dependency required.
- Use **Canvas** for network lines and neurons to avoid a huge number of DOM/SVG elements in the full overview. Use ordinary HTML for panels, controls and labels.
- Keep rendering and WebSocket/event handling separate (`network.js` versus `main.js`).
- Draw all connections **once** in Ready state, then show the focused subset while Training.
- Distinguish weight colour/intensity from neuron activation colour/intensity; show clear legends.
- On each received snapshot, update image, predictions, loss, highlights, visible weight colours and progress. Prevent overlapping animation frames and stale state after Pause/Reset.
- During Fast mode, coalesce/redraw at an affordable cadence while still counting every trained sample.

## 6. Implementation milestones for the coding agent

Implement **one milestone at a time**. Do not redesign the user’s existing PyTorch model or introduce unrelated functionality.

### Milestone 1 — Inspect existing project and build static UI

- Read `train.py`/model/data pipeline to confirm actual architecture and dataset format.
- Build the page layout, neuron/connection renderer, legend, sample/output panels and bottom progress bar with **mock data clearly marked as mock**.
- Verify that the Ready-state renderer can draw **all** network connections without freezing the browser.

**Acceptance:** page loads locally, full overview is drawn before Start, and training view can switch to partial lines.

### Milestone 2 — Connect real model state

- Start the Python server and establish the WebSocket.
- Send real architecture/initial weights from the loaded PyTorch model.
- Show the real initial network in Ready state; no training controls required yet.

**Acceptance:** displayed layer counts/weights come from the model rather than hard-coded mock values.

### Milestone 3 — Single-sample real-time training

- Set `batch_size=1` and preserve the existing optimiser/loss.
- Add backend step snapshots and frontend live rendering for each image in Slow mode.
- Show true label, prediction/probabilities, activations, updated weights and loss.

**Acceptance:** each observed frame corresponds to one real completed optimiser step, and displayed numbers can be verified against PyTorch logs.

### Milestone 4 — Controls and progress

- Implement Start, Pause, Resume, Stop (or Reset), speed selection, and bottom progress bar.
- Ensure Slow mode permits human observation of each frame; Fast mode speeds up training by reducing wait/visual refresh, not by skipping optimiser steps.
- Show epoch and actual completed/total steps accurately.

**Acceptance:** Pause is reliable, Resume continues without restarting, Fast completes all intended samples, and progress reaches 100% on successful completion.

### Milestone 5 — Inspect neurons and improve usability

- Add neuron focus/hover details and corresponding connections.
- Add a compact loss history chart if desired.
- Verify legibility, legends, resize behaviour and performance.

**Acceptance:** user can investigate individual neuron activations/weights and see meaningful changes during training.

## 7. Performance risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Full initial overview draws ~109k lines | Use Canvas, render/cached static overview once; switch to partial lines before updates |
| Batch size 1 requires many updates | Adjustable pacing, coalesced fast-mode rendering and progress-only updates |
| Full weights over WebSocket every step are expensive | Send visible connection values; use occasional full snapshots or binary encoding if needed |
| Training blocks the local web server | Use a worker/thread/process with coordinated pause and event messaging |
| Predictions and weights belong to different moments | Explicitly label pre-update outputs and post-update weights |
| Browser disconnects mid-training | Pause safely and permit reconnect, or document a simpler stop/restart policy |
| Dense 784-input layer is hard to inspect | Show compact neuron layout, zoom/pan or focused inspection without hiding actual model size |

## 8. Plan B — Offline capture and replay (fallback only)

If live training/rendering cannot keep up, or networking becomes disproportionately complex:

1. Keep the same frontend and rendering code.
2. Replace live WebSocket step events with a recorded stream or structured snapshot files.
3. During PyTorch training, record per-step image, label, prediction, loss and needed activations, plus sampled weight snapshots (not necessarily every full tensor every step).
4. Replay snapshots in the same UI with Slow/Normal/Fast, Pause, timeline scrub and progress.
5. Clearly identify this as **replay**, not live training.

This is a fallback, not the starting implementation.

## 9. Out of scope for the first version

- Rewriting the trained model in JavaScript.
- Supporting arbitrary neural network architectures beyond the existing MNIST MLP.
- Distributed training, cloud deployment or account systems.
- Detailed gradient/backpropagation animations or optimiser internals unless explicitly requested later.
- Saving an enormous complete model snapshot for every single sample.

## 10. Definition of done

The project is complete for Version 1 when the user can open a local webpage, see **all neurons and connections before training**, press Start to observe actual **batch-size-1 PyTorch training**, see the display switch to **partial connections**, watch each MNIST image and its predicted probabilities with neuron activations and changing weight colours, adjust between slow human-observable and fast modes, pause/resume safely, and watch a correct progress bar below the visualisation reach completion.
