/**
 * main.js — Page state, WebSocket client, panel rendering and UI wiring.
 *
 * Every number on the page comes from the backend. When the page connects, the
 * server sends the real PyTorch model as a `model_info` message, followed by an
 * `inference` message for one real MNIST sample: its 28 x 28 pixels, the true
 * label, the forward pass logits and softmax probabilities, the activations of
 * every layer and the cross-entropy loss of that sample.
 *
 * Training (Milestone 4) is driven from here and never starts on its own:
 *
 *   start / pause / reset / set_speed   -> backend/trainer.py TrainingSession
 *   training_state                      <- idle | running | paused | complete
 *   training_step                       <- one real optimiser step, over and over
 *   request_weights / weights           <- every current weight, for the Full view
 *
 * A `training_step` frame carries the newest weights together with the image,
 * prediction and loss of the step that was just taken. network.js keeps
 * handling all canvas drawing; this file decides which of that data to show.
 */

import {
  ARCHITECTURE,
  MAX_DISPLAY_WEIGHT,
  MIN_DISPLAY_WEIGHT,
  NetworkView,
  PARTIAL_TOP_K,
  countConnections,
  weightColour,
} from "./network.js";

/** WebSocket endpoint exposed by backend/server.py. */
const SOCKET_PATH = "/ws";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;
/** Re-enable the button if the backend never answers a next_sample request. */
const NEXT_SAMPLE_TIMEOUT_MS = 5000;
/**
 * The Full view needs all 109,184 weights, which is far too much to send with
 * every step, so it refreshes its weights at most this often while training.
 */
const FULL_WEIGHT_REFRESH_MS = 1000;

const elements = {
  architectureSummary: document.getElementById("architecture-summary"),
  batchSize: document.getElementById("batch-size"),
  statusBadge: document.getElementById("status-badge"),
  modelBadge: document.getElementById("model-badge"),
  networkCanvas: document.getElementById("network-canvas"),
  weightScaleBar: document.getElementById("weight-scale-bar"),
  connectionNote: document.getElementById("connection-note"),
  modeAll: document.getElementById("mode-all"),
  modePartial: document.getElementById("mode-partial"),
  sampleCanvas: document.getElementById("sample-canvas"),
  sampleIndex: document.getElementById("sample-index"),
  sampleTrue: document.getElementById("sample-true"),
  samplePredicted: document.getElementById("sample-predicted"),
  sampleVerdict: document.getElementById("sample-verdict"),
  nextSample: document.getElementById("btn-next-sample"),
  outputList: document.getElementById("output-list"),
  metricLoss: document.getElementById("metric-loss"),
  metricEpoch: document.getElementById("metric-epoch"),
  metricStep: document.getElementById("metric-step"),
  progressBar: document.getElementById("progress-bar"),
  progressFill: document.getElementById("progress-fill"),
  progressText: document.getElementById("progress-text"),
  progressPercent: document.getElementById("progress-percent"),
  start: document.getElementById("btn-start"),
  pause: document.getElementById("btn-pause"),
  reset: document.getElementById("btn-reset"),
  speedSelect: document.getElementById("speed-select"),
  controlsNote: document.getElementById("controls-note"),
};

/** The `model_info` payload from the backend, or null until it arrives. */
let model = null;
/** The newest `training_state` payload: idle, running, paused or complete. */
let training = null;
/** Output bar elements, built once and then updated in place. */
let outputRows = [];
/** Timer that re-enables the Next sample button if a reply never arrives. */
let nextSampleTimeout = 0;
/**
 * Highest training step already drawn. Frames can arrive slightly out of order
 * after a reconnect, so an older frame must never overwrite a newer one.
 */
let lastFrameStep = 0;
/** True while the canvas holds only the selected weights of a training frame. */
let weightsArePartial = false;
/** Newest complete weight matrices, kept for the Full view between snapshots. */
let fullWeights = [];
/** Timestamp of the last full-weight request, to throttle the Full view. */
let lastFullWeightRequest = 0;

// The view starts without weights: neurons and captions render immediately and
// the 109,184 connections appear as soon as the model arrives.
const view = new NetworkView(elements.networkCanvas, { architecture: ARCHITECTURE });

function formatNumber(value) {
  return value.toLocaleString("en-GB");
}

/** Updates the header status badge: connecting, ready or failed. */
function setStatus(text, tone) {
  elements.statusBadge.textContent = text;
  elements.statusBadge.classList.toggle("is-ready", tone === "ready");
  elements.statusBadge.classList.toggle("is-error", tone === "error");
}

/**
 * Decodes the backend's base64 weight/activation payload into float32 values.
 * The server sends raw little-endian bytes in row-major order — the same layout
 * PyTorch uses — so this is a straight copy into a Float32Array.
 */
function decodeFloat32(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Decodes one base64 byte payload — the 28 x 28 sample image — into values in
 * 0..1, the same range `torchvision.transforms.ToTensor()` produces.
 */
function decodePixels(base64, expectedLength) {
  const binary = window.atob(base64);
  if (expectedLength && binary.length !== expectedLength) {
    throw new Error(`expected ${expectedLength} pixels, got ${binary.length}`);
  }
  const pixels = new Float32Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    pixels[i] = binary.charCodeAt(i) / 255;
  }
  return pixels;
}

/** Decodes the connection indices of a partial training frame (uint16). */
function decodeUint16(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Uint16Array(bytes.buffer);
}

/**
 * Turns the selected connections of one training frame back into the dense
 * [out][in] matrices network.js draws. Everything outside the selection stays
 * zero: those connections are simply not drawn in partial mode, and the drawn
 * indices come from the same fixed selection in every frame.
 */
function decodeVisibleWeights(layers) {
  return layers.map((layer) => {
    const inSize = layer.in;
    const outSize = layer.out;
    const k = layer.k;
    const indices = decodeUint16(layer.indices);
    const values = decodeFloat32(layer.values);
    const weights = new Float32Array(inSize * outSize);

    for (let out = 0; out < outSize; out += 1) {
      const rowOffset = out * inSize;
      for (let position = 0; position < k; position += 1) {
        const slot = out * k + position;
        weights[rowOffset + indices[slot]] = values[slot];
      }
    }
    return weights;
  });
}

/**
 * The fixed Partial-view indices of `model_info`, as per-neuron rows: exactly the
 * indices the backend chose from the initial weights, so both sides draw the same
 * 1,010 connections for the whole run.
 */
function decodeSelection(partial) {
  const layers = (partial && partial.selection) || [];
  return layers.map((layer) => {
    const indices = decodeUint16(layer.indices);
    const k = layer.k;
    const rows = new Array(layer.out);
    for (let out = 0; out < layer.out; out += 1) {
      rows[out] = indices.subarray(out * k, (out + 1) * k);
    }
    return rows;
  });
}

function handleModelInfo(message) {
  const layers = message.layers || [];
  const architecture = message.architecture || [];
  const weights = layers.map((layer) => decodeFloat32(layer.weights));

  // Guard against a truncated or mismatched payload before drawing 109k lines.
  const shapesMatch =
    weights.length === architecture.length - 1 &&
    weights.every((values, index) => values.length === layers[index].in * layers[index].out);
  const connectionCount = weights.reduce((total, values) => total + values.length, 0);

  if (!shapesMatch || connectionCount !== countConnections(architecture)) {
    console.error("model_info does not match its own architecture", message);
    setStatus("Model data invalid", "error");
    return;
  }

  model = message;
  // The backend chose the Partial view's connections once from these initial
  // weights; handing them over keeps both sides on exactly the same indices.
  const selection = decodeSelection(message.partial);
  view.setModel({
    architecture,
    weights,
    selection: selection.length === weights.length ? selection : null,
  });
  fullWeights = weights;
  weightsArePartial = false;

  elements.architectureSummary.textContent = architecture.map((layer) => layer.size).join(" → ");
  if (message.batch_size) elements.batchSize.textContent = String(message.batch_size);
  elements.modelBadge.textContent =
    message.weights_state === "trained" ? "Real weights" : "Real weights · untrained";

  // The same payload carries the training status, so a page that is reloaded
  // mid-run shows the right buttons, speed and progress straight away.
  if (message.training) applyTrainingState(message.training);
  else setStatus("Ready", "ready");

  console.info(
    `model_info: ${architecture.map((layer) => layer.size).join("→")}, ` +
      `${formatNumber(connectionCount)} weights, sha256 ${String(message.weights_checksum).slice(0, 16)} ` +
      `from ${message.weights_source} (${message.weights_state}), ` +
      `${formatNumber(message.partial ? message.partial.count : 0)} partial connections fixed`,
  );
}

/**
 * Renders one real inference snapshot: sample image, labels, softmax
 * probabilities, loss and the per-layer activations on the network canvas.
 */
function handleInference(message) {
  const sample = message.sample || {};
  const width = sample.width || 28;
  const height = sample.height || 28;

  let image;
  let probabilities;
  let activations;
  try {
    image = { width, height, pixels: decodePixels(sample.pixels, width * height) };
    probabilities = decodeFloat32(message.probabilities);
    activations = (message.activations || []).map(decodeFloat32);
  } catch (error) {
    console.error("could not read the inference message", error, message);
    setStatus("Sample data invalid", "error");
    return;
  }

  // A forward pass must never move a weight; the checksum proves it. Once a real
  // optimiser step has run, the weights legitimately differ from `model_info`.
  if (model && !hasTrained() && message.weights_checksum !== model.weights_checksum) {
    console.error(
      "weights changed between model_info and an inference sample",
      message.weights_checksum,
      model.weights_checksum,
    );
    setStatus("Weights changed", "error");
  }

  drawSampleImage(elements.sampleCanvas, image);

  const correct = message.predicted_label === message.true_label;
  elements.sampleIndex.textContent = `${formatNumber(sample.index + 1)} / ${formatNumber(sample.total)}`;
  elements.sampleTrue.textContent = String(message.true_label);
  elements.samplePredicted.textContent = String(message.predicted_label);
  elements.sampleVerdict.textContent = correct ? "Correct" : "Incorrect";
  elements.sampleVerdict.className = `pill ${correct ? "pill-ok" : "pill-bad"}`;

  updateOutputRows(outputRows, {
    probabilities,
    trueLabel: message.true_label,
    predictedLabel: message.predicted_label,
  });

  elements.metricLoss.textContent = Number(message.loss).toFixed(4);
  view.setActivations(activations);

  window.clearTimeout(nextSampleTimeout);
  elements.nextSample.disabled = training?.state === "running";

  console.info(
    `inference: sample ${sample.index + 1}/${sample.total}, true ${message.true_label}, ` +
      `predicted ${message.predicted_label}, loss ${Number(message.loss).toFixed(4)}, ` +
      `activations ${activations.map((values) => values.length).join("/")}`,
  );
}

/** True once at least one real optimiser step has been taken. */
function hasTrained() {
  return Boolean(training && training.step > 0);
}

/** Sends one command to the backend, if the socket is open. */
function sendCommand(command) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(command));
  return true;
}

/** Header badge text for each training state. */
const STATE_BADGES = {
  idle: "Ready",
  running: "Training",
  paused: "Paused",
  complete: "Finished",
};

/**
 * Applies a `training_state` payload (or the `training` block of `model_info`):
 * badge, buttons, speed and the progress bar, all from real completed steps.
 */
function applyTrainingState(state) {
  const restarted = training === null || state.run !== training.run;
  training = state;

  // Progress always mirrors the backend: a reset really does go back to zero,
  // and steps can never run backwards inside one run.
  lastFrameStep = state.step;
  if (restarted) lastFullWeightRequest = 0;

  updateControls();
  renderProgress(state.step, state.total_steps, state);
}

/** Enables only the buttons that make sense in the current state. */
function updateControls() {
  const connected = Boolean(training);
  const state = connected ? training.state : "idle";
  const running = state === "running";

  elements.start.disabled = !connected || running;
  elements.start.textContent = state === "paused" ? "Resume" : "Start";
  elements.pause.disabled = !connected || !running;
  elements.reset.disabled = !connected || state === "idle";
  elements.speedSelect.disabled = !connected;
  if (connected) elements.speedSelect.value = training.speed;
  elements.nextSample.disabled = !model || running;

  elements.controlsNote.textContent = connected
    ? training.message
    : "Waiting for the backend to send the model…";

  if (!connected) setStatus("Connecting…");
  else if (state === "idle" || state === "complete") setStatus(STATE_BADGES[state], "ready");
  else setStatus(STATE_BADGES[state] || "Ready");
}

/** Handles a `training_state` message (start, pause, speed, reset, finish). */
function handleTrainingState(message) {
  applyTrainingState(message);
}

/**
 * Requests the complete weight matrices. Used by the Full view, which cannot be
 * drawn from the selected weights of a training frame.
 */
function requestFullWeights(force = false) {
  const now = performance.now();
  if (!force && now - lastFullWeightRequest < FULL_WEIGHT_REFRESH_MS) return;
  if (sendCommand({ type: "request_weights" })) lastFullWeightRequest = now;
}

/**
 * Handles one `training_step` message: a real optimiser step, with the image it
 * trained, the prediction and loss read before the update, and the weights read
 * after it.
 */
function handleTrainingStep(frame) {
  // Frames from a previous run, or an older frame that arrived late, must never
  // overwrite the newer state. Re-drawing the newest step is harmless.
  if (!training || frame.run !== training.run) return;
  if (frame.step < lastFrameStep) return;
  lastFrameStep = frame.step;

  training.step = frame.step;
  training.epoch = frame.epoch;
  training.total_steps = frame.total_steps;
  training.epochs = frame.epochs;

  const sample = frame.sample || {};
  const width = sample.width || 28;
  const height = sample.height || 28;

  let image;
  let probabilities;
  let activations;
  try {
    image = { width, height, pixels: decodePixels(sample.pixels, width * height) };
    probabilities = decodeFloat32(frame.probabilities);
    activations = (frame.activations || []).map(decodeFloat32);
  } catch (error) {
    console.error("could not read a training_step frame", error);
    return;
  }

  drawSampleImage(elements.sampleCanvas, image);

  elements.sampleIndex.textContent =
    `${formatNumber(sample.position)} / ${formatNumber(sample.total)}`;
  elements.sampleTrue.textContent = String(frame.true_label);
  elements.samplePredicted.textContent = String(frame.predicted_label);
  elements.sampleVerdict.textContent = frame.correct ? "Correct" : "Incorrect";
  elements.sampleVerdict.className = `pill ${frame.correct ? "pill-ok" : "pill-bad"}`;

  updateOutputRows(outputRows, {
    probabilities,
    trueLabel: frame.true_label,
    predictedLabel: frame.predicted_label,
  });

  elements.metricLoss.textContent = Number(frame.loss).toFixed(4);

  if (view.mode === "full") {
    // The frame only carries the selected weights, so the Full view keeps its
    // activations and refreshes its complete weights at a human pace.
    view.updateFrame({ activations });
    requestFullWeights();
  } else {
    view.updateFrame({ weights: decodeVisibleWeights(frame.weights.layers), activations });
    weightsArePartial = true;
  }

  renderProgress(frame.step, frame.total_steps, training);
}

/** Handles a `weights` message: every current weight and bias, as drawn. */
function handleWeights(message) {
  if (!model) return;

  const layers = message.layers || [];
  const architecture = model.architecture;
  const weights = layers.map((layer) => decodeFloat32(layer.weights));
  const shapesMatch =
    weights.length === architecture.length - 1 &&
    weights.every((values, index) => values.length === layers[index].in * layers[index].out);

  if (!shapesMatch) {
    console.error("weights message does not match the model architecture", message);
    return;
  }

  fullWeights = weights;
  weightsArePartial = false;
  view.updateFrame({ weights });

  model.weights_checksum = message.weights_checksum;
  elements.modelBadge.textContent = message.step ? "Real weights · trained" : "Real weights";

  console.info(
    `weights: ${formatNumber(weights.reduce((total, values) => total + values.length, 0))} ` +
      `weights after ${formatNumber(message.step || 0)} step(s)`,
  );
}

let socket = null;
let reconnectAttempts = 0;

function socketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${SOCKET_PATH}`;
}

function scheduleReconnect() {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    setStatus("Backend unavailable", "error");
    elements.connectionNote.textContent =
      "No backend connection. Start it with “python -m backend.server” in the project " +
      "root, then reload this page.";
    return;
  }

  reconnectAttempts += 1;
  setStatus(`Reconnecting ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}…`);
  window.setTimeout(connectSocket, RECONNECT_DELAY_MS);
}

function connectSocket() {
  if (reconnectAttempts === 0) setStatus("Connecting…");
  socket = new WebSocket(socketUrl());

  socket.addEventListener("open", () => {
    reconnectAttempts = 0;
  });

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      console.error("could not read WebSocket message", error);
      return;
    }
    if (message.type === "model_info") handleModelInfo(message);
    else if (message.type === "inference") handleInference(message);
    else if (message.type === "training_step") handleTrainingStep(message);
    else if (message.type === "training_state") handleTrainingState(message);
    else if (message.type === "weights") handleWeights(message);
    else if (message.type === "error") console.error("backend error:", message.message);
  });

  socket.addEventListener("close", () => {
    socket = null;
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    console.warn("WebSocket error — is the backend running?");
  });
}

/** Describes what the canvas is currently showing, including render timing. */
function updateConnectionNote({ mode, connections, milliseconds, cached }) {
  if (!model) {
    elements.connectionNote.textContent =
      "Waiting for the backend to send the model — there are no weights to draw yet.";
    return;
  }

  const total = formatNumber(view.totalConnections);
  const shown = formatNumber(connections);
  const ms = `${Math.round(milliseconds)} ms`;

  if (mode === "partial") {
    elements.connectionNote.textContent =
      `Partial mode: the same ${PARTIAL_TOP_K} connections per neuron from the first step ` +
      `to the last (${shown} of ${total} real weights, drawn in ${ms}). Their colours follow ` +
      `the current weights on one fixed scale of ±${MAX_DISPLAY_WEIGHT}.`;
    return;
  }

  elements.connectionNote.textContent =
    `All connections: ${total} real weights drawn in ${ms}${cached ? " and cached" : ""}, ` +
    `on the same fixed scale of ±${MAX_DISPLAY_WEIGHT} as partial mode. ` +
    `Switch to partial mode to preview the reduced training view.`;
}

function setMode(mode) {
  if (mode === "full" && weightsArePartial && fullWeights.length) {
    // The partial frames only carry the selected weights, so put the newest
    // complete matrices back before drawing every connection, and ask for a
    // fresh snapshot to catch up with the steps that happened meanwhile.
    view.applyWeights(fullWeights);
    weightsArePartial = false;
  }

  view.setMode(mode);
  if (mode === "full") requestFullWeights();

  const partial = mode === "partial";
  elements.modeAll.classList.toggle("is-active", !partial);
  elements.modePartial.classList.toggle("is-active", partial);
  elements.modeAll.setAttribute("aria-pressed", String(!partial));
  elements.modePartial.setAttribute("aria-pressed", String(partial));
}

/** Draws the 28 × 28 grayscale sample; CSS scales it up with crisp pixels. */
function drawSampleImage(canvas, image) {
  const context = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;

  const frame = context.createImageData(image.width, image.height);
  for (let i = 0; i < image.pixels.length; i += 1) {
    const value = Math.round(Math.min(1, Math.max(0, image.pixels[i])) * 255);
    frame.data[i * 4] = value;
    frame.data[i * 4 + 1] = value;
    frame.data[i * 4 + 2] = value;
    frame.data[i * 4 + 3] = 255;
  }
  context.putImageData(frame, 0, 0);
}

function buildOutputRows(container, count) {
  container.textContent = "";
  const rows = [];

  for (let index = 0; index < count; index += 1) {
    const row = document.createElement("div");
    row.className = "output-row";

    const label = document.createElement("span");
    label.className = "output-label";
    label.textContent = String(index);

    const track = document.createElement("span");
    track.className = "output-track";
    const fill = document.createElement("span");
    fill.className = "output-fill";
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "output-value";
    value.textContent = "0.0%";

    row.append(label, track, value);
    container.appendChild(row);
    rows.push({ row, fill, value });
  }

  return rows;
}

function updateOutputRows(rows, state) {
  rows.forEach((entry, index) => {
    const probability = state.probabilities[index] || 0;
    const percent = probability * 100;
    entry.fill.style.width = `${percent.toFixed(1)}%`;
    entry.value.textContent = `${percent.toFixed(1)}%`;
    entry.row.classList.toggle("is-predicted", index === state.predictedLabel);
    entry.row.classList.toggle("is-true", index === state.trueLabel);
  });
}

/**
 * Renders training progress from real completed optimiser steps: the bar, the
 * epoch counter and the current state. `state` is the newest `training_state`,
 * which knows the number of epochs the backend really configured.
 */
function renderProgress(completed, total, state = null) {
  const percent = total > 0 ? (completed / total) * 100 : 0;
  const epochs = state ? state.epochs : 1;
  const started = Boolean(state && state.state !== "idle");

  const epoch = state ? (started ? state.epoch + 1 : 0) : 0;
  const label = !state
    ? "waiting for the backend"
    : { running: "training", paused: "paused", complete: "complete" }[state.state] ||
      "not started";

  elements.metricEpoch.textContent = `${formatNumber(epoch)} / ${formatNumber(epochs)}`;
  elements.metricStep.textContent = `${formatNumber(completed)} / ${formatNumber(total)}`;

  elements.progressFill.style.width = `${percent.toFixed(2)}%`;
  elements.progressText.textContent =
    `${formatNumber(completed)} / ${formatNumber(total)} steps · ` +
    `epoch ${formatNumber(epoch)} / ${formatNumber(epochs)} · ${label}`;
  elements.progressPercent.textContent = `${percent.toFixed(1)}%`;
  elements.progressBar.setAttribute("aria-valuenow", percent.toFixed(1));
}

/** Asks the backend for the next MNIST sample's forward pass. */
function requestNextSample() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  elements.nextSample.disabled = true;
  socket.send(JSON.stringify({ type: "next_sample" }));

  // Never leave the button stuck if the reply is lost.
  window.clearTimeout(nextSampleTimeout);
  nextSampleTimeout = window.setTimeout(() => {
    elements.nextSample.disabled = false;
  }, NEXT_SAMPLE_TIMEOUT_MS);
}

/**
 * Paints the weight legend from the renderer's own colour function, so the
 * swatch can never drift from the lines it describes. The five sampled weights
 * are the stops of the scale and the tick labels printed under the bar.
 */
function paintWeightLegend() {
  const weights = [
    MIN_DISPLAY_WEIGHT,
    MIN_DISPLAY_WEIGHT / 2,
    0,
    MAX_DISPLAY_WEIGHT / 2,
    MAX_DISPLAY_WEIGHT,
  ];
  const stops = weights.map((weight) => {
    const { r, g, b } = weightColour(weight);
    return `rgb(${r}, ${g}, ${b})`;
  });

  elements.weightScaleBar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
}

function init() {
  elements.architectureSummary.textContent = ARCHITECTURE.map((layer) => layer.size).join(" → ");

  paintWeightLegend();

  view.onRender = updateConnectionNote;
  view.render();

  // Empty panels until the first real sample arrives.
  outputRows = buildOutputRows(elements.outputList, ARCHITECTURE[ARCHITECTURE.length - 1].size);
  updateOutputRows(outputRows, { probabilities: [], trueLabel: -1, predictedLabel: -1 });
  elements.metricLoss.textContent = "–";
  elements.metricEpoch.textContent = "–";
  elements.metricStep.textContent = "–";
  elements.progressText.textContent = "Waiting for the model…";
  elements.progressPercent.textContent = "0.0%";
  elements.nextSample.disabled = true;
  updateControls();

  // Ask the backend for the real model and the first real sample. Until it
  // answers, only the neurons are drawn and the badge stays on "Connecting…".
  connectSocket();
}

elements.modeAll.addEventListener("click", () => setMode("full"));
elements.modePartial.addEventListener("click", () => setMode("partial"));
elements.nextSample.addEventListener("click", requestNextSample);

elements.start.addEventListener("click", () => {
  // Start in the reduced view: drawing 109,184 lines twenty times a second
  // would make the browser miss the very frames it is meant to show.
  if (view.mode === "full") setMode("partial");
  sendCommand({ type: "start" });
});

elements.pause.addEventListener("click", () => sendCommand({ type: "pause" }));

elements.reset.addEventListener("click", () => sendCommand({ type: "reset" }));

elements.speedSelect.addEventListener("change", () => {
  sendCommand({ type: "set_speed", speed: elements.speedSelect.value });
});

init();

