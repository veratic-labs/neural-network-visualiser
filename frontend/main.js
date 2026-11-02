/**
 * main.js — Page state, WebSocket client, panel rendering and UI wiring.
 *
 * The network weights come from the backend: when the page connects, the server
 * sends the real PyTorch model as a `model_info` message and network.js draws
 * exactly those values. No training runs yet, so the weights stay at the
 * model's initial values and no further messages follow. network.js keeps
 * handling all canvas drawing.
 *
 * The sample image, prediction, loss and progress panels are still local
 * placeholders from ./mock.js and are labelled "mock" in the UI.
 */

import { ARCHITECTURE, NetworkView, PARTIAL_TOP_K, countConnections } from "./network.js";
import { createMockState } from "./mock.js";

/** WebSocket endpoint exposed by backend/server.py. */
const SOCKET_PATH = "/ws";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

const elements = {
  architectureSummary: document.getElementById("architecture-summary"),
  batchSize: document.getElementById("batch-size"),
  statusBadge: document.getElementById("status-badge"),
  modelBadge: document.getElementById("model-badge"),
  networkCanvas: document.getElementById("network-canvas"),
  connectionNote: document.getElementById("connection-note"),
  modeAll: document.getElementById("mode-all"),
  modePartial: document.getElementById("mode-partial"),
  sampleCanvas: document.getElementById("sample-canvas"),
  sampleTrue: document.getElementById("sample-true"),
  samplePredicted: document.getElementById("sample-predicted"),
  sampleVerdict: document.getElementById("sample-verdict"),
  outputList: document.getElementById("output-list"),
  metricLoss: document.getElementById("metric-loss"),
  metricEpoch: document.getElementById("metric-epoch"),
  metricStep: document.getElementById("metric-step"),
  progressBar: document.getElementById("progress-bar"),
  progressFill: document.getElementById("progress-fill"),
  progressText: document.getElementById("progress-text"),
  progressPercent: document.getElementById("progress-percent"),
};

const mock = createMockState();

/** The `model_info` payload from the backend, or null until it arrives. */
let model = null;

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
 * Decodes the backend's base64 payload into float32 values. The server sends
 * raw little-endian bytes in row-major [out][in] order — the same layout as
 * `layer.weight` in PyTorch — so this is a straight copy into a Float32Array.
 */
function decodeFloat32(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/** Renders the real model the backend just sent. */
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
  view.setModel({ architecture, weights });

  elements.architectureSummary.textContent = architecture.map((layer) => layer.size).join(" → ");
  if (message.batch_size) elements.batchSize.textContent = String(message.batch_size);
  elements.modelBadge.textContent =
    message.weights_state === "trained" ? "Real weights" : "Real weights · untrained";
  setStatus("Ready", "ready");

  console.info(
    `model_info: ${architecture.map((layer) => layer.size).join("→")}, ` +
      `${formatNumber(connectionCount)} weights, sha256 ${String(message.weights_checksum).slice(0, 16)} ` +
      `from ${message.weights_source} (${message.weights_state})`,
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
      `Partial mode: the ${PARTIAL_TOP_K} strongest connections by |weight| into each neuron ` +
      `(${shown} of ${total} real weights, drawn in ${ms}). This is the reduced view used during training.`;
    return;
  }

  elements.connectionNote.textContent =
    `All connections: ${total} real weights drawn in ${ms}${cached ? " and cached" : ""}. ` +
    `Switch to partial mode to preview the reduced training view.`;
}

function setMode(mode) {
  view.setMode(mode);
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

function renderMetrics() {
  const { loss, epoch, step, totalSteps } = mock;
  const percent = totalSteps > 0 ? (step / totalSteps) * 100 : 0;

  elements.metricLoss.textContent = loss.toFixed(4);
  elements.metricEpoch.textContent = String(epoch);
  elements.metricStep.textContent = `${formatNumber(step)} / ${formatNumber(totalSteps)}`;

  elements.progressFill.style.width = `${percent.toFixed(2)}%`;
  elements.progressText.textContent =
    `${formatNumber(step)} / ${formatNumber(totalSteps)} steps · epoch ${epoch}`;
  elements.progressPercent.textContent = `${percent.toFixed(1)}%`;
  elements.progressBar.setAttribute("aria-valuenow", percent.toFixed(1));
}

function init() {
  elements.architectureSummary.textContent = ARCHITECTURE.map((layer) => layer.size).join(" → ");

  view.onRender = updateConnectionNote;
  view.render();

  drawSampleImage(elements.sampleCanvas, mock.image);
  elements.sampleTrue.textContent = String(mock.trueLabel);
  elements.samplePredicted.textContent = String(mock.predictedLabel);
  const correct = mock.trueLabel === mock.predictedLabel;
  elements.sampleVerdict.textContent = correct ? "Correct" : "Incorrect";
  elements.sampleVerdict.className = `pill ${correct ? "pill-ok" : "pill-bad"}`;

  updateOutputRows(buildOutputRows(elements.outputList, mock.probabilities.length), mock);
  renderMetrics();

  // Ask the backend for the real model. Until it answers, only the neurons are
  // drawn and the status badge stays on "Connecting…".
  connectSocket();
}

elements.modeAll.addEventListener("click", () => setMode("full"));
elements.modePartial.addEventListener("click", () => setMode("partial"));

init();

