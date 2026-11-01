/**
 * main.js — Page state, panel rendering and UI wiring.
 *
 * Milestone 1 only: every value comes from ./mock.js. There is no backend, no
 * WebSocket and no training yet, so the training controls and the speed
 * selector are rendered but disabled.
 *
 * In a later milestone this file gains the WebSocket client; network.js keeps
 * handling all canvas drawing.
 */

import { ARCHITECTURE, NetworkView, PARTIAL_TOP_K } from "./network.js";
import { createMockState } from "./mock.js";

const elements = {
  architectureSummary: document.getElementById("architecture-summary"),
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

const mock = createMockState(ARCHITECTURE);

const view = new NetworkView(elements.networkCanvas, {
  architecture: ARCHITECTURE,
  weights: mock.weights,
});

function formatNumber(value) {
  return value.toLocaleString("en-GB");
}

/** Describes what the canvas is currently showing, including render timing. */
function updateConnectionNote({ mode, connections, milliseconds, cached }) {
  const total = formatNumber(view.totalConnections);
  const shown = formatNumber(connections);
  const ms = `${Math.round(milliseconds)} ms`;

  if (mode === "partial") {
    elements.connectionNote.textContent =
      `Partial mode: the ${PARTIAL_TOP_K} strongest connections by |weight| into each neuron ` +
      `(${shown} of ${total} lines, drawn in ${ms}). This is the reduced view planned for live training.`;
    return;
  }

  elements.connectionNote.textContent =
    `All connections: ${total} lines drawn in ${ms}${cached ? " and cached" : ""}. ` +
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
}

elements.modeAll.addEventListener("click", () => setMode("full"));
elements.modePartial.addEventListener("click", () => setMode("partial"));

init();

