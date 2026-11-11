/**
 * panels.js — the DOM panels that the live page and the replay page share.
 *
 * `main.js` (live: WebSocket) and `replay.js` (recorded files) show the same
 * panels, so the code that writes them is kept here once instead of being copied
 * into a second page. Both pages use the same element ids and pass their own
 * `elements` map, so each page can only update its own DOM.
 *
 * Two things deliberately do not live here:
 *   - the network canvas itself, which `network.js` draws in both modes;
 *   - the live page's weight legend, which `main.js` paints from
 *     `network.js`'s own `weightColour()`. The replay page shows the same five
 *     stops as a fixed CSS gradient (style.css), checked against the renderer by
 *     tests/test_milestone5b.py.
 *
 * Everything written here is a number the caller already has: this module
 * formats and places values, it never derives, smooths or invents one.
 */

/** Formats a count for the page: 60000 -> "60,000". */
export function formatNumber(value) {
  return value.toLocaleString("en-GB");
}

/** Updates the header status badge: connecting, ready or failed. */
export function setStatus(elements, text, tone) {
  elements.statusBadge.textContent = text;
  elements.statusBadge.classList.toggle("is-ready", tone === "ready");
  elements.statusBadge.classList.toggle("is-error", tone === "error");
}

/** Draws the 28 × 28 grayscale sample; CSS scales it up with crisp pixels. */
export function drawSampleImage(canvas, image) {
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

/**
 * The "Current sample" panel: the image, the sample counter, the true and the
 * predicted digit, and whether they agree.
 *
 * `state.correct` is `null` when there is no sample to show (the live page
 * before the first message, the replay at Step 0, which recorded no sample). The
 * values are then printed as an em dash and `state.verdict` explains why.
 *
 * @param {object} elements page element map
 * @param {{ image?: object|null, position?: number|null, total?: number|null,
 *           trueLabel?: number|null, predictedLabel?: number|null,
 *           correct?: boolean|null, verdict?: string }} state
 */
export function updateSamplePanel(
  elements,
  {
    image = null,
    position = null,
    total = null,
    trueLabel = null,
    predictedLabel = null,
    correct = null,
    verdict = "",
  },
) {
  if (image) drawSampleImage(elements.sampleCanvas, image);

  const hasSample = correct !== null;
  elements.sampleIndex.textContent =
    hasSample && position !== null && total !== null
      ? `${formatNumber(position)} / ${formatNumber(total)}`
      : "–";
  elements.sampleTrue.textContent = hasSample ? String(trueLabel) : "–";
  elements.samplePredicted.textContent = hasSample ? String(predictedLabel) : "–";
  elements.sampleVerdict.textContent = hasSample
    ? correct
      ? "Correct"
      : "Incorrect"
    : verdict || "–";
  elements.sampleVerdict.className = hasSample ? `pill ${correct ? "pill-ok" : "pill-bad"}` : "pill";
}
/** Builds one labelled bar per class once; `updateOutputRows` then fills them. */
export function buildOutputRows(container, count) {
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

/** Paints the 10 softmax bars of one forward pass. */
export function updateOutputRows(rows, state) {
  rows.forEach((entry, index) => {
    const probability = state.probabilities[index] || 0;
    const percent = probability * 100;
    entry.fill.style.width = `${percent.toFixed(1)}%`;
    entry.value.textContent = `${percent.toFixed(1)}%`;
    entry.row.classList.toggle("is-predicted", index === state.predictedLabel);
    entry.row.classList.toggle("is-true", index === state.trueLabel);
  });
}

/** Shows the cross-entropy loss of the displayed sample, or "–" if there is none. */
export function updateLossMetric(elements, loss) {
  const value = Number(loss);
  elements.metricLoss.textContent =
    loss === null || loss === undefined || !Number.isFinite(value) ? "–" : value.toFixed(4);
}

/**
 * The bottom progress bar, the epoch counter and the step counter, from real
 * completed optimiser steps. `label` names the mode, for example "training",
 * "paused" or "replay · checkpoint 1,235 / 6,001".
 *
 * The epoch and step counters belong to the live page's metrics panel; the
 * replay page shows only the progress bar, so those two are written only when
 * the page provides them.
 *
 * @param {object} elements page element map
 * @param {{ completed: number, total: number, epoch: number, epochs: number,
 *           label: string }} state
 */
export function renderProgressPanel(elements, { completed, total, epoch, epochs, label }) {
  const percent = total > 0 ? (completed / total) * 100 : 0;

  if (elements.metricEpoch) {
    elements.metricEpoch.textContent = `${formatNumber(epoch)} / ${formatNumber(epochs)}`;
    elements.metricStep.textContent = `${formatNumber(completed)} / ${formatNumber(total)}`;
  }

  elements.progressFill.style.width = `${percent.toFixed(2)}%`;
  elements.progressText.textContent =
    `${formatNumber(completed)} / ${formatNumber(total)} steps · ` +
    `epoch ${formatNumber(epoch)} / ${formatNumber(epochs)} · ${label}`;
  elements.progressPercent.textContent = `${percent.toFixed(1)}%`;
  elements.progressBar.setAttribute("aria-valuenow", percent.toFixed(1));
}
