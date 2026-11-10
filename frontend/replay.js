/**
 * replay.js — Milestone 5B: the offline replay page.
 *
 * This page shows a recorded run instead of a live one. It reads the static
 * files of one recording directory through `recording.js` (manifest, initial
 * weights, biases, fixed Partial indices, checkpoint chunks), draws the network
 * with the same `NetworkView` as `main.js`, and writes the image, facts, bars,
 * metrics and progress with the same `panels.js`. Nothing draws twice and
 * nothing here trains, sends or invents a value.
 *
 * One rule runs through the whole file: the checkpoint on screen is always a
 * recorded one. The timeline holds checkpoint indices, Previous and Next move by
 * exactly one checkpoint, playback advances by one checkpoint per tick, and a
 * decode that returns a different step than the manifest promised is treated as
 * an error rather than drawn. Values are never interpolated, smoothed or scaled.
 *
 * Capture convention (plan.md section 7.3): the image, prediction,
 * probabilities, activations and loss of a checkpoint come from the forward pass
 * *before* that step's optimiser update; the 1,010 Partial weights were read
 * *after* it. The page states this next to the sample it describes.
 */

import { MAX_DISPLAY_WEIGHT, NetworkView } from "./network.js";
import {
  buildOutputRows,
  formatNumber,
  renderProgressPanel,
  setStatus,
  updateLossMetric,
  updateOutputRows,
  updateSamplePanel,
} from "./panels.js";
import { Recording, RecordingError } from "./recording.js";

/**
 * The recording to replay, relative to this page — never root-absolute, so the
 * same files work from a local static server and from a subdirectory such as a
 * GitHub Pages project URL. `replay.html` sets it in one place.
 */
const RECORDING_URL = document.body.dataset.recording || "../recordings/mnist-epoch1/";

/**
 * How long one recorded checkpoint stays on screen, in milliseconds. A speed
 * setting changes only this dwell time: never the checkpoints, their order, the
 * step labels or any value.
 */
const SPEEDS = Object.freeze({ slow: 1000, normal: 400, fast: 120 });

const elements = {
  architectureSummary: document.getElementById("architecture-summary"),
  runName: document.getElementById("run-name"),
  statusBadge: document.getElementById("status-badge"),
  modelBadge: document.getElementById("model-badge"),
  networkCanvas: document.getElementById("network-canvas"),
  connectionNote: document.getElementById("connection-note"),
  modeAll: document.getElementById("mode-all"),
  modePartial: document.getElementById("mode-partial"),
  sampleCanvas: document.getElementById("sample-canvas"),
  sampleStep: document.getElementById("sample-step"),
  sampleIndex: document.getElementById("sample-index"),
  sampleTrue: document.getElementById("sample-true"),
  samplePredicted: document.getElementById("sample-predicted"),
  sampleVerdict: document.getElementById("sample-verdict"),
  outputList: document.getElementById("output-list"),
  metricLoss: document.getElementById("metric-loss"),
  metricEpoch: document.getElementById("metric-epoch"),
  metricStep: document.getElementById("metric-step"),
  recordingCheckpoints: document.getElementById("recording-checkpoints"),
  recordingSteps: document.getElementById("recording-steps"),
  recordingCadence: document.getElementById("recording-cadence"),
  dataNote: document.getElementById("data-note"),
  timeline: document.getElementById("timeline"),
  timelineReadout: document.getElementById("timeline-readout"),
  timelineFirst: document.getElementById("timeline-first"),
  timelineMiddle: document.getElementById("timeline-middle"),
  timelineLast: document.getElementById("timeline-last"),
  play: document.getElementById("btn-play"),
  pause: document.getElementById("btn-pause"),
  prev: document.getElementById("btn-prev"),
  next: document.getElementById("btn-next"),
  speedSelect: document.getElementById("speed-select"),
  controlsNote: document.getElementById("controls-note"),
  progressBar: document.getElementById("progress-bar"),
  progressFill: document.getElementById("progress-fill"),
  progressText: document.getElementById("progress-text"),
  progressPercent: document.getElementById("progress-percent"),
};

/** The open recording, or null until its manifest has loaded and validated. */
let recording = null;
/** The shared canvas renderer: the same class `main.js` uses. */
let view = null;
/** Output bar elements, built once from the recorded architecture. */
let outputRows = [];
/** The Step 0 weights from model.bin: the only full weights ever recorded. */
let initialWeights = [];
/** The fixed Partial selection from partial-indices.bin, never recomputed. */
let partialSelection = [];
/** The newest checkpoint's Partial weights, spread into the renderer's matrices. */
let checkpointWeights = null;
/** Index of the checkpoint on screen, or null before the first one is drawn. */
let currentIndex = null;
/** Bumped on every seek so a slow fetch can never overwrite a newer selection. */
let seekToken = 0;
let playing = false;
let playTimer = 0;
let speed = "normal";

/** Index of the last recorded checkpoint. */
function lastIndex() {
  return recording ? recording.checkpointCount - 1 : 0;
}

/** The recording's directory name, for the header ("mnist-epoch1"). */
function runName(url) {
  const trimmed = url.replace(/\/+$/, "");
  const name = trimmed.slice(trimmed.lastIndexOf("/") + 1);
  return name || trimmed || "recording";
}

/**
 * The recorded 28 × 28 image as the renderer wants it: values in 0..1, the same
 * range `torchvision.transforms.ToTensor()` produces for the live page.
 */
function sampleImage(checkpoint) {
  const pixels = new Float32Array(checkpoint.image.length);
  for (let i = 0; i < pixels.length; i += 1) pixels[i] = checkpoint.image[i] / 255;
  return { width: recording.imageWidth, height: recording.imageHeight, pixels };
}

/**
 * One activation array per layer, in layer order: the input pixels followed by
 * the recorded A1, A2 and output activations. The recording stores only the
 * three computed layers; the input layer is the image divided by 255, exactly
 * as the manifest's `activations.input` note and the live page derive it.
 */
function layerActivations(pixels, recorded) {
  const layers = [pixels];
  let offset = 0;
  for (const size of recording.activationSizes) {
    if (offset + size > recorded.length) {
      throw new RecordingError("the recorded activations are shorter than their layer sizes");
    }
    layers.push(recorded.subarray(offset, offset + size));
    offset += size;
  }
  if (offset !== recorded.length) {
    throw new RecordingError(
      `the checkpoint holds ${recorded.length} activations, the manifest describes ${offset}`,
    );
  }
  return layers;
}

/**
 * Spreads one checkpoint's recorded Partial weights back into dense per-layer
 * matrices, which is the shape `NetworkView` reads: the recorded connections get
 * their recorded weight and every other entry stays zero, so the Partial view
 * draws exactly the 1,010 lines and nothing else. The selection itself is never
 * recomputed here — it is the fixed one from `partial-indices.bin`.
 */
function densePartialWeights(partialWeights, selection) {
  const matrices = recording.layers.map((layer) => new Float32Array(layer.in * layer.out));
  let cursor = 0;

  for (let layer = 0; layer < matrices.length; layer += 1) {
    const matrix = matrices[layer];
    const inputs = recording.layers[layer].in;
    const rows = selection[layer];
    for (let out = 0; out < rows.length; out += 1) {
      const indices = rows[out];
      for (let k = 0; k < indices.length; k += 1) {
        matrix[out * inputs + indices[k]] = partialWeights[cursor];
        cursor += 1;
      }
    }
  }

  if (cursor !== partialWeights.length || cursor !== recording.partial.count) {
    throw new RecordingError(
      `scattered ${cursor} Partial weights, the manifest records ${recording.partial.count}`,
    );
  }
  return matrices;
}

/** Describes what the canvas is showing, including render timing. */
function updateConnectionNote({ mode, connections, milliseconds, cached }) {
  if (!recording) {
    elements.connectionNote.textContent = "Loading the recorded network…";
    return;
  }

  const ms = `${Math.round(milliseconds)} ms`;
  if (mode === "partial") {
    elements.connectionNote.textContent =
      `Partial mode: the same ${formatNumber(recording.partial.count)} connections recorded at ` +
      `Step 0, recoloured from the selected checkpoint's own weights (${formatNumber(connections)} ` +
      `lines drawn in ${ms}) on the fixed ±${MAX_DISPLAY_WEIGHT} scale. The selection never changes.`;
    return;
  }

  elements.connectionNote.textContent =
    `All ${formatNumber(view.totalConnections)} connections from the Step 0 initial weights ` +
    `(${ms}${cached ? ", cached" : ""}), on the same fixed ±${MAX_DISPLAY_WEIGHT} scale. ` +
    `Later full-network weights were not recorded, so this view stays still; Partial mode follows ` +
    `the selected checkpoint.`;
}

/** Enables exactly the controls that can act from the current position. */
function updateControls() {
  const ready = Boolean(recording) && currentIndex !== null;
  const atStart = ready && currentIndex <= 0;
  const atEnd = ready && currentIndex >= lastIndex();

  elements.timeline.disabled = !ready;
  elements.play.disabled = !ready || playing || atEnd;
  elements.pause.disabled = !ready || !playing;
  elements.prev.disabled = !ready || atStart;
  elements.next.disabled = !ready || atEnd;
  elements.speedSelect.disabled = !ready;
}

/** The one-line state under the controls. */
function describeState(step, index) {
  const position =
    `step ${formatNumber(step)} · checkpoint ${formatNumber(index + 1)} of ` +
    `${formatNumber(recording.checkpointCount)}`;
  if (playing) return `Playing · one recorded checkpoint every ${SPEEDS[speed]} ms · ${position}`;
  if (index === 0) return `Initial model · ${position} · press Play or Next to advance`;
  return (
    `Paused at ${position} · Previous and Next move one recorded checkpoint ` +
    `(±${formatNumber(recording.cadence)} steps)`
  );
}

/**
 * Puts one recorded checkpoint on screen: the network, the sample, the ten bars,
 * the loss, the timeline and the progress bar. Every value comes from the
 * checkpoint itself; nothing is derived beyond the shape conversions above.
 */
function applyCheckpoint(checkpoint) {
  const hasSample = checkpoint.hasSample === true;
  const image = hasSample ? sampleImage(checkpoint) : null;
  const activations = hasSample ? layerActivations(image.pixels, checkpoint.activations) : null;

  checkpointWeights = densePartialWeights(checkpoint.partialWeights, partialSelection);
  if (view.mode === "partial") {
    view.updateFrame({ weights: checkpointWeights, activations });
  } else if (activations) {
    view.updateFrame({ activations });
  } else {
    view.setActivations([]);
  }

  if (!hasSample) {
    // Step 0 recorded no sample: clear the frame instead of leaving the
    // previously shown digit next to a panel that says there is none.
    const context = elements.sampleCanvas.getContext("2d");
    context.clearRect(0, 0, elements.sampleCanvas.width, elements.sampleCanvas.height);
  }

  updateSamplePanel(elements, {
    image,
    position: hasSample ? checkpoint.samplePosition : null,
    total: hasSample ? recording.trainSamples : null,
    trueLabel: hasSample ? checkpoint.trueLabel : null,
    predictedLabel: hasSample ? checkpoint.predictedLabel : null,
    correct: hasSample ? checkpoint.predictedLabel === checkpoint.trueLabel : null,
    verdict: "No sample at Step 0 · the recording captured its first forward pass at the next checkpoint",
  });
  elements.sampleStep.textContent = `step ${formatNumber(checkpoint.step)}`;

  updateOutputRows(outputRows, {
    probabilities: hasSample ? checkpoint.probabilities : [],
    trueLabel: hasSample ? checkpoint.trueLabel : -1,
    predictedLabel: hasSample ? checkpoint.predictedLabel : -1,
  });
  updateLossMetric(elements, hasSample ? checkpoint.loss : null);

  const step = checkpoint.step;
  const epoch = Math.min(recording.epochs, Math.ceil(step / recording.trainSamples));
  renderProgressPanel(elements, {
    completed: step,
    total: recording.totalSteps,
    epoch,
    epochs: recording.epochs,
    label:
      `replay · checkpoint ${formatNumber(checkpoint.index + 1)} / ` +
      `${formatNumber(recording.checkpointCount)}`,
  });

  currentIndex = checkpoint.index;
  elements.timeline.value = String(currentIndex);
  elements.timeline.setAttribute("aria-valuetext", `step ${formatNumber(step)}`);
  elements.timelineReadout.textContent = `step ${formatNumber(step)}`;
  elements.controlsNote.textContent = describeState(step, currentIndex);
  setStatus(elements, `Replay · step ${formatNumber(step)}`, "ready");
  updateControls();

  // Playback always moves forward, so keep the next checkpoint warm.
  recording.prefetch(currentIndex + 1);
}

/**
 * Shows the checkpoint at `index`, waiting for its chunk if necessary. A newer
 * seek bumps `seekToken`, so a slow chunk that finishes late is dropped instead
 * of replacing the checkpoint the user selected.
 */
async function showCheckpoint(index) {
  if (!recording) return;

  const target = Math.min(Math.max(0, Math.round(index)), lastIndex());
  const token = ++seekToken;

  // A visible sign that a chunk is being read: the chunk may already be cached,
  // in which case this text is replaced before the browser paints it.
  elements.connectionNote.textContent =
    `Fetching checkpoint ${formatNumber(target + 1)} of ${formatNumber(recording.checkpointCount)} ` +
    `(step ${formatNumber(recording.stepOf(target))})…`;

  let checkpoint;
  try {
    checkpoint = await recording.checkpoint(target);
    if (checkpoint.index !== target) {
      throw new RecordingError(
        `checkpoint ${target} reported index ${checkpoint.index}: the manifest and the data disagree`,
      );
    }
  } catch (error) {
    if (token === seekToken) fail(error);
    return;
  }
  if (token !== seekToken) return;

  try {
    applyCheckpoint(checkpoint);
  } catch (error) {
    fail(error);
  }
}

/** Starts stepping forward through the recording, one checkpoint per tick. */
function play() {
  if (!recording || playing || currentIndex === null) return;
  if (currentIndex >= lastIndex()) return;

  playing = true;
  updateControls();
  elements.controlsNote.textContent = describeState(recording.stepOf(currentIndex), currentIndex);
  scheduleNext();
}

/** Queues the next checkpoint at the current speed. */
function scheduleNext() {
  window.clearTimeout(playTimer);
  if (!playing) return;

  playTimer = window.setTimeout(async () => {
    if (!playing) return;
    if (currentIndex >= lastIndex()) {
      pause(`Reached the last recorded checkpoint (step ${formatNumber(recording.lastStep)}).`);
      return;
    }
    await showCheckpoint(currentIndex + 1);
    if (playing) scheduleNext();
  }, SPEEDS[speed] || SPEEDS.normal);
}

/** Stops playback without moving; the current checkpoint stays on screen. */
function pause(message = "") {
  playing = false;
  window.clearTimeout(playTimer);
  if (!recording) return;

  if (message) {
    elements.controlsNote.textContent = message;
  } else if (currentIndex !== null) {
    elements.controlsNote.textContent = describeState(recording.stepOf(currentIndex), currentIndex);
  }
  updateControls();
}

/** Switches between the static Step 0 weights and the recorded Partial weights. */
function setMode(mode) {
  const partial = mode === "partial";

  if (view && view.mode !== mode) {
    // Only the Partial weights were recorded after Step 0, so Full mode always
    // falls back to the initial matrices from model.bin.
    view.applyWeights(partial && checkpointWeights ? checkpointWeights : initialWeights);
    view.setMode(mode);
  }

  elements.modeAll.classList.toggle("is-active", !partial);
  elements.modePartial.classList.toggle("is-active", partial);
  elements.modeAll.setAttribute("aria-pressed", String(!partial));
  elements.modePartial.setAttribute("aria-pressed", String(partial));
}

/**
 * Stops everything and says plainly that the recording could not be read. The
 * message is the specific one from `recording.js` — a missing manifest, a bad
 * record size, a step that disagrees with the manifest — never a generic
 * "something went wrong", and never a silently empty diagram.
 */
function fail(error) {
  playing = false;
  window.clearTimeout(playTimer);

  const message = error instanceof RecordingError ? error.message : `${(error && error.message) || error}`;
  console.error("replay stopped:", error);

  setStatus(elements, "Replay data error", "error");
  elements.dataNote.textContent = `Could not read the recording: ${message}`;
  elements.controlsNote.textContent =
    "Replay is unavailable until the recorded files can be read. Reload the page to try again.";
  elements.timeline.disabled = true;
  elements.play.disabled = true;
  elements.pause.disabled = true;
  elements.prev.disabled = true;
  elements.next.disabled = true;
  elements.speedSelect.disabled = true;
}

/**
 * Reads the manifest, the initial weights, the biases and the fixed Partial
 * indices, then shows the first recorded checkpoint. Every failure path ends in
 * `fail()` with the recording's own message.
 */
async function start() {
  view = new NetworkView(elements.networkCanvas);
  view.onRender = updateConnectionNote;
  elements.runName.textContent = runName(RECORDING_URL);
  elements.modelBadge.textContent = "Replay";
  updateControls();

  elements.dataNote.textContent = `Reading ${RECORDING_URL}manifest.json…`;
  try {
    recording = await Recording.open(RECORDING_URL);
  } catch (error) {
    fail(error);
    return;
  }

  elements.architectureSummary.textContent = recording.architecture
    .map((layer) => formatNumber(layer.size))
    .join(" → ");
  elements.recordingCheckpoints.textContent = formatNumber(recording.checkpointCount);
  elements.recordingSteps.textContent =
    `${formatNumber(recording.firstStep)} – ${formatNumber(recording.lastStep)} ` +
    `of ${formatNumber(recording.totalSteps)}`;
  elements.recordingCadence.textContent =
    recording.cadence === 1 ? "every step" : `every ${formatNumber(recording.cadence)} steps`;
  elements.dataNote.textContent =
    "Manifest read and validated. Loading the initial weights, the biases and the " +
    "fixed Partial indices…";

  try {
    const [weights, biases, selection] = await Promise.all([
      recording.loadInitialWeights(),
      recording.loadBiases(),
      recording.loadPartialSelection(),
    ]);
    initialWeights = weights;
    partialSelection = selection;

    view.setModel({
      architecture: recording.architecture,
      weights: initialWeights,
      selection: partialSelection,
    });

    const biasCount = biases.reduce((total, values) => total + values.length, 0);
    const cadence =
      recording.cadence === 1 ? "every step" : `every ${formatNumber(recording.cadence)} steps`;
    elements.dataNote.textContent =
      `${formatNumber(recording.connectionCount)} initial weights, ${formatNumber(biasCount)} ` +
      `biases and ${formatNumber(recording.partial.count)} fixed Partial connections were read ` +
      `from ${RECORDING_URL} together with ${formatNumber(recording.checkpointCount)} checkpoints ` +
      `(${cadence}).`;
  } catch (error) {
    fail(error);
    return;
  }

  outputRows = buildOutputRows(
    elements.outputList,
    recording.architecture[recording.architecture.length - 1].size,
  );
  updateOutputRows(outputRows, { probabilities: [], trueLabel: -1, predictedLabel: -1 });

  const last = lastIndex();
  elements.timeline.max = String(last);
  elements.timeline.value = "0";
  elements.timelineFirst.textContent = `step ${formatNumber(recording.firstStep)}`;
  elements.timelineMiddle.textContent = `step ${formatNumber(recording.stepOf(Math.floor(last / 2)))}`;
  elements.timelineLast.textContent = `step ${formatNumber(recording.lastStep)}`;
  elements.speedSelect.value = speed;

  await showCheckpoint(0);
}

elements.modeAll.addEventListener("click", () => setMode("full"));
elements.modePartial.addEventListener("click", () => setMode("partial"));

elements.play.addEventListener("click", play);
elements.pause.addEventListener("click", () => pause());
elements.prev.addEventListener("click", () => {
  // A manual seek stops playback, so the checkpoint the user picked always stays
  // on screen until they press Play again.
  if (playing) pause();
  void showCheckpoint((currentIndex === null ? 0 : currentIndex) - 1);
});
elements.next.addEventListener("click", () => {
  if (playing) pause();
  void showCheckpoint((currentIndex === null ? 0 : currentIndex) + 1);
});

// Dragging the timeline seeks to that recorded checkpoint. The slider holds
// checkpoint indices, so one arrow key moves one recorded checkpoint.
elements.timeline.addEventListener("input", () => {
  if (!recording) return;
  if (playing) pause();
  void showCheckpoint(Number(elements.timeline.value));
});

elements.speedSelect.addEventListener("change", () => {
  speed = SPEEDS[elements.speedSelect.value] ? elements.speedSelect.value : "normal";
  if (!playing || currentIndex === null) return;

  // Restart the pending tick so the new dwell time applies from now on.
  scheduleNext();
  elements.controlsNote.textContent = describeState(recording.stepOf(currentIndex), currentIndex);
});

// `NetworkView` observes its own container, so no resize handler is needed here.

void start();
