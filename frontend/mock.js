/**
 * mock.js — Local placeholder data for Milestone 1.
 *
 * NOTHING in this file is real training data.
 *
 * - The weights imitate PyTorch's default nn.Linear initialisation
 *   (uniform in ±1/sqrt(fan_in)) so the overview looks like an untrained model.
 * - The "digit" is drawn with a browser font so the sample panel can be
 *   reviewed before the backend exists.
 * - Prediction, loss and progress are fixed sample values.
 *
 * Real model weights arrive in Milestone 2 and real MNIST samples in
 * Milestone 3, both over the WebSocket. This module then disappears.
 */

const MOCK_SEED = 20260101;

/** MNIST training set size — one optimiser step per sample (batch_size = 1). */
const MOCK_TOTAL_STEPS = 60000;

/** Mock softmax output: class 3 is predicted, class 3 is also the true label. */
const MOCK_PROBABILITIES = [0.03, 0.01, 0.14, 0.52, 0.02, 0.09, 0.04, 0.05, 0.07, 0.03];

/** Deterministic PRNG so reloading the page always shows the same mock model. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * One Float32Array per layer, flattened as [out][in] — the same ordering as
 * `layer1.weight` in PyTorch.
 */
function createMockWeights(architecture, random) {
  const weights = [];
  for (let layer = 0; layer < architecture.length - 1; layer += 1) {
    const inSize = architecture[layer].size;
    const outSize = architecture[layer + 1].size;
    const bound = 1 / Math.sqrt(inSize);
    const values = new Float32Array(inSize * outSize);
    for (let i = 0; i < values.length; i += 1) {
      values[i] = (random() * 2 - 1) * bound;
    }
    weights.push(values);
  }
  return weights;
}

/**
 * A 28 × 28 grayscale digit rendered from a font, mirroring the tensor shape
 * produced by torchvision's ToTensor() (values in 0..1).
 */
function createMockDigitImage(size = 28) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#ffffff";
  ctx.font = `700 ${Math.round(size * 0.84)}px "DejaVu Sans", "Segoe UI", Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("3", size / 2, size / 2 + 1);

  const { data } = ctx.getImageData(0, 0, size, size);
  const pixels = new Float32Array(size * size);
  for (let i = 0; i < pixels.length; i += 1) {
    pixels[i] = data[i * 4] / 255;
  }
  return { width: size, height: size, pixels };
}

/**
 * @param {Array<{ name: string, size: number }>} architecture
 */
export function createMockState(architecture) {
  const random = mulberry32(MOCK_SEED);
  const weights = createMockWeights(architecture, random);

  const probabilities = MOCK_PROBABILITIES.slice();
  let predictedLabel = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[predictedLabel]) predictedLabel = i;
  }

  return {
    weights,
    image: createMockDigitImage(28),
    trueLabel: 3,
    predictedLabel,
    probabilities,
    loss: 0.7421,
    epoch: 1,
    step: 8400,
    totalSteps: MOCK_TOTAL_STEPS,
  };
}
