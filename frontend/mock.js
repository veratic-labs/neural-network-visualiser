/**
 * mock.js — Local placeholder values for the panels that are not live yet.
 *
 * NOTHING in this file is real training data, and it contains no weights: the
 * network overview is drawn from the real PyTorch model sent by the backend.
 *
 * - The "digit" is drawn with a browser font so the sample panel can be
 *   reviewed before real MNIST samples arrive.
 * - Prediction, loss and progress are fixed sample values.
 *
 * Real MNIST samples, predictions, losses and progress arrive with training in
 * Milestone 3/4. This module then disappears.
 */

/** MNIST training set size — one optimiser step per sample (batch_size = 1). */
const MOCK_TOTAL_STEPS = 60000;

/** Mock softmax output: class 3 is predicted, class 3 is also the true label. */
const MOCK_PROBABILITIES = [0.03, 0.01, 0.14, 0.52, 0.02, 0.09, 0.04, 0.05, 0.07, 0.03];

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
 * Placeholder panel values. Weights are deliberately absent: the network view
 * is fed by the backend's `model_info` message.
 */
export function createMockState() {
  const probabilities = MOCK_PROBABILITIES.slice();
  let predictedLabel = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[predictedLabel]) predictedLabel = i;
  }

  return {
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
