/**
 * network.js — Canvas rendering of the MNIST MLP.
 *
 * The architecture mirrors the existing PyTorch model in `training/training.py`:
 *
 *   layer1 = nn.Linear(784, 128)  -> torch.relu
 *   layer2 = nn.Linear(128, 64)   -> torch.relu
 *   layer3 = nn.Linear(64, 10)    -> logits (nn.CrossEntropyLoss)
 *
 * That is 986 neurons and 109,184 weight connections, so everything is drawn on
 * a single <canvas> (never one DOM/SVG element per line). The "all" view is
 * rendered once into an offscreen canvas and re-used until the weights or the
 * canvas size change.
 *
 * This module only renders the data it is handed. It has no knowledge of where
 * that data comes from: `main.js` receives the real weights and the real
 * per-layer activations from the FastAPI backend over a WebSocket and hands
 * them over with `setModel()` and `setActivations()`.
 */

export const ARCHITECTURE = [
  { name: "Input", size: 784 },
  { name: "Hidden 1", size: 128 },
  { name: "Hidden 2", size: 64 },
  { name: "Output", size: 10 },
];

/** Number of strongest connections kept per receiving neuron in partial mode. */
export const PARTIAL_TOP_K = 5;

// Positive and negative weights use two contrasting hues; alpha and line width
// carry the magnitude. Neuron fill is deliberately a separate encoding: teal,
// brighter for a stronger activation, never the sign of a weight.
const POSITIVE_COLOUR = { r: 90, g: 169, b: 255 };
const NEGATIVE_COLOUR = { r: 255, g: 122, b: 69 };
const ACTIVATION_COLOUR = { r: 94, g: 234, b: 212 };

const INTENSITY_BINS = 8;
const ALPHA_MIN = 0.04;
const ALPHA_MAX = 0.42;

/**
 * Total number of weight connections between consecutive layers.
 * @param {Array<{ size: number }>} architecture
 * @returns {number}
 */
export function countConnections(architecture) {
  let total = 0;
  for (let layer = 0; layer < architecture.length - 1; layer += 1) {
    total += architecture[layer].size * architecture[layer + 1].size;
  }
  return total;
}

function toRgba({ r, g, b }, alpha) {
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

function maxAbs(values) {
  let maximum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const magnitude = Math.abs(values[i]);
    if (magnitude > maximum) maximum = magnitude;
  }
  return maximum;
}

/**
 * Indices of the `k` largest-magnitude values inside one row of a weight matrix.
 * The rule is deterministic: identical weights always produce identical lines.
 */
function topIndicesByMagnitude(values, offset, count, k) {
  const scored = new Array(count);
  for (let i = 0; i < count; i += 1) {
    scored[i] = { index: i, magnitude: Math.abs(values[offset + i]) };
  }
  scored.sort((a, b) => b.magnitude - a.magnitude);

  const limit = Math.min(k, count);
  const indices = new Array(limit);
  for (let i = 0; i < limit; i += 1) indices[i] = scored[i].index;
  return indices;
}

/**
 * Dense layers overlap heavily when every line is drawn. Their lines are drawn
 * slightly fainter so the layer structure stays readable; within a layer the
 * intensity still encodes |weight| relative to that layer's strongest weight.
 */
function layerAlphaScale(connectionCount) {
  const scale = Math.sqrt(20000 / Math.max(1, connectionCount));
  return Math.min(1, Math.max(0.45, scale));
}

export class NetworkView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{ architecture?: Array<{ name: string, size: number }>, weights?: Float32Array[] }} options
   *   `weights[layer]` is the [out][in] flattened matrix between layer and layer + 1.
   */
  constructor(canvas, { architecture = ARCHITECTURE, weights = [] } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.architecture = architecture;

    this.mode = "full";
    this.weights = [];
    this.selection = [];
    this.layerMaxAbs = [];
    this.activations = [];
    this.activationMax = [];
    this.totalConnections = countConnections(architecture);

    this.layout = [];
    this.width = 0;
    this.height = 0;
    this.dpr = 1;

    this.cache = document.createElement("canvas");
    this.cacheCtx = this.cache.getContext("2d");
    this.cacheValid = false;
    this.fullDrawMs = 0;

    /** @type {{ mode: string, connections: number, milliseconds: number, cached: boolean } | null} */
    this.lastFrame = null;
    /** Optional callback invoked after every render (used for the info line). */
    this.onRender = null;

    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas.parentElement || canvas);

    this.setWeights(weights);
    this.resize();
  }

  setWeights(weights) {
    this.weights = weights || [];
    this.layerMaxAbs = this.weights.map((weight) => maxAbs(weight));
    this.selection = this.weights.map((weight, layer) => {
      const inSize = this.architecture[layer].size;
      const outSize = this.architecture[layer + 1].size;
      const rows = new Array(outSize);
      for (let out = 0; out < outSize; out += 1) {
        rows[out] = topIndicesByMagnitude(weight, out * inSize, inSize, PARTIAL_TOP_K);
      }
      return rows;
    });
    this.cacheValid = false;
    this.render();
  }

  /**
   * Replaces the layer sizes and the weights together, as sent by the backend's
   * `model_info` message. Layer captions then show the sizes the model really
   * has instead of the default ARCHITECTURE.
   */
  setModel({ architecture, weights }) {
    if (architecture && architecture.length >= 2) {
      this.architecture = architecture;
      this.totalConnections = countConnections(architecture);
      // resize() only recomputes the layout when the canvas changes size, so a
      // new architecture has to redo it here.
      if (this.width > 0 && this.height > 0) this.computeLayout();
    }
    this.setWeights(weights);
  }

  /**
   * One activation array per layer, in layer order: the normalised pixels, the
   * ReLU outputs of the hidden layers and the output probabilities. Neuron fill
   * brightness shows activation while line colour keeps showing weight sign.
   * @param {Float32Array[]} activations
   */
  setActivations(activations) {
    this.activations = activations || [];
    this.activationMax = this.activations.map((values) => maxAbs(values));
    // Neurons are painted on top of the cached connections, so the expensive
    // full-network bitmap stays valid and only this cheap overlay is redrawn.
    this.render();
  }

  /** @param {"full" | "partial"} mode */
  setMode(mode) {
    const next = mode === "partial" ? "partial" : "full";
    if (next === this.mode) return;
    this.mode = next;
    this.render();
  }

  visibleConnectionCount() {
    if (this.mode !== "partial") return this.totalConnections;
    let total = 0;
    for (const rows of this.selection) {
      for (const row of rows) total += row.length;
    }
    return total;
  }

  resize() {
    const host = this.canvas.parentElement || this.canvas;
    const rect = host.getBoundingClientRect();
    const width = Math.max(320, Math.round(rect.width));
    const height = Math.max(360, Math.round(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (width === this.width && height === this.height && dpr === this.dpr) return;

    this.width = width;
    this.height = height;
    this.dpr = dpr;

    for (const target of [this.canvas, this.cache]) {
      target.width = Math.round(width * dpr);
      target.height = Math.round(height * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cacheCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.computeLayout();
    this.cacheValid = false;
    this.render();
  }

  computeLayout() {
    const padX = this.width < 620 ? 34 : 62;
    const padTop = 26;
    const padBottom = 48;
    const usableWidth = Math.max(1, this.width - padX * 2);
    const usableHeight = Math.max(1, this.height - padTop - padBottom);
    const layerCount = this.architecture.length;

    this.layout = this.architecture.map((layer, index) => {
      const x = layerCount === 1 ? this.width / 2 : padX + (usableWidth * index) / (layerCount - 1);
      const step = usableHeight / layer.size;
      const ys = new Float32Array(layer.size);
      for (let i = 0; i < layer.size; i += 1) {
        ys[i] = padTop + step * (i + 0.5);
      }
      return { x, ys, step, size: layer.size };
    });
  }

  render() {
    if (!this.width || !this.height || this.layout.length === 0) return;

    let milliseconds = 0;
    let cached = false;

    if (this.mode === "full") {
      if (!this.cacheValid) {
        this.cacheCtx.clearRect(0, 0, this.width, this.height);
        this.fullDrawMs = this.drawConnectionsFull(this.cacheCtx);
        this.cacheValid = true;
      } else {
        cached = true;
      }

      // Blit the cached connection bitmap 1:1 in device pixels, restore the CSS
      // scale, then draw the (changeable) neurons and captions on top. Only the
      // 109k connections are cached; activations can update without redrawing
      // them.
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(this.cache, 0, 0);
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.drawNeurons(this.ctx);
      this.drawLayerCaptions(this.ctx);
      milliseconds = this.fullDrawMs;
    } else {
      this.ctx.clearRect(0, 0, this.width, this.height);
      milliseconds = this.drawConnectionsPartial(this.ctx);
      this.drawNeurons(this.ctx);
      this.drawLayerCaptions(this.ctx);
    }

    this.lastFrame = {
      mode: this.mode,
      connections: this.visibleConnectionCount(),
      milliseconds,
      cached,
    };
    if (this.onRender) this.onRender(this.lastFrame);
  }

  drawConnectionsFull(ctx) {
    const started = performance.now();
    ctx.lineCap = "butt";
    ctx.lineWidth = 1;

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const from = this.layout[layer];
      const to = this.layout[layer + 1];
      const weight = this.weights[layer];
      if (!from || !to || !weight) continue;
      const inSize = from.size;
      if (weight.length !== inSize * to.size) continue;

      const scale = this.layerMaxAbs[layer] > 0 ? 1 / this.layerMaxAbs[layer] : 0;
      const alphaScale = layerAlphaScale(weight.length);

      // One path per (sign, intensity) pair: 16 stroke calls per layer instead
      // of 100,352 individual line draws.
      const paths = [];
      for (let i = 0; i < INTENSITY_BINS * 2; i += 1) paths.push(new Path2D());

      for (let out = 0; out < to.size; out += 1) {
        const y2 = to.ys[out];
        const rowOffset = out * inSize;
        for (let input = 0; input < inSize; input += 1) {
          const w = weight[rowOffset + input];
          const magnitude = Math.min(1, Math.abs(w) * scale);
          const bin = Math.min(INTENSITY_BINS - 1, Math.floor(magnitude * INTENSITY_BINS));
          const path = paths[(w < 0 ? INTENSITY_BINS : 0) + bin];
          path.moveTo(from.x, from.ys[input]);
          path.lineTo(to.x, y2);
        }
      }

      for (let bin = 0; bin < INTENSITY_BINS; bin += 1) {
        const t = (bin + 0.5) / INTENSITY_BINS;
        const alpha = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * Math.pow(t, 1.4)) * alphaScale;
        ctx.strokeStyle = toRgba(POSITIVE_COLOUR, alpha);
        ctx.stroke(paths[bin]);
        ctx.strokeStyle = toRgba(NEGATIVE_COLOUR, alpha);
        ctx.stroke(paths[INTENSITY_BINS + bin]);
      }
    }

    return performance.now() - started;
  }

  drawConnectionsPartial(ctx) {
    const started = performance.now();
    ctx.lineCap = "round";

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const from = this.layout[layer];
      const to = this.layout[layer + 1];
      const weight = this.weights[layer];
      const rows = this.selection[layer];
      if (!from || !to || !weight || !rows) continue;
      const inSize = from.size;
      if (weight.length !== inSize * to.size) continue;

      const scale = this.layerMaxAbs[layer] > 0 ? 1 / this.layerMaxAbs[layer] : 0;

      for (let out = 0; out < to.size; out += 1) {
        const y2 = to.ys[out];
        const rowOffset = out * inSize;
        const indices = rows[out];
        for (let k = 0; k < indices.length; k += 1) {
          const input = indices[k];
          const w = weight[rowOffset + input];
          const magnitude = Math.min(1, Math.abs(w) * scale);
          const alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * Math.pow(magnitude, 1.2);
          ctx.strokeStyle = toRgba(w < 0 ? NEGATIVE_COLOUR : POSITIVE_COLOUR, alpha);
          ctx.lineWidth = 0.8 + magnitude * 1.6;
          ctx.beginPath();
          ctx.moveTo(from.x, from.ys[input]);
          ctx.lineTo(to.x, y2);
          ctx.stroke();
        }
      }
    }

    return performance.now() - started;
  }

  drawNeurons(ctx) {
    // Neutral slate dot keeps the layer structure visible; the activation tint
    // is painted on top, so an inactive neuron is simply not tinted.
    for (let layer = 0; layer < this.architecture.length; layer += 1) {
      const { x, ys, step, size } = this.layout[layer];
      const radius = Math.max(0.7, Math.min(5, step * 0.34));
      const values = this.activations[layer];
      const scaled = values && values.length === size ? values : null;
      const scale = this.activationMax[layer] > 0 ? 1 / this.activationMax[layer] : 0;

      for (let i = 0; i < size; i += 1) {
        ctx.beginPath();
        ctx.arc(x, ys[i], radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(203, 213, 225, 0.85)";
        ctx.fill();

        if (!scaled) continue;
        const magnitude = Math.min(1, Math.abs(scaled[i]) * scale);
        if (magnitude <= 0.02) continue;

        ctx.beginPath();
        ctx.arc(x, ys[i], radius, 0, Math.PI * 2);
        ctx.fillStyle = toRgba(ACTIVATION_COLOUR, 0.12 + 0.88 * Math.pow(magnitude, 0.6));
        ctx.fill();
      }
    }

    // The 10 output neurons are big enough to carry their class label.
    const outputLayer = this.layout[this.layout.length - 1];
    ctx.font = '500 11px system-ui, "Segoe UI", sans-serif';
    ctx.fillStyle = "rgba(147, 164, 192, 0.95)";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (let i = 0; i < outputLayer.size; i += 1) {
      ctx.fillText(String(i), outputLayer.x + 11, outputLayer.ys[i]);
    }
  }

  drawLayerCaptions(ctx) {
    const baseline = this.height - 14;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";

    for (let layer = 0; layer < this.architecture.length; layer += 1) {
      const { x } = this.layout[layer];
      const meta = this.architecture[layer];
      ctx.font = '600 12px system-ui, "Segoe UI", sans-serif';
      ctx.fillStyle = "rgba(230, 237, 247, 0.92)";
      ctx.fillText(meta.name, x, baseline - 15);
      ctx.font = '500 11px system-ui, "Segoe UI", sans-serif';
      ctx.fillStyle = "rgba(147, 164, 192, 0.95)";
      ctx.fillText(`${meta.size} neurons`, x, baseline);
    }
  }
}

