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
 * that data comes from: `replay.js` reads the recorded weights and the recorded
 * per-layer activations and hands them over with `setModel()`,
 * `setActivations()` and, once per checkpoint, `updateFrame()`.
 *
 * Every connection is coloured by `weightColour()`: one fixed diverging scale
 * from dark red (negative) through neutral grey (zero) to dark purple
 * (positive), drawn on a light canvas. Because the scale never moves, the same
 * weight always looks the same — in every layer, in both modes and at every
 * training step.
 */

export const ARCHITECTURE = [
  { name: "Input", size: 784 },
  { name: "Hidden 1", size: 128 },
  { name: "Hidden 2", size: 64 },
  { name: "Output", size: 10 },
];

/** Number of connections kept per receiving neuron in partial mode. */
export const PARTIAL_TOP_K = 5;

/**
 * The one weight scale every layer and both modes share, from the most negative
 * to the most positive weight that is still drawn distinctly. A weight is
 * `Math.abs(weight) / MAX_DISPLAY_WEIGHT` of the way to the end of the scale and
 * saturates there instead of stretching it. Because the scale is fixed, an
 * unchanged weight always looks the same, in every layer and at every step, no
 * matter what the other weights do. It is purely visual and has no influence on
 * training.
 */
export const MIN_DISPLAY_WEIGHT = -0.5;
export const MAX_DISPLAY_WEIGHT = 0.5;

/**
 * The diverging weight scale, running from MIN_DISPLAY_WEIGHT to
 * MAX_DISPLAY_WEIGHT: dark red for strongly negative weights, light orange for
 * weakly negative ones, neutral grey at zero, light blue for weakly positive
 * weights and dark purple for strongly positive ones.
 *
 * Colour carries both the sign and the magnitude, so every connection can be
 * drawn at the same, fairly high opacity and even weak weights stay visible on
 * the light canvas. Neuron fill is a deliberately separate encoding: teal,
 * deeper for a stronger activation, never the sign of a weight.
 */
const NEGATIVE_COLOUR = { r: 185, g: 28, b: 28 }; // #b91c1c — strong negative
const NEGATIVE_WEAK_COLOUR = { r: 253, g: 186, b: 116 }; // #fdba74 — weak negative
const ZERO_COLOUR = { r: 203, g: 213, b: 225 }; // #cbd5e1 — no weight
const POSITIVE_WEAK_COLOUR = { r: 147, g: 197, b: 253 }; // #93c5fd — weak positive
const POSITIVE_COLOUR = { r: 109, g: 40, b: 217 }; // #6d28d9 — strong positive
const ACTIVATION_COLOUR = { r: 13, g: 148, b: 136 }; // #0d9488
const NEURON_COLOUR = { r: 148, g: 163, b: 184 }; // #94a3b8

/**
 * The stops of the weight scale. They are evenly spaced along the weight axis
 * (every step is MAX_DISPLAY_WEIGHT / 2), so a plain five-stop gradient over
 * these same colours reproduces the ramp exactly — which is how the replay
 * page's fixed CSS gradient draws the legend.
 */
const WEIGHT_COLOUR_STOPS = [
  { weight: MIN_DISPLAY_WEIGHT, colour: NEGATIVE_COLOUR },
  { weight: MIN_DISPLAY_WEIGHT / 2, colour: NEGATIVE_WEAK_COLOUR },
  { weight: 0, colour: ZERO_COLOUR },
  { weight: MAX_DISPLAY_WEIGHT / 2, colour: POSITIVE_WEAK_COLOUR },
  { weight: MAX_DISPLAY_WEIGHT, colour: POSITIVE_COLOUR },
];

/**
 * Every connection is drawn at this opacity, whatever its weight: magnitude is
 * in the colour, so weak lines do not have to fade into the background to stay
 * unobtrusive.
 */
const CONNECTION_OPACITY = 0.9;

/**
 * How many colours the cached connection styles sample the scale into. Each line
 * still gets the colour of its own weight; the table only avoids building and
 * re-parsing 100,000 CSS strings on every Full repaint.
 */
const COLOUR_STEPS = 256;

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
 * It is used once, when a model is loaded, to derive the initial partial
 * selection; that selection is then fixed for the rest of the run.
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

/** A colour `t` of the way from `from` to `to` (0 = from, 1 = to). */
function mixColour(from, to, t) {
  return {
    r: Math.round(from.r + (to.r - from.r) * t),
    g: Math.round(from.g + (to.g - from.g) * t),
    b: Math.round(from.b + (to.b - from.b) * t),
  };
}

/**
 * A weight as a 0..1 display magnitude on the fixed scale shared by every layer
 * and both modes. Values at or beyond MAX_DISPLAY_WEIGHT saturate instead of
 * stretching the scale, so the same weight always has the same magnitude.
 */
function weightScaleMagnitude(weight) {
  return Math.min(1, Math.abs(weight) / MAX_DISPLAY_WEIGHT);
}

/**
 * The same magnitude as a signed position in -1..1, which is what puts a
 * negative weight on the warm half of the scale and a positive one on the cool
 * half. The magnitude saturates, the sign never does.
 */
function weightScalePosition(weight) {
  const magnitude = weightScaleMagnitude(weight);
  return weight < 0 ? -magnitude : magnitude;
}

/**
 * The colour of a weight on the diverging scale: smooth interpolation between
 * the five stops, with values beyond ±MAX_DISPLAY_WEIGHT clamped to the endpoint
 * colours.
 *
 * This is the only place where a weight becomes a colour, so a given weight
 * always looks the same: no per-layer factor, no normalisation against the other
 * weights, no dependence on the training step. The replay page's legend uses
 * these same stops, and tests/test_milestone5b.py checks the gradient against
 * this function, so the swatches and the lines cannot disagree.
 *
 * @param {number} weight
 * @returns {{ r: number, g: number, b: number }}
 */
export function weightColour(weight) {
  const scaled = weightScalePosition(weight) * MAX_DISPLAY_WEIGHT;

  for (let index = 0; index < WEIGHT_COLOUR_STOPS.length - 1; index += 1) {
    const from = WEIGHT_COLOUR_STOPS[index];
    const to = WEIGHT_COLOUR_STOPS[index + 1];
    if (scaled > to.weight) continue;

    const span = to.weight - from.weight;
    const t = span === 0 ? 0 : (scaled - from.weight) / span;
    return mixColour(from.colour, to.colour, t);
  }

  return WEIGHT_COLOUR_STOPS[WEIGHT_COLOUR_STOPS.length - 1].colour;
}

/**
 * Cached CSS colours for the connections: 256 samples of `weightColour()` at a
 * constant opacity, built once. Every line is drawn individually, so the Full
 * view can overlay 109,184 translucent lines exactly like the Partial view does,
 * without building a colour string for each one.
 */
let connectionStyles = null;

/** The weight at the centre of one colour step. */
function weightAtStep(step) {
  const position = (step + 0.5) / COLOUR_STEPS; // 0..1
  return (position * 2 - 1) * MAX_DISPLAY_WEIGHT;
}

/** The stroke style of a weight: a table lookup, not a change of colour rule. */
function connectionStyle(weight) {
  if (!connectionStyles) {
    connectionStyles = new Array(COLOUR_STEPS);
    for (let step = 0; step < COLOUR_STEPS; step += 1) {
      connectionStyles[step] = toRgba(weightColour(weightAtStep(step)), CONNECTION_OPACITY);
    }
  }

  const position = (weightScalePosition(weight) + 1) / 2; // 0..1, most negative first
  const step = Math.min(COLOUR_STEPS - 1, Math.max(0, Math.floor(position * COLOUR_STEPS)));
  return connectionStyles[step];
}

/**
 * A fallback Partial-view selection: the PARTIAL_TOP_K largest |weight| incoming
 * connections of every neuron, read once from freshly loaded weights.
 *
 * `replay.js` passes the selection recorded in `partial-indices.bin` to
 * `setModel()`, which is what keeps the page on exactly the recorded 1,010
 * indices; this is only used by a caller that has no selection to hand.
 */
function initialSelection(architecture, weights) {
  if (!Array.isArray(weights) || !architecture || architecture.length < 2) return [];
  if (weights.length !== architecture.length - 1) return [];

  return weights.map((values, layer) => {
    const inSize = architecture[layer].size;
    const outSize = architecture[layer + 1].size;
    const rows = new Array(outSize);
    for (let out = 0; out < outSize; out += 1) {
      rows[out] = topIndicesByMagnitude(values, out * inSize, inSize, PARTIAL_TOP_K);
    }
    return rows;
  });
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
    /**
     * The fixed Partial-view connections: one array of input indices per output
     * neuron, per layer. Chosen once by the recorder from the initial weights
     * and stored in the recording, never recomputed.
     */
    this.selection = [];
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
    this.applyWeights(weights);
    this.render();
  }

  /**
   * Replaces the weight matrices without rendering, so a caller that also has
   * new activations can draw both with one render.
   *
   * Nothing about the selection is recomputed here: the Partial view keeps
   * drawing the connections chosen at load, only their values change.
   * @param {Float32Array[]} weights
   */
  applyWeights(weights) {
    this.weights = weights || [];
    this.cacheValid = false;
  }

  /**
   * The fixed Partial-view connection indices, as read from the recording's
   * `partial-indices.bin`: one array of input indices per receiving neuron, per
   * layer.
   * @param {Array<ArrayLike<number>[]>} selection
   */
  setSelection(selection) {
    this.selection = Array.isArray(selection) ? selection : [];
    this.cacheValid = false;
  }

  /**
   * One checkpoint's frame: the newest weights and the activations of the same
   * step, drawn with a single render.
   *
   * In partial mode a checkpoint only carries the values of the fixed
   * selection, so the matrices are sparse (zero elsewhere). The indices
   * themselves never change, so `this.selection` is left untouched: the same
   * 1,010 lines stay on screen and only their colours follow the new weights.
   *
   * @param {{ weights?: Float32Array[], activations?: Float32Array[] }} frame
   */
  updateFrame({ weights = null, activations = null } = {}) {
    const hasWeights = Array.isArray(weights) && weights.length > 0;
    const hasActivations = Array.isArray(activations) && activations.length > 0;

    if (hasWeights) this.applyWeights(weights);
    if (hasActivations) {
      this.activations = activations;
      this.activationMax = activations.map((values) => maxAbs(values));
    }
    if (!hasWeights && !hasActivations) return;

    this.render();
  }

  /**
   * Replaces the layer sizes, the weights and the fixed partial selection
   * together, as the recording's manifest and files describe them. Layer
   * captions then show the sizes the model really has instead of the default
   * ARCHITECTURE.
   *
   * @param {{ architecture: object[], weights: Float32Array[],
   *           selection?: Array<ArrayLike<number>[]> | null }} model
   */
  setModel({ architecture, weights, selection = null }) {
    if (architecture && architecture.length >= 2) {
      this.architecture = architecture;
      this.totalConnections = countConnections(architecture);
      // resize() only recomputes the layout when the canvas changes size, so a
      // new architecture has to redo it here.
      if (this.width > 0 && this.height > 0) this.computeLayout();
    }
    this.setSelection(
      Array.isArray(selection) && selection.length > 0
        ? selection
        : initialSelection(this.architecture, weights),
    );
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

      // The same rule as the Partial view, line by line: every connection gets
      // the colour of its own weight at the same constant opacity, so dense areas
      // blend into the average of the weights that are really there instead of
      // taking the colour of whichever group happened to be drawn last.
      for (let out = 0; out < to.size; out += 1) {
        const y2 = to.ys[out];
        const rowOffset = out * inSize;
        for (let input = 0; input < inSize; input += 1) {
          ctx.strokeStyle = connectionStyle(weight[rowOffset + input]);
          ctx.beginPath();
          ctx.moveTo(from.x, from.ys[input]);
          ctx.lineTo(to.x, y2);
          ctx.stroke();
        }
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

      for (let out = 0; out < to.size; out += 1) {
        const y2 = to.ys[out];
        const rowOffset = out * inSize;
        const indices = rows[out];
        if (!indices) continue;
        for (let k = 0; k < indices.length; k += 1) {
          const input = indices[k];
          const w = weight[rowOffset + input];
          // Colour carries sign and magnitude at a constant opacity; width keeps
          // growing with the magnitude, as it always has in this view.
          ctx.strokeStyle = connectionStyle(w);
          ctx.lineWidth = 0.8 + weightScaleMagnitude(w) * 1.6;
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
    // is mixed on top, so an inactive neuron is simply not tinted.
    for (let layer = 0; layer < this.architecture.length; layer += 1) {
      const { x, ys, step, size } = this.layout[layer];
      const radius = Math.max(0.7, Math.min(5, step * 0.34));
      const values = this.activations[layer];
      const scaled = values && values.length === size ? values : null;
      const scale = this.activationMax[layer] > 0 ? 1 / this.activationMax[layer] : 0;

      for (let i = 0; i < size; i += 1) {
        ctx.beginPath();
        ctx.arc(x, ys[i], radius, 0, Math.PI * 2);
        ctx.fillStyle = toRgba(NEURON_COLOUR, 0.9);
        ctx.fill();

        if (!scaled) continue;
        const magnitude = Math.min(1, Math.abs(scaled[i]) * scale);
        if (magnitude <= 0.02) continue;

        // A deeper teal for a stronger activation: dark enough to read on the
        // light canvas, and never a weight colour, so the two never blur.
        ctx.beginPath();
        ctx.arc(x, ys[i], radius, 0, Math.PI * 2);
        ctx.fillStyle = toRgba(
          mixColour(NEURON_COLOUR, ACTIVATION_COLOUR, 0.15 + 0.85 * Math.pow(magnitude, 0.6)),
          1,
        );
        ctx.fill();
      }
    }

    // The 10 output neurons are big enough to carry their class label.
    const outputLayer = this.layout[this.layout.length - 1];
    ctx.font = '500 11px system-ui, "Segoe UI", sans-serif';
    ctx.fillStyle = "rgba(71, 85, 105, 0.95)";
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
      ctx.fillStyle = "rgba(30, 41, 59, 0.92)";
      ctx.fillText(meta.name, x, baseline - 15);
      ctx.font = '500 11px system-ui, "Segoe UI", sans-serif';
      ctx.fillStyle = "rgba(100, 116, 139, 0.95)";
      ctx.fillText(`${meta.size} neurons`, x, baseline);
    }
  }
}

