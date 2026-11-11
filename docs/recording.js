/**
 * recording.js — reads a recorded training run from static files.
 *
 * A recording is the output of `recorder/record_training.py` (Milestone 5A): a
 * JSON manifest that describes every byte, one file of initial full weights,
 * one file of fixed Partial-mode connection indices, and numbered chunk files of
 * fixed-stride checkpoint records. This module is the only place that knows that
 * layout, so the replay never parses numbers out of JSON and never guesses an
 * offset.
 *
 * It uses nothing but `fetch()`, `ArrayBuffer` and `DataView`. The manifest and
 * the two small index files are read once; checkpoint chunks are fetched on
 * demand as the replay seeks, and a few recently used chunks stay in memory so
 * stepping or playing back and forth does not download them again. It opens no
 * socket, contacts nothing but the static files it is pointed at, and
 * recomputes nothing: every number it returns is the number in the file.
 *
 * Binary layout (plan.md section 7.4), one 5,688-byte record per checkpoint:
 *
 * ======  ======  ==========================================  ==================
 * Offset  Bytes   Field                                       Type
 * ======  ======  ==========================================  ==================
 * 0       4       completed optimiser step                    uint32
 * 4       4       sample position in the epoch (1-based)      uint32
 * 8       1       true label (0-9)                            uint8
 * 9       1       predicted digit                             uint8
 * 10      1       flags (bit 0 = has sample)                  uint8
 * 11      1       reserved (0)                                uint8
 * 12      784     image, row-major 28 x 28                    uint8
 * 796     40      softmax probabilities, classes 0-9          float32 x 10
 * 836     4       cross-entropy loss                          float32
 * 840     808     activations A1 (128), A2 (64), output (10)  float32 x 202
 * 1,648   4,040   Partial weights over the 1,010 indices     float32 x 1,010
 * ======  ======  ==========================================  ==================
 *
 * The offsets, dtypes and counts are taken from the manifest, and every one of
 * them is checked against the declared record size before a record is decoded.
 */

/** Format name and version this adapter understands (`manifest.format`). */
export const RECORDING_FORMAT = "mlp-visualiser-recording";
export const RECORDING_VERSION = 1;

/**
 * Chunks kept in memory. One chunk holds 500 checkpoints (2.8 MB), so four cover
 * every neighbour of the checkpoint on screen while playback advances: a full
 * recording is never downloaded, and a seek normally reuses a cached chunk.
 */
const DEFAULT_CHUNK_CACHE_SIZE = 4;

const DTYPE_BYTES = { uint8: 1, uint16: 2, uint32: 4, float32: 4 };

/**
 * True when this platform stores typed arrays little-endian, which is the byte
 * order every recording uses. Every mainstream browser is; the check only exists
 * so the fallback below is unambiguous rather than accidental.
 */
const LITTLE_ENDIAN_PLATFORM = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** An error a reader can act on: a missing file, a broken manifest, a bad chunk. */
export class RecordingError extends Error {
  constructor(message) {
    super(message);
    this.name = "RecordingError";
  }
}

function readFloat32(view, offset, count) {
  if (count === 0) return new Float32Array(0);
  if (LITTLE_ENDIAN_PLATFORM) {
    // The bytes are already little-endian float32 in row-major order, exactly
    // what Float32Array wants. Copy them so a decoded record never keeps an
    // entire chunk alive once that chunk leaves the cache.
    return new Float32Array(view.buffer, view.byteOffset + offset, count).slice();
  }
  const values = new Float32Array(count);
  for (let i = 0; i < count; i += 1) values[i] = view.getFloat32(offset + i * 4, true);
  return values;
}

async function fetchBytes(url, label) {
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    const reason = error && error.message ? error.message : String(error);
    throw new RecordingError(`could not fetch ${label} from ${url}: ${reason}`);
  }
  if (!response.ok) {
    throw new RecordingError(`${label} is missing at ${url} (HTTP ${response.status} ${response.statusText})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * Expands the manifest's chunk naming pattern, for example
 * `chunks/chunk-{index:04d}.bin` for chunk 12 -> `chunks/chunk-0012.bin`.
 */
function chunkFileName(naming, index) {
  return naming.replace(/\{index(?::0?(\d+)d)?\}/g, (_, width) =>
    String(index).padStart(width ? Number(width) : 1, "0"),
  );
}

function requireNumber(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new RecordingError(`manifest ${label} is not a non-negative integer (${value})`);
  }
  return value;
}

/**
 * One recorded run, read from a directory of static files.
 *
 * `open()` fetches and validates only `manifest.json`; weights, indices and
 * checkpoint chunks are read when they are first needed.
 */
export class Recording {
  /**
   * @param {string} baseUrl   directory of the recording, with a trailing slash
   * @param {object} manifest  the parsed `manifest.json`
   * @param {{ chunkCacheSize?: number }} options
   */
  constructor(baseUrl, manifest, { chunkCacheSize = DEFAULT_CHUNK_CACHE_SIZE } = {}) {
    this.baseUrl = baseUrl;
    this.manifest = manifest;
    this.chunkCacheSize = Math.max(1, chunkCacheSize);
    /** Chunk index -> ArrayBuffer, oldest first (a Map keeps insertion order). */
    this.chunks = new Map();
    /** Chunk index -> in-flight promise, so one chunk is never fetched twice. */
    this.pendingChunks = new Map();

    this.readManifest();
  }

  /**
   * Opens a recording directory. The only network request is the manifest.
   * @param {string} baseUrl directory holding `manifest.json`
   * @param {{ chunkCacheSize?: number }} options
   * @returns {Promise<Recording>}
   */
  static async open(baseUrl, options = {}) {
    if (typeof baseUrl !== "string" || baseUrl.trim() === "") {
      throw new RecordingError("no recording directory was given");
    }
    const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    let manifest;
    try {
      const bytes = await fetchBytes(`${base}manifest.json`, "recording manifest");
      manifest = JSON.parse(new TextDecoder().decode(bytes));
    } catch (error) {
      if (error instanceof RecordingError) throw error;
      throw new RecordingError(`manifest.json at ${base} is not valid JSON: ${error.message}`);
    }
    return new Recording(base, manifest, options);
  }

  /** Validates the manifest and derives every offset the decoder needs. */
  readManifest() {
    const { manifest } = this;
    const format = manifest.format;
    if (!format || typeof format !== "object") {
      throw new RecordingError("the manifest has no format block");
    }
    if (format.name !== RECORDING_FORMAT) {
      throw new RecordingError(`unexpected recording format "${format.name}"`);
    }
    if (format.version !== RECORDING_VERSION) {
      throw new RecordingError(`unsupported recording version ${format.version}`);
    }
    if (format.byte_order !== "little-endian") {
      throw new RecordingError(`unsupported byte order "${format.byte_order}"`);
    }

    // --- record layout -----------------------------------------------------
    const record = format.record;
    if (!record || !Array.isArray(record.fields) || record.fields.length === 0) {
      throw new RecordingError("the manifest has no record layout");
    }
    this.recordSize = requireNumber(format.record_size, "format.record_size");
    if (record.size_bytes !== this.recordSize) {
      throw new RecordingError(
        `manifest record sizes disagree: format.record_size ${this.recordSize} vs record.size_bytes ${record.size_bytes}`,
      );
    }

    this.layout = {};
    for (const field of record.fields) {
      const width = DTYPE_BYTES[field.dtype];
      if (!width) throw new RecordingError(`unknown dtype "${field.dtype}" for field ${field.name}`);
      const bytes = requireNumber(field.bytes, `field ${field.name} bytes`);
      const count = requireNumber(field.count, `field ${field.name} count`);
      if (bytes !== count * width) {
        throw new RecordingError(
          `field ${field.name} declares ${bytes} bytes for ${count} x ${field.dtype}`,
        );
      }
      if (field.offset + bytes > this.recordSize) {
        throw new RecordingError(`field ${field.name} overruns the ${this.recordSize}-byte record`);
      }
      this.layout[field.name] = field;
    }

    for (const name of [
      "step",
      "sample_position",
      "true_label",
      "predicted_label",
      "flags",
      "image",
      "probabilities",
      "loss",
      "activations",
      "partial_weights",
    ]) {
      if (!this.layout[name]) throw new RecordingError(`the record layout has no "${name}" field`);
    }
    this.hasSampleBit = requireNumber(
      (record.flags && record.flags.has_sample_bit) || 0,
      "record.flags.has_sample_bit",
    );

    // --- chunk layout ------------------------------------------------------
    const chunks = format.chunks;
    if (!chunks || typeof chunks.naming !== "string") {
      throw new RecordingError("the manifest has no chunk layout");
    }
    this.chunkSize = requireNumber(chunks.chunk_size_records, "chunks.chunk_size_records");
    if (this.chunkSize === 0) throw new RecordingError("chunks.chunk_size_records is zero");
    this.chunkNaming = chunks.naming;

    // --- recorded checkpoints ---------------------------------------------
    const training = manifest.training || {};
    const checkpoints = manifest.checkpoints || {};
    this.checkpointCount = requireNumber(
      training.checkpoint_count ?? checkpoints.count,
      "training.checkpoint_count",
    );
    if (this.checkpointCount === 0) throw new RecordingError("the recording has no checkpoints");
    this.chunkCount = Math.ceil(this.checkpointCount / this.chunkSize);

    this.recordedSteps = Array.isArray(training.recorded_steps) ? training.recorded_steps : [];
    if (this.recordedSteps.length !== this.checkpointCount) {
      throw new RecordingError(
        `manifest lists ${this.recordedSteps.length} steps for ${this.checkpointCount} checkpoints`,
      );
    }
    this.stepToIndex = new Map(this.recordedSteps.map((step, index) => [step, index]));
    this.cadence = requireNumber(training.snapshot_cadence ?? 1, "training.snapshot_cadence");
    this.firstStep = this.recordedSteps[0];
    this.lastStep = this.recordedSteps[this.recordedSteps.length - 1];

    // --- model -------------------------------------------------------------
    const model = manifest.model || {};
    this.architecture = Array.isArray(model.architecture) ? model.architecture : [];
    this.layers = Array.isArray(model.layers) ? model.layers : [];
    if (this.architecture.length < 2 || this.layers.length !== this.architecture.length - 1) {
      throw new RecordingError("the manifest does not describe a layered model");
    }
    for (const layer of this.layers) {
      if (layer.weight_count !== layer.in * layer.out) {
        throw new RecordingError(`layer ${layer.index} has an inconsistent weight count`);
      }
    }
    this.connectionCount = this.layers.reduce((total, layer) => total + layer.weight_count, 0);
    this.totalSteps = requireNumber(model.total_steps ?? this.lastStep, "model.total_steps");
    this.epochs = Math.max(1, model.epochs || 1);
    this.trainSamples = requireNumber(
      model.train_samples ?? this.totalSteps,
      "model.train_samples",
    );

    const images = manifest.images || {};
    this.imageWidth = images.width || 28;
    this.imageHeight = images.height || 28;

    const activations = manifest.activations || {};
    this.activationSizes = Array.isArray(activations.sizes) ? activations.sizes : [];
    if (this.layout.activations.count !== this.activationSizes.reduce((a, b) => a + b, 0)) {
      throw new RecordingError("the activation sizes do not add up to the stored activation count");
    }

    // --- initial full weights (and biases) and fixed Partial indices -------
    const full = manifest.full || {};
    this.weightsFile = full.weights_file || "model.bin";
    this.biasesFile = full.biases_file || "model-biases.bin";
    this.biasCount = requireNumber(
      full.bias_count ?? this.layers.reduce((total, layer) => total + (layer.bias_count || 0), 0),
      "full.bias_count",
    );

    const partial = manifest.partial || {};
    this.partial = {
      count: requireNumber(partial.count, "partial.count"),
      topK: requireNumber(partial.top_k, "partial.top_k"),
      indicesFile: partial.indices_file || "partial-indices.bin",
      layers: Array.isArray(partial.layers) ? partial.layers : [],
    };
    if (this.partial.layers.length !== this.layers.length) {
      throw new RecordingError("the Partial layers do not match the model layers");
    }
    if (this.layout.partial_weights.count !== this.partial.count) {
      throw new RecordingError(
        `each record stores ${this.layout.partial_weights.count} Partial weights, the manifest says ${this.partial.count}`,
      );
    }
    this.partialWeightCount = this.partial.layers.reduce(
      (total, layer) => total + layer.out * layer.k,
      0,
    );
    if (this.partialWeightCount !== this.partial.count) {
      throw new RecordingError("the Partial layer sizes do not add up to the recorded index count");
    }

    const colour = manifest.colour || {};
    this.colour = { min: colour.min, max: colour.max };
  }

  // --- lookups -------------------------------------------------------------

  /** The recorded optimiser step of checkpoint `index`. */
  stepOf(index) {
    return this.recordedSteps[index];
  }

  /** The checkpoint index that recorded `step`, or -1 if that step was not recorded. */
  indexOfStep(step) {
    const index = this.stepToIndex.get(step);
    return index === undefined ? -1 : index;
  }

  /** Absolute URL of a file inside the recording directory. */
  url(name) {
    return `${this.baseUrl}${String(name).replace(/^\/+/, "")}`;
  }

  chunkUrl(chunkIndex) {
    return this.url(chunkFileName(this.chunkNaming, chunkIndex));
  }

  /** Records stored in chunk `chunkIndex` (the last chunk is usually short). */
  chunkRecordCount(chunkIndex) {
    return Math.min(this.chunkSize, this.checkpointCount - chunkIndex * this.chunkSize);
  }

  chunkOf(index) {
    return Math.floor(index / this.chunkSize);
  }

  cachedChunkCount() {
    return this.chunks.size;
  }

  // --- chunk loading -------------------------------------------------------

  /**
   * One chunk's bytes, from the cache when possible. The cache keeps a small
   * number of recently used chunks; the least recently used one is dropped.
   * @param {number} chunkIndex
   * @returns {Promise<ArrayBuffer>}
   */
  async loadChunk(chunkIndex) {
    if (!Number.isInteger(chunkIndex) || chunkIndex < 0 || chunkIndex >= this.chunkCount) {
      throw new RecordingError(`chunk ${chunkIndex} is outside 0..${this.chunkCount - 1}`);
    }

    const cached = this.chunks.get(chunkIndex);
    if (cached) {
      // Refresh its position so the LRU order follows what is on screen.
      this.chunks.delete(chunkIndex);
      this.chunks.set(chunkIndex, cached);
      return cached;
    }

    const inFlight = this.pendingChunks.get(chunkIndex);
    if (inFlight) return inFlight;

    const url = this.chunkUrl(chunkIndex);
    const expected = this.chunkRecordCount(chunkIndex) * this.recordSize;
    const request = fetchBytes(url, `chunk ${chunkIndex}`)
      .then((bytes) => {
        if (bytes.byteLength !== expected) {
          throw new RecordingError(
            `chunk ${chunkIndex} is ${bytes.byteLength} bytes, the manifest implies ${expected}`,
          );
        }
        this.cacheChunk(chunkIndex, bytes.buffer);
        return bytes.buffer;
      })
      .finally(() => this.pendingChunks.delete(chunkIndex));

    this.pendingChunks.set(chunkIndex, request);
    return request;
  }

  cacheChunk(chunkIndex, buffer) {
    this.chunks.delete(chunkIndex);
    this.chunks.set(chunkIndex, buffer);
    while (this.chunks.size > this.chunkCacheSize) {
      const oldest = this.chunks.keys().next().value;
      this.chunks.delete(oldest);
    }
  }

  /**
   * Fetches the chunk holding `index` in the background. Prefetching is only an
   * optimisation: a failure is ignored here and reported by the seek that needs
   * the chunk.
   */
  prefetch(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.checkpointCount) return;
    const chunkIndex = this.chunkOf(index);
    if (this.chunks.has(chunkIndex) || this.pendingChunks.has(chunkIndex)) return;
    void this.loadChunk(chunkIndex).catch(() => {});
  }

  // --- decoding ------------------------------------------------------------

  /**
   * Decodes checkpoint `index` from its chunk. The step in the record is checked
   * against the manifest, so a decoded checkpoint can never describe a step the
   * manifest did not record at that position.
   * @param {number} index checkpoint index, 0-based
   * @returns {Promise<object>} the recorded values of that checkpoint
   */
  async checkpoint(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.checkpointCount) {
      throw new RecordingError(`checkpoint ${index} is outside 0..${this.checkpointCount - 1}`);
    }

    const chunkIndex = this.chunkOf(index);
    const buffer = await this.loadChunk(chunkIndex);
    const offset = (index % this.chunkSize) * this.recordSize;
    const record = this.decodeRecord(new DataView(buffer), offset);

    const expected = this.recordedSteps[index];
    if (record.step !== expected) {
      throw new RecordingError(
        `checkpoint ${index} records step ${record.step}, the manifest says ${expected}`,
      );
    }
    return { index, chunkIndex, ...record };
  }

  /** Reads every field of one record with the offsets the manifest declares. */
  decodeRecord(view, offset) {
    const layout = this.layout;
    const flags = view.getUint8(offset + layout.flags.offset);
    const imageStart = view.byteOffset + offset + layout.image.offset;

    return {
      step: view.getUint32(offset + layout.step.offset, true),
      samplePosition: view.getUint32(offset + layout.sample_position.offset, true),
      trueLabel: view.getUint8(offset + layout.true_label.offset),
      predictedLabel: view.getUint8(offset + layout.predicted_label.offset),
      hasSample: (flags & (1 << this.hasSampleBit)) !== 0,
      image: new Uint8Array(view.buffer, imageStart, layout.image.count).slice(),
      probabilities: readFloat32(
        view,
        offset + layout.probabilities.offset,
        layout.probabilities.count,
      ),
      loss: view.getFloat32(offset + layout.loss.offset, true),
      activations: readFloat32(view, offset + layout.activations.offset, layout.activations.count),
      partialWeights: readFloat32(
        view,
        offset + layout.partial_weights.offset,
        layout.partial_weights.count,
      ),
    };
  }

  // --- one-off files -------------------------------------------------------

  /**
   * The initial full weight matrices, one Float32Array per layer in `[out][in]`
   * row-major order. Read once: the recording stores no later full weights, so
   * Full mode shows exactly these values for the whole replay.
   * @returns {Promise<Float32Array[]>}
   */
  async loadInitialWeights() {
    const bytes = await fetchBytes(this.url(this.weightsFile), "initial weights");
    const expected = this.connectionCount * 4;
    if (bytes.byteLength !== expected) {
      throw new RecordingError(
        `${this.weightsFile} is ${bytes.byteLength} bytes, expected ${expected} for ${this.connectionCount} weights`,
      );
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 0;
    const weights = this.layers.map((layer) => {
      const count = layer.in * layer.out;
      const values = readFloat32(view, offset, count);
      offset += count * 4;
      return values;
    });
    if (offset !== expected) {
      throw new RecordingError("the initial weights do not cover every layer exactly");
    }
    return weights;
  }

  /**
   * The initial biases, one Float32Array per layer. They are recorded with the
   * initial weights so a recording describes a complete model; the replay reads
   * them once at startup and reports the count, because the diagram draws
   * weights and activations, and every activation it shows is recorded.
   * @returns {Promise<Float32Array[]>}
   */
  async loadBiases() {
    const bytes = await fetchBytes(this.url(this.biasesFile), "initial biases");
    const expected = this.biasCount * 4;
    if (bytes.byteLength !== expected) {
      throw new RecordingError(
        `${this.biasesFile} is ${bytes.byteLength} bytes, expected ${expected} for ${this.biasCount} biases`,
      );
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 0;
    const biases = this.layers.map((layer) => {
      const count = layer.bias_count || 0;
      const values = readFloat32(view, offset, count);
      offset += count * 4;
      return values;
    });
    if (offset !== expected) {
      throw new RecordingError("the initial biases do not cover every layer exactly");
    }
    return biases;
  }

  /**
   * The fixed Partial-mode selection: one array of input indices per receiving
   * neuron, per layer, in the recorded order. The same selection applies to
   * every checkpoint, so the Partial weights of a checkpoint line up with it.
   * @returns {Promise<number[][][]>}
   */
  async loadPartialSelection() {
    const bytes = await fetchBytes(this.url(this.partial.indicesFile), "Partial indices");
    const expected = this.partial.count * 2;
    if (bytes.byteLength !== expected) {
      throw new RecordingError(
        `${this.partial.indicesFile} is ${bytes.byteLength} bytes, expected ${expected} for ${this.partial.count} uint16 indices`,
      );
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let cursor = 0;
    const selection = this.partial.layers.map((layer) => {
      if (layer.k !== this.partial.topK) {
        throw new RecordingError(`Partial layer ${layer.index} stores ${layer.k} indices per neuron`);
      }
      const rows = new Array(layer.out);
      for (let out = 0; out < layer.out; out += 1) {
        const row = new Array(layer.k);
        for (let k = 0; k < layer.k; k += 1) {
          const input = view.getUint16(cursor * 2, true);
          if (input >= layer.in) {
            throw new RecordingError(
              `Partial index ${input} is outside layer ${layer.index} (${layer.in} inputs)`,
            );
          }
          row[k] = input;
          cursor += 1;
        }
        rows[out] = row;
      }
      return rows;
    });

    if (cursor !== this.partial.count) {
      throw new RecordingError(
        `${this.partial.indicesFile} holds ${cursor} indices, the manifest says ${this.partial.count}`,
      );
    }
    return selection;
  }
}
