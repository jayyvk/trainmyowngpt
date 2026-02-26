// microgpt.js — Karpathy's microgpt faithfully ported to JavaScript
// Runs in a Web Worker. Same algorithm, same architecture, same optimizer.

// ============================================================================
// Autograd Engine (identical to Karpathy's Value class)
// ============================================================================
let _id = 0;

class Value {
  constructor(data, children = [], localGrads = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
    this._id = _id++;
  }
  add(other) {
    other = other instanceof Value ? other : new Value(other);
    return new Value(this.data + other.data, [this, other], [1, 1]);
  }
  mul(other) {
    other = other instanceof Value ? other : new Value(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data]);
  }
  pow(n) {
    return new Value(Math.pow(this.data, n), [this], [n * Math.pow(this.data, n - 1)]);
  }
  log() {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }
  exp() {
    const v = Math.exp(this.data);
    return new Value(v, [this], [v]);
  }
  relu() {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }
  neg() { return this.mul(-1); }
  sub(other) { return this.add(other instanceof Value ? other.neg() : new Value(-other)); }
  div(other) { other = other instanceof Value ? other : new Value(other); return this.mul(other.pow(-1)); }

  backward() {
    const topo = [];
    const visited = new Set();
    const build = (v) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) build(child);
        topo.push(v);
      }
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// ============================================================================
// Math helpers (same as Karpathy)
// ============================================================================
function linear(x, w) {
  return w.map(row => {
    let s = new Value(0);
    for (let i = 0; i < x.length; i++) s = s.add(row[i].mul(x[i]));
    return s;
  });
}

function softmax(logits) {
  let maxVal = -Infinity;
  for (const v of logits) if (v.data > maxVal) maxVal = v.data;
  const exps = logits.map(v => v.sub(maxVal).exp());
  let total = new Value(0);
  for (const e of exps) total = total.add(e);
  return exps.map(e => e.div(total));
}

function rmsnorm(x) {
  let ms = new Value(0);
  for (const xi of x) ms = ms.add(xi.mul(xi));
  ms = ms.div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map(xi => xi.mul(scale));
}

// ============================================================================
// RNG
// ============================================================================
let rngState = 42;
function seededRandom() {
  rngState = (rngState * 1664525 + 1013904223) & 0xFFFFFFFF;
  return (rngState >>> 0) / 0xFFFFFFFF;
}
function gaussRandom(std) {
  const u1 = seededRandom();
  const u2 = seededRandom();
  return std * Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}
function sampleFrom(probs) {
  const r = Math.random();
  let c = 0;
  for (let i = 0; i < probs.length; i++) { c += probs[i]; if (r <= c) return i; }
  return probs.length - 1;
}

// ============================================================================
// Model state
// ============================================================================
let stateDict = {};
let params = [];
let adamM, adamV;
let config = {};
let uchars = []; // unique chars (no BOS)
let BOS = 0;
let vocabSize = 0;
let docs = [];

function initTokenizer(text, separator = '\n') {
  docs = text.trim().split(separator).filter(l => l.trim()).map(l => l.trim());
  // Shuffle
  for (let i = docs.length - 1; i > 0; i--) {
    const j = Math.floor(seededRandom() * (i + 1));
    [docs[i], docs[j]] = [docs[j], docs[i]];
  }
  // Karpathy: unique chars sorted, BOS is the last token
  const charSet = new Set();
  for (const doc of docs) for (const ch of doc) charSet.add(ch);
  uchars = Array.from(charSet).sort();
  BOS = uchars.length; // BOS token id is after all chars
  vocabSize = uchars.length + 1; // chars + BOS

  return {
    vocabSize,
    numDocs: docs.length,
    chars: uchars,
    sampleDocs: docs.slice(0, 10),
  };
}

function matrix(nout, nin, std = 0.08) {
  const m = new Array(nout);
  for (let i = 0; i < nout; i++) {
    m[i] = new Array(nin);
    for (let j = 0; j < nin; j++) {
      m[i][j] = new Value(gaussRandom(std));
    }
  }
  return m;
}

function initModel(cfg) {
  config = { ...cfg };
  rngState = config.seed || 42;
  _id = 0;

  const { n_embd, n_layer, block_size } = config;

  // Karpathy: separate lm_head (no weight tying)
  stateDict = {
    wte: matrix(vocabSize, n_embd),
    wpe: matrix(block_size, n_embd),
    lm_head: matrix(vocabSize, n_embd),
  };
  for (let i = 0; i < n_layer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wo`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd);
  }

  params = [];
  for (const name of Object.keys(stateDict)) {
    for (const row of stateDict[name]) {
      for (const p of row) params.push(p);
    }
  }

  adamM = new Float64Array(params.length);
  adamV = new Float64Array(params.length);

  return { numParams: params.length };
}

// ============================================================================
// GPT forward pass (matches Karpathy exactly)
// ============================================================================
function gpt(tokenId, posId, keys, values) {
  const { n_embd, n_head, n_layer, block_size } = config;
  const headDim = Math.floor(n_embd / n_head);

  // Embeddings
  let x = stateDict.wte[tokenId].map((t, i) => t.add(stateDict.wpe[posId % block_size][i]));
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; li++) {
    // Attention block
    const xRes = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn = [];
    for (let h = 0; h < n_head; h++) {
      const hs = h * headDim;
      const qH = q.slice(hs, hs + headDim);
      const kH = keys[li].map(ki => ki.slice(hs, hs + headDim));
      const vH = values[li].map(vi => vi.slice(hs, hs + headDim));

      const attnLogits = kH.map(kT => {
        let dot = new Value(0);
        for (let j = 0; j < headDim; j++) dot = dot.add(qH[j].mul(kT[j]));
        return dot.div(Math.sqrt(headDim));
      });
      const attnWeights = softmax(attnLogits);
      for (let j = 0; j < headDim; j++) {
        let sum = new Value(0);
        for (let t = 0; t < vH.length; t++) sum = sum.add(attnWeights[t].mul(vH[t][j]));
        xAttn.push(sum);
      }
    }
    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xRes[i]));

    // MLP block (Karpathy uses plain relu, not squared relu)
    const xRes2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    x = x.map(xi => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xRes2[i]));
  }

  return linear(x, stateDict.lm_head);
}

// ============================================================================
// Generation (matches Karpathy's inference)
// ============================================================================
function generate(prompt = '', temperature = 0.5, maxLen = null) {
  if (!maxLen) maxLen = config.block_size;
  const nLayer = config.n_layer;
  const keys = Array.from({ length: nLayer }, () => []);
  const valuesCache = Array.from({ length: nLayer }, () => []);

  if (prompt) {
    let tokenId = BOS;
    let logits = gpt(tokenId, 0, keys, valuesCache);
    for (let i = 0; i < prompt.length; i++) {
      const idx = uchars.indexOf(prompt[i]);
      if (idx >= 0) {
        tokenId = idx;
        logits = gpt(tokenId, i + 1, keys, valuesCache);
      }
    }
    const startPos = prompt.length + 1;
    const probs = softmax(logits.map(l => l.div(temperature)));
    tokenId = sampleFrom(probs.map(p => p.data));
    if (tokenId === BOS) return prompt || '(empty)';
    const generated = [prompt, uchars[tokenId]];
    for (let pos = startPos + 1; pos < maxLen; pos++) {
      logits = gpt(tokenId, pos - 1, keys, valuesCache);
      const probs2 = softmax(logits.map(l => l.div(temperature)));
      tokenId = sampleFrom(probs2.map(p => p.data));
      if (tokenId === BOS) break;
      generated.push(uchars[tokenId]);
    }
    return generated.join('');
  } else {
    let tokenId = BOS;
    const sample = [];
    for (let pos = 0; pos < maxLen; pos++) {
      const logits = gpt(tokenId, pos, keys, valuesCache);
      const probs = softmax(logits.map(l => l.div(temperature)));
      tokenId = sampleFrom(probs.map(p => p.data));
      if (tokenId === BOS) break;
      sample.push(uchars[tokenId]);
    }
    return sample.join('') || '(empty)';
  }
}

// ============================================================================
// Training (matches Karpathy's training loop)
// ============================================================================
function trainStep(step, totalSteps) {
  const { block_size, n_layer } = config;
  // Karpathy's Adam hyperparams
  const learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;

  // Pick document, tokenize with BOS on both sides
  const doc = docs[step % docs.length];
  let tokens = [BOS];
  for (const ch of doc) {
    const idx = uchars.indexOf(ch);
    if (idx >= 0) tokens.push(idx);
  }
  tokens.push(BOS); // BOS on both sides, not EOS
  const n = Math.min(block_size, tokens.length - 1);

  const keys = Array.from({ length: n_layer }, () => []);
  const valuesCache = Array.from({ length: n_layer }, () => []);

  // Forward: collect per-position losses
  const losses = [];
  for (let pos = 0; pos < n; pos++) {
    const logits = gpt(tokens[pos], pos, keys, valuesCache);
    const probs = softmax(logits);
    const lossT = probs[tokens[pos + 1]].log().neg();
    losses.push(lossT);
  }

  // Average loss over document, then single backward
  let loss = new Value(0);
  for (const lt of losses) loss = loss.add(lt);
  loss = loss.div(n);
  loss.backward();

  // Adam optimizer (Karpathy's exact hyperparams)
  const lrT = learning_rate * (1 - step / totalSteps);
  for (let i = 0; i < params.length; i++) {
    adamM[i] = beta1 * adamM[i] + (1 - beta1) * params[i].grad;
    adamV[i] = beta2 * adamV[i] + (1 - beta2) * params[i].grad * params[i].grad;
    const mHat = adamM[i] / (1 - Math.pow(beta1, step + 1));
    const vHat = adamV[i] / (1 - Math.pow(beta2, step + 1));
    params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
    params[i].grad = 0;
  }

  return { loss: loss.data, lr: lrT, doc };
}

// ============================================================================
// Web Worker message handler
// ============================================================================
let stopRequested = false;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'load_dataset') {
    const result = initTokenizer(data.text, data.separator || '\n');
    self.postMessage({ type: 'dataset_loaded', data: result });
  }

  if (type === 'init_model') {
    const result = initModel(data.config);
    self.postMessage({ type: 'model_initialized', data: { ...result, config } });
  }

  if (type === 'train') {
    stopRequested = false;
    const totalSteps = config.num_steps;
    const startTime = performance.now();

    for (let step = 0; step < totalSteps; step++) {
      if (stopRequested) {
        self.postMessage({ type: 'stopped', data: { step } });
        return;
      }

      const stepStart = performance.now();
      const result = trainStep(step, totalSteps);
      const stepTime = performance.now() - stepStart;

      let sample = null;
      if (step % 50 === 0 || step === totalSteps - 1) {
        sample = generate('', 0.5);
      }

      self.postMessage({
        type: 'step',
        data: {
          step: step + 1,
          totalSteps,
          loss: Math.round(result.loss * 10000) / 10000,
          lr: Math.round(result.lr * 100000) / 100000,
          stepTimeMs: Math.round(stepTime),
          sample,
          doc: result.doc,
          elapsed: Math.round(performance.now() - startTime),
        }
      });

      if (step % 5 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    const samples = Array.from({ length: 10 }, () => generate('', 0.5));
    const totalTime = performance.now() - startTime;
    self.postMessage({
      type: 'complete',
      data: { samples, totalSteps, totalTimeMs: Math.round(totalTime) }
    });
  }

  if (type === 'stop') { stopRequested = true; }

  if (type === 'generate') {
    const { prompt, temperature, count } = data;
    const samples = Array.from({ length: count || 5 }, () =>
      generate(prompt || '', temperature || 0.5)
    );
    self.postMessage({ type: 'generated', data: { samples, prompt, temperature } });
  }
};
