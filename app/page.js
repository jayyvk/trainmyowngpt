'use client';
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

const PRESETS = {
  ycstartups: { label: 'YC Startups', desc: '5,000+ startup names', file: '/datasets/ycstartups.txt', data: null },
  names: { label: 'Baby Names', desc: '2,000+ popular names', file: '/datasets/names.txt', data: null },
  dinos: { label: 'Dinosaurs', desc: '1,500+ species names', file: '/datasets/dinos.txt', data: null },
  words: { label: 'English Words', desc: '10,000 common words', file: '/datasets/words.txt', data: null },
};

const WIRES = [
  ['dataset', 'tokenizer'],
  ['tokenizer', 'config'],
  ['config', 'training'],
  ['training', 'metrics'],
  ['training', 'generate'],
];

// Position registry
function useNodePositions() {
  const pos = useRef({});
  const [tick, setTick] = useState(0);
  const update = useCallback((id, x, y, w, h) => {
    pos.current[id] = { x, y, w, h };
    setTick(t => t + 1);
  }, []);
  return { positions: pos.current, update, tick };
}

// Draggable
function useDraggable(ix, iy, id, onPos) {
  const [p, setP] = useState({ x: ix, y: iy });
  const drag = useRef(false);
  const off = useRef({ x: 0, y: 0 });
  const ref = useRef(null);

  const onMouseDown = useCallback((e) => {
    if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(e.target.tagName)) return;
    drag.current = true;
    off.current = { x: e.clientX - p.x, y: e.clientY - p.y };
    e.preventDefault();
  }, [p]);

  useEffect(() => {
    const mv = (e) => { if (drag.current) setP({ x: e.clientX - off.current.x, y: e.clientY - off.current.y }); };
    const up = () => { drag.current = false; };
    window.addEventListener('mousemove', mv);
    window.addEventListener('mouseup', up);
    return () => { window.removeEventListener('mousemove', mv); window.removeEventListener('mouseup', up); };
  }, []);

  useEffect(() => {
    if (ref.current) {
      const r = ref.current.getBoundingClientRect();
      onPos(id, p.x, p.y, r.width, r.height);
    }
  }, [p, id, onPos]);

  return { p, onMouseDown, ref };
}

// Node
function Node({ x, y, w, title, children, active, accent, id, onPosChange }) {
  const { p, onMouseDown, ref } = useDraggable(x, y, id, onPosChange);
  return (
    <div ref={ref} onMouseDown={onMouseDown} style={{
      position: 'absolute', left: p.x, top: p.y, width: w,
      background: '#fff', borderRadius: '3px',
      border: '1px solid ' + (active ? (accent || '#333') : '#e0e0e0'),
      boxShadow: active ? '0 2px 12px ' + (accent || '#333') + '15' : '0 1px 3px rgba(0,0,0,0.04)',
      cursor: 'grab', userSelect: 'none', zIndex: 10,
      transition: 'border-color 0.3s, box-shadow 0.3s',
    }}>
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid ' + (active ? (accent || '#333') + '20' : '#f0f0f0'),
        display: 'flex', alignItems: 'center', gap: '8px',
      }}>
        <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '1.5px', color: active ? '#555' : '#bbb', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase' }}>{title}</span>
      </div>
      <div style={{ padding: '12px', fontFamily: 'JetBrains Mono, monospace', fontSize: '11px' }}>{children}</div>
    </div>
  );
}

// Wires
function Wires({ positions, wires, tick }) {
  return (
    <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 5 }}>
      {wires.map(([from, to], i) => {
        const a = positions[from], b = positions[to];
        if (!a || !b) return null;
        // Smart connection: pick closest edges
        const aCx = a.x + a.w / 2, aCy = a.y + a.h / 2;
        const bCx = b.x + b.w / 2, bCy = b.y + b.h / 2;
        // Determine which edges to connect
        let x1, y1, x2, y2;
        const dx = bCx - aCx, dy = bCy - aCy;
        if (Math.abs(dx) > Math.abs(dy)) {
          // Horizontal connection
          if (dx > 0) { x1 = a.x + a.w; y1 = aCy; x2 = b.x; y2 = bCy; }
          else { x1 = a.x; y1 = aCy; x2 = b.x + b.w; y2 = bCy; }
        } else {
          // Vertical connection
          if (dy > 0) { x1 = aCx; y1 = a.y + a.h; x2 = bCx; y2 = b.y; }
          else { x1 = aCx; y1 = a.y; x2 = bCx; y2 = b.y + b.h; }
        }
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        let d;
        if (Math.abs(dx) > Math.abs(dy)) {
          d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
        } else {
          d = `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
        }
        return (
          <g key={i}>
            <path d={d} fill="none" stroke="#d4d4d4" strokeWidth="1.5" strokeDasharray="6,4" />
            <circle cx={x2} cy={y2} r="3" fill="#d4d4d4" />
            <circle cx={x1} cy={y1} r="2.5" fill="none" stroke="#d4d4d4" strokeWidth="1" />
          </g>
        );
      })}
    </svg>
  );
}

// Loss chart (smoothed)
function LossChart({ data, height = 70 }) {
  if (!data.length) return null;
  const sm = []; let ema = data[0].loss;
  for (const d of data) { ema = 0.95 * ema + 0.05 * d.loss; sm.push(ema); }
  const mx = Math.max(...sm), mn = Math.min(...sm), rng = mx - mn || 1;
  const toY = l => ((mx - l) / rng) * 90 + 5;
  const toX = i => (i / Math.max(sm.length - 1, 1)) * 100;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height, display: 'block' }}>
      {data.length < 400 && data.map((d, i) => <circle key={i} cx={toX(i)} cy={toY(Math.max(mn, Math.min(mx, d.loss)))} r="0.3" fill="#e8e8e8" vectorEffect="non-scaling-stroke" />)}
      <polyline fill="none" stroke="#333" strokeWidth="1.2" points={sm.map((v, i) => `${toX(i)},${toY(v)}`).join(' ')} vectorEffect="non-scaling-stroke" />
      {sm.length > 1 && <circle cx={toX(sm.length - 1)} cy={toY(sm[sm.length - 1])} r="2" fill="#333" vectorEffect="non-scaling-stroke" />}
      <text x="1" y="8" fontSize="5" fill="#ccc" vectorEffect="non-scaling-stroke">{mx.toFixed(1)}</text>
      <text x="1" y="98" fontSize="5" fill="#ccc" vectorEffect="non-scaling-stroke">{mn.toFixed(1)}</text>
    </svg>
  );
}

// Step time bar chart
function StepTimeChart({ data, height = 35 }) {
  if (!data.length) return null;
  const mx = Math.max(...data, 1);
  return (
    <svg viewBox={'0 0 ' + data.length + ' 100'} preserveAspectRatio="none" style={{ width: '100%', height, display: 'block' }}>
      {data.map((v, i) => <rect key={i} x={i} y={100 - (v / mx) * 90} width="0.8" height={(v / mx) * 90} fill="#dbeafe" />)}
    </svg>
  );
}

// ============================================================================
// Main App
// ============================================================================
export default function TrainMyOwnGPT() {
  const workerRef = useRef(null);
  const { positions, update: updatePos, tick: posTick } = useNodePositions();

  const [dataset, setDataset] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [config, setConfig] = useState({ n_embd: 16, n_head: 4, n_layer: 1, block_size: 16, num_steps: 1000, learning_rate: 0.01, seed: 42 });
  const [training, setTraining] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  const [stepTimes, setStepTimes] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [currentSample, setCurrentSample] = useState('');
  const [stepTime, setStepTime] = useState(0);
  const [totalTime, setTotalTime] = useState(0);
  const [finalSamples, setFinalSamples] = useState([]);
  const [genPrompt, setGenPrompt] = useState('');
  const [genTemp, setGenTemp] = useState(0.8);
  const [genResults, setGenResults] = useState([]);
  const [generating, setGenerating] = useState(false);
  const [runHistory, setRunHistory] = useState([]);

  useEffect(() => {
    const w = new Worker('/microgpt-worker.js');
    workerRef.current = w;
    w.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === 'dataset_loaded') {
        setDataset(data);
        // Auto-suggest steps based on dataset size: aim for ~3 passes through data
        const suggested = Math.min(3000, Math.max(500, data.numDocs * 3));
        setConfig(c => ({ ...c, num_steps: Math.round(suggested / 50) * 50 }));
      }
      if (type === 'step') {
        setCurrentStep(data.step); setCurrentLoss(data.loss); setStepTime(data.stepTimeMs); setTotalTime(data.elapsed);
        setLossHistory(prev => [...prev, { step: data.step, loss: data.loss }]);
        setStepTimes(prev => [...prev, data.stepTimeMs]);
        if (data.sample) setCurrentSample(data.sample);
      }
      if (type === 'complete') {
        setTraining(false); setModelReady(true); setFinalSamples(data.samples); setTotalTime(data.totalTimeMs);
      }
      if (type === 'stopped') setTraining(false);
      if (type === 'generated') { setGenResults(data.samples); setGenerating(false); }
    };
    return () => w.terminate();
  }, []);

  const loadPreset = useCallback(async (key) => {
    setSelectedPreset(key);
    if (!key || !PRESETS[key]) return;
    setModelReady(false); setLossHistory([]); setStepTimes([]); setCurrentStep(0); setFinalSamples([]); setGenResults([]);

    let text = PRESETS[key].data;
    if (!text && PRESETS[key].file) {
      try {
        const res = await fetch(PRESETS[key].file);
        text = await res.text();
      } catch (err) {
        console.error('Failed to load dataset:', err);
        return;
      }
    }
    if (text) workerRef.current?.postMessage({ type: 'load_dataset', data: { text } });
  }, []);

  const handleFileDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setSelectedPreset('custom'); setModelReady(false); setLossHistory([]); setStepTimes([]); setFinalSamples([]); setGenResults([]);
      workerRef.current?.postMessage({ type: 'load_dataset', data: { text: ev.target.result } });
    };
    reader.readAsText(file);
  }, []);

  const startTraining = useCallback(() => {
    if (!dataset) return;
    setTraining(true); setModelReady(false); setLossHistory([]); setStepTimes([]); setCurrentStep(0); setCurrentSample(''); setFinalSamples([]); setGenResults([]);
    workerRef.current?.postMessage({ type: 'init_model', data: { config } });
    setTimeout(() => workerRef.current?.postMessage({ type: 'train' }), 50);
  }, [dataset, config]);

  const stopTraining = useCallback(() => workerRef.current?.postMessage({ type: 'stop' }), []);

  const doGenerate = useCallback(() => {
    setGenerating(true);
    workerRef.current?.postMessage({ type: 'generate', data: { prompt: genPrompt, temperature: genTemp, count: 8 } });
  }, [genPrompt, genTemp]);

  const paramCount = useMemo(() => {
    const { n_embd, n_layer, block_size } = config;
    const v = dataset?.vocabSize || 28;
    // wte + wpe + lm_head + per-layer (wq,wk,wv,wo,fc1,fc2)
    return v * n_embd + block_size * n_embd + v * n_embd +
      n_layer * (4 * n_embd * n_embd + 4 * n_embd * n_embd + n_embd * 4 * n_embd + n_embd * 4 * n_embd);
  }, [config, dataset]);

  const flopsPerStep = paramCount * config.block_size * 2;
  const totalFlops = flopsPerStep * currentStep;
  const fmt = (ms) => ms < 1000 ? ms + 'ms' : (ms / 1000).toFixed(1) + 's';
  const fmtF = (f) => f > 1e9 ? (f / 1e9).toFixed(1) + 'G' : f > 1e6 ? (f / 1e6).toFixed(1) + 'M' : f > 1e3 ? (f / 1e3).toFixed(0) + 'K' : '' + f;

  // Center the layout based on viewport
  const [offset, setOffset] = useState(60);
  useEffect(() => {
    const calcOffset = () => {
      const totalWidth = 1140;
      const ox = Math.max(60, Math.floor((window.innerWidth - totalWidth) / 2));
      setOffset(ox);
    };
    calcOffset();
    window.addEventListener('resize', calcOffset);
    return () => window.removeEventListener('resize', calcOffset);
  }, []);

  // Auto-save completed runs
  useEffect(() => {
    if (modelReady && lossHistory.length > 0) {
      const finalLoss = lossHistory[lossHistory.length - 1].loss;
      setRunHistory(prev => [...prev, { id: Date.now(), config: { ...config }, finalLoss, totalTime, samples: finalSamples.slice(0, 3) }]);
    }
  }, [modelReady]);

  const sliderStyle = { flex: 1, accentColor: '#999', cursor: 'pointer', height: '2px' };

  return (
    <div style={{
      width: '100vw', height: '100vh', overflow: 'auto', position: 'relative',
      fontFamily: "'JetBrains Mono', monospace", background: '#fafafa',
      backgroundImage: 'radial-gradient(circle, #e0e0e0 0.8px, transparent 0.8px)', backgroundSize: '20px 20px',
    }}>
      <div style={{ position: 'fixed', top: 16, left: 20, zIndex: 100, fontSize: '10px', fontWeight: 700, letterSpacing: '3px', color: '#ccc' }}>TRAIN MY OWN GPT</div>
      <a href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95" target="_blank" rel="noopener noreferrer"
        style={{ position: 'fixed', top: 16, right: 20, zIndex: 100, fontSize: '9px', letterSpacing: '1px', color: '#ccc', textDecoration: 'none', cursor: 'pointer' }}
        onMouseEnter={(e) => e.currentTarget.style.color = '#999'}
        onMouseLeave={(e) => e.currentTarget.style.color = '#ccc'}
      >based on karpathy's microgpt · runs in your browser</a>

      <Wires positions={positions} wires={WIRES} tick={posTick} />

      {/* 01 DATASET */}
      <Node x={offset} y={80} w={280} title="Dataset" active={!!dataset} accent="#2563eb" id="dataset" onPosChange={updatePos}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <select value={selectedPreset} onChange={(e) => loadPreset(e.target.value)}
            style={{ padding: '6px 8px', fontSize: '11px', fontFamily: 'inherit', border: '1px solid #e0e0e0', borderRadius: '3px', background: '#fafafa', color: '#555', cursor: 'pointer', outline: 'none' }}>
            <option value="">choose dataset...</option>
            {Object.entries(PRESETS).map(([k, v]) => <option key={k} value={k}>{v.label} — {v.desc}</option>)}
          </select>
          <input type="file" accept=".txt,.csv" id="fileUpload" style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]; if (!file) return;
              const reader = new FileReader();
              reader.onload = (ev) => {
                setSelectedPreset('custom'); setModelReady(false); setLossHistory([]); setStepTimes([]); setFinalSamples([]); setGenResults([]);
                workerRef.current?.postMessage({ type: 'load_dataset', data: { text: ev.target.result } });
              };
              reader.readAsText(file);
            }} />
          <div onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = '#999'; }} onDragLeave={(e) => { e.currentTarget.style.borderColor = '#d4d4d4'; }} onDrop={handleFileDrop}
            onClick={() => document.getElementById('fileUpload')?.click()}
            style={{ border: '1.5px dashed #d4d4d4', borderRadius: '3px', padding: '14px', textAlign: 'center', color: '#bbb', fontSize: '10px', cursor: 'pointer' }}>
            drop or click to upload .txt<div style={{ fontSize: '8px', color: '#ddd', marginTop: '3px' }}>one entry per line</div>
          </div>
          {dataset && <div style={{ fontSize: '10px', color: '#888', lineHeight: '16px' }}>
            <span style={{ color: '#333', fontWeight: 600 }}>{dataset.numDocs.toLocaleString()}</span> entries · vocab <span style={{ color: '#333', fontWeight: 600 }}>{dataset.vocabSize}</span> chars
            <div style={{ marginTop: '4px', color: '#bbb', fontSize: '9px' }}>{dataset.sampleDocs.slice(0, 6).join(', ')}...</div>
          </div>}
        </div>
      </Node>

      {/* 02 TOKENIZER */}
      <Node x={offset} y={380} w={280} title="Tokenizer" active={!!dataset} accent="#8b5cf6" id="tokenizer" onPosChange={updatePos}>
        {dataset ? <div style={{ color: '#888', lineHeight: '16px' }}>
          <div style={{ marginBottom: '6px' }}>character-level · vocab <span style={{ fontWeight: 700, color: '#333' }}>{dataset.vocabSize}</span></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px', marginBottom: '6px' }}>
            {['BOS', ...dataset.chars].slice(0, 30).map((ch, i) => {
              const display = ch === ' ' ? '␣' : ch === '\t' ? '⇥' : ch === "'" ? "'" : ch || '?';
              return <span key={i} style={{ padding: '1px 4px', fontSize: '8px', background: i < 1 ? '#fef3c7' : '#f5f5f5', border: '1px solid ' + (i < 1 ? '#fde68a' : '#e8e8e8'), borderRadius: '2px', color: i < 1 ? '#92400e' : '#888', fontWeight: i < 1 ? 600 : 400, minWidth: '14px', textAlign: 'center', display: 'inline-block' }}>{display}</span>;
            })}
          </div>
          <div style={{ fontSize: '9px', color: '#bbb' }}>&quot;{dataset.sampleDocs[0]}&quot; → [BOS, {dataset.sampleDocs[0]?.split('').join(', ')}, BOS]</div>
        </div> : <div style={{ color: '#ddd', fontStyle: 'italic' }}>waiting for dataset...</div>}
      </Node>

      {/* 03 ARCHITECTURE */}
      <Node x={offset + 310} y={80} w={300} title="Architecture" active={!!dataset} accent="#f59e0b" id="config" onPosChange={updatePos}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {[
            { key: 'n_embd', label: 'embedding dim', min: 4, max: 128, step: 4 },
            { key: 'n_head', label: 'attention heads', min: 1, max: 16, step: 1 },
            { key: 'n_layer', label: 'layers', min: 1, max: 8, step: 1 },
            { key: 'block_size', label: 'context window', min: 4, max: 64, step: 4 },
            { key: 'num_steps', label: 'training steps', min: 50, max: 3000, step: 50 },
            { key: 'learning_rate', label: 'learning rate', min: 0.001, max: 0.05, step: 0.001 },
          ].map(({ key, label, min, max, step }) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '9px', color: '#999', minWidth: '85px' }}>{label}</span>
              <input type="range" min={min} max={max} step={step} value={config[key]} disabled={training}
                onChange={(e) => setConfig(c => ({ ...c, [key]: parseFloat(e.target.value) }))} style={sliderStyle} />
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#333', minWidth: '36px', textAlign: 'right' }}>
                {key === 'learning_rate' ? config[key].toFixed(3) : config[key]}
              </span>
            </div>
          ))}
          <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '6px', display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#999' }}>
            <span>params: <span style={{ fontWeight: 700, color: '#333' }}>{paramCount.toLocaleString()}</span></span>
            <span>{fmtF(flopsPerStep)} FLOPs/step</span>
          </div>
        </div>
      </Node>

      {/* 04 TRAINING */}
      <Node x={offset + 310} y={440} w={300} title="Training" active={training || modelReady} accent="#22c55e" id="training" onPosChange={updatePos}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {!training && !modelReady && (
            <button onClick={startTraining} disabled={!dataset} style={{
              padding: '10px', fontSize: '11px', fontWeight: 600, fontFamily: 'inherit', letterSpacing: '1px',
              background: dataset ? '#333' : '#e8e8e8', color: dataset ? '#fff' : '#bbb',
              border: 'none', borderRadius: '3px', cursor: dataset ? 'pointer' : 'not-allowed',
            }}>▶ TRAIN</button>
          )}
          {training && <>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#999' }}>
              <span>step {currentStep}/{config.num_steps}</span>
              <span>loss {currentLoss.toFixed(4)}</span>
            </div>
            <div style={{ height: '2px', background: '#f0f0f0', borderRadius: '1px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: (currentStep / config.num_steps) * 100 + '%', background: '#333', transition: 'width 0.1s' }} />
            </div>
            <LossChart data={lossHistory} height={50} />
            {currentSample && <div style={{ fontSize: '10px', color: '#888' }}>sample: <span style={{ color: '#333', fontWeight: 600 }}>{currentSample}</span></div>}
            <div style={{ fontSize: '9px', color: '#bbb' }}>{stepTime}ms/step · {fmt(totalTime)} elapsed</div>
            <button onClick={stopTraining} style={{ padding: '5px', fontSize: '9px', fontFamily: 'inherit', background: 'transparent', border: '1px solid #e0e0e0', borderRadius: '3px', color: '#999', cursor: 'pointer' }}>■ STOP</button>
          </>}
          {modelReady && <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: '#22c55e', letterSpacing: '2px', marginBottom: '4px' }}>✓ GPT READY</div>
            <div style={{ fontSize: '9px', color: '#bbb', marginBottom: '8px' }}>loss {currentLoss.toFixed(4)} · {config.num_steps} steps · {fmt(totalTime)}</div>
            <div style={{ display: 'flex', gap: '6px', justifyContent: 'center' }}>
              <button onClick={startTraining} style={{ padding: '5px 10px', fontSize: '9px', fontFamily: 'inherit', background: 'transparent', border: '1px solid #e0e0e0', borderRadius: '3px', color: '#999', cursor: 'pointer' }}>RETRAIN</button>
            </div>
          </div>}
        </div>
      </Node>

      {/* 05 COMPUTE (was metrics - now shows compute/energy, not duplicate loss) */}
      <Node x={offset + 640} y={80} w={310} title="Metrics" active={lossHistory.length > 0} accent="#ec4899" id="metrics" onPosChange={updatePos}>
        {lossHistory.length > 0 ? <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
            {[
              { label: 'LOSS', value: currentLoss.toFixed(4) },
              { label: 'PERPLEXITY', value: Math.exp(currentLoss).toFixed(1) },
              { label: 'TOTAL FLOPS', value: fmtF(totalFlops) },
              { label: 'THROUGHPUT', value: (stepTimes.length && stepTimes[stepTimes.length-1] > 0) ? fmtF(Math.round(flopsPerStep / (stepTimes[stepTimes.length-1] / 1000))) + '/s' : '—' },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#f8f8f8', borderRadius: '3px', padding: '5px 7px' }}>
                <div style={{ fontSize: '7px', color: '#bbb', letterSpacing: '1px' }}>{label}</div>
                <div style={{ fontWeight: 700, color: '#333' }}>{value}</div>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: '8px', color: '#bbb', letterSpacing: '1px', marginBottom: '2px' }}>STEP TIME (ms)</div>
            <StepTimeChart data={stepTimes.slice(-100)} height={35} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8px', color: '#ccc', marginTop: '2px' }}>
              <span>avg {stepTimes.length ? Math.round(stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length) : 0}ms</span>
              <span>total {fmt(totalTime)}</span>
            </div>
          </div>
          {/* Scale comparison */}
          <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '6px', fontSize: '8px', color: '#bbb', lineHeight: '14px' }}>
            <div>your GPT: <span style={{ color: '#555' }}>{paramCount.toLocaleString()} params · {fmtF(totalFlops)} FLOPs</span></div>
            <div>GPT-4 est: <span style={{ color: '#555' }}>~1.8T params · ~10²⁵ FLOPs</span></div>
          </div>
          {runHistory.length > 0 && <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '4px' }}>
            <div style={{ fontSize: '8px', color: '#bbb', letterSpacing: '1px', marginBottom: '3px' }}>SAVED RUNS</div>
            {runHistory.map((run, i) => (
              <div key={run.id} style={{ fontSize: '9px', color: '#888', padding: '2px 0', display: 'flex', justifyContent: 'space-between' }}>
                <span>#{i + 1} {run.config.n_layer}L/{run.config.n_embd}d</span>
                <span style={{ color: '#555' }}>loss {run.finalLoss.toFixed(3)} · {fmt(run.totalTime)}</span>
              </div>
            ))}
          </div>}
        </div> : <div style={{ color: '#ddd', fontStyle: 'italic' }}>waiting for training...</div>}
      </Node>

      {/* 06 GENERATE */}
      <Node x={offset + 640} y={440} w={310} title="Generate" active={modelReady} accent="#06b6d4" id="generate" onPosChange={updatePos}>
        {modelReady ? <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', gap: '4px' }}>
            <input type="text" value={genPrompt} onChange={(e) => setGenPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doGenerate()}
              placeholder="type a beginning..."
              style={{ flex: 1, padding: '6px 8px', fontSize: '11px', fontFamily: 'inherit', border: '1px solid #e0e0e0', borderRadius: '3px', outline: 'none', color: '#333' }} />
            <button onClick={doGenerate} disabled={generating} style={{
              padding: '6px 10px', fontSize: '10px', fontWeight: 600, fontFamily: 'inherit',
              background: '#333', color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
            }}>{generating ? '...' : '→'}</button>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px', color: '#999' }}>
            <span>temp</span>
            <input type="range" min="0.1" max="1.5" step="0.1" value={genTemp}
              onChange={(e) => setGenTemp(parseFloat(e.target.value))} style={sliderStyle} />
            <span style={{ fontWeight: 600, color: '#333' }}>{genTemp.toFixed(1)}</span>
          </div>
          {genResults.length === 0 && <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
            {['', 'ka', 'em', 'ch', 'j', 'al'].map((pr) => (
              <button key={pr} onClick={() => { setGenPrompt(pr); setTimeout(doGenerate, 50); }}
                style={{ padding: '3px 6px', fontSize: '8px', fontFamily: 'inherit', background: '#f5f5f5', border: '1px solid #e8e8e8', borderRadius: '2px', color: '#888', cursor: 'pointer' }}>
                {pr ? '"' + pr + '..."' : 'random'}
              </button>
            ))}
          </div>}
          {genResults.length > 0 && <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {genResults.map((s, i) => (
              <div key={i} style={{
                padding: '4px 6px', fontSize: '11px',
                background: i === 0 ? '#f0f7ff' : '#f8f8f8', borderRadius: '2px', color: '#333',
                fontWeight: i === 0 ? 600 : 400, border: i === 0 ? '1px solid #dbeafe' : '1px solid transparent',
              }}>
                {genPrompt && <span style={{ color: '#2563eb' }}>{genPrompt}</span>}
                <span>{s.startsWith(genPrompt) ? s.slice(genPrompt.length) : s}</span>
              </div>
            ))}
          </div>}
          {genResults.length === 0 && finalSamples.length > 0 && <div>
            <div style={{ fontSize: '8px', color: '#bbb', letterSpacing: '1px', marginBottom: '4px' }}>TRAINING SAMPLES</div>
            <div style={{ fontSize: '10px', color: '#666', lineHeight: '18px' }}>{finalSamples.slice(0, 6).join(' · ')}</div>
          </div>}
        </div> : <div style={{ color: '#ddd', fontStyle: 'italic' }}>waiting for model...</div>}
      </Node>

      <div style={{ position: 'fixed', bottom: 16, left: 20, zIndex: 100, fontSize: '8px', color: '#ccc', letterSpacing: '0.5px' }}>
        100% in-browser · no data leaves your device · {paramCount.toLocaleString()} parameters
      </div>

      <div style={{ position: 'fixed', bottom: 16, right: 20, zIndex: 100, display: 'flex', gap: '12px', alignItems: 'center' }}>
        <span style={{ fontSize: '8px', color: '#ddd', letterSpacing: '0.5px' }}>built by jay</span>
        <a href="https://github.com/jayyvk/trainmyowngpt" target="_blank" rel="noopener noreferrer" style={{ color: '#ccc', transition: 'color 0.2s' }}
          onMouseEnter={(e) => e.currentTarget.style.color = '#666'} onMouseLeave={(e) => e.currentTarget.style.color = '#ccc'}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        </a>
        <a href="https://www.linkedin.com/in/jayvk/" target="_blank" rel="noopener noreferrer" style={{ color: '#ccc', transition: 'color 0.2s' }}
          onMouseEnter={(e) => e.currentTarget.style.color = '#0a66c2'} onMouseLeave={(e) => e.currentTarget.style.color = '#ccc'}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
        </a>
        <a href="https://x.com/jaynotai" target="_blank" rel="noopener noreferrer" style={{ color: '#ccc', transition: 'color 0.2s' }}
          onMouseEnter={(e) => e.currentTarget.style.color = '#000'} onMouseLeave={(e) => e.currentTarget.style.color = '#ccc'}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
        </a>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        * { box-sizing: border-box; }
        input[type=range] { -webkit-appearance: none; background: #e8e8e8; border-radius: 2px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 50%; background: #999; cursor: pointer; }
        ::selection { background: #dbeafe; }
      `}} />
    </div>
  );
}
