import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, AreaChart, Area } from 'recharts';
import { Calculator, TrendingUp, Settings, Info, RefreshCw } from 'lucide-react';

const SolowModel = () => {
  // --- State ---
  const [alpha, setAlpha] = useState(0.50); // Output elasticity
  const [s, setS] = useState(0.20);         // Savings rate
  const [delta, setDelta] = useState(0.06); // Depreciation rate
  const [kCurrent, setKCurrent] = useState(50); // Current capital stock
  const [activeTab, setActiveTab] = useState('diagram'); // 'diagram' or 'time-series'

  // --- Calculations ---
  
  // 1. Steady State Calculation: s * k^alpha = delta * k  =>  k* = (s/delta)^(1/(1-alpha))
  const steadyStateK = Math.pow(s / delta, 1 / (1 - alpha));
  const steadyStateY = Math.pow(steadyStateK, alpha);
  const goldenRuleS = alpha; // The savings rate that maximizes consumption in steady state

  // 2. Current State Values
  const yCurrent = Math.pow(kCurrent, alpha);
  const investmentCurrent = s * yCurrent;
  const depreciationCurrent = delta * kCurrent;
  const changeInK = investmentCurrent - depreciationCurrent;
  const kNext = kCurrent + changeInK;

  // --- Data Generation ---

  // 1. Phase Diagram Data (k vs y, i, dep)
  const phaseData = useMemo(() => {
    const data = [];
    // Determine range: go up to at least 1.5x the max of (current K or steady state K)
    const maxK = Math.max(kCurrent, steadyStateK) * 1.5;
    const steps = 50;
    
    for (let i = 0; i <= steps; i++) {
      const k = (maxK / steps) * i;
      // Avoid k=0 for log/power issues if alpha < 0 (though alpha usually 0-1)
      const valK = k === 0 ? 0.01 : k; 
      
      data.push({
        k: parseFloat(k.toFixed(1)),
        output: parseFloat(Math.pow(valK, alpha).toFixed(2)),
        investment: parseFloat((s * Math.pow(valK, alpha)).toFixed(2)),
        depreciation: parseFloat((delta * valK).toFixed(2)),
      });
    }
    return data;
  }, [alpha, s, delta, kCurrent, steadyStateK]);

  // 2. Time Series Data (Convergence over time)
  const timeSeriesData = useMemo(() => {
    const data = [];
    let k = kCurrent;
    const periods = 50;

    for (let t = 0; t <= periods; t++) {
      const y = Math.pow(k, alpha);
      const i = s * y;
      const dep = delta * k;
      const c = y - i; // Consumption
      
      data.push({
        t,
        capital: parseFloat(k.toFixed(2)),
        output: parseFloat(y.toFixed(2)),
        consumption: parseFloat(c.toFixed(2))
      });

      // Update k for next period
      k = k + i - dep;
    }
    return data;
  }, [alpha, s, delta, kCurrent]);

  // --- Formatters ---
  const fmt = (num) => num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-700 pb-4">
          <div>
            <h1 className="text-3xl font-bold text-blue-400 flex items-center gap-3">
              <TrendingUp className="w-8 h-8" />
              Interactive Solow Growth Model
            </h1>
            <p className="text-slate-400 mt-1">Simulate capital accumulation and steady-state convergence</p>
          </div>
          
          {/* Simple Instructions */}
          <div className="mt-4 md:mt-0 flex gap-2 bg-slate-800 p-2 rounded-lg text-xs text-slate-300">
             <Info size={16} className="text-blue-400" />
             <span>
               Adjust sliders to see how <b>Savings (s)</b> and <b>Depreciation (δ)</b> shift the steady state.
             </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Controls */}
          <div className="lg:col-span-3 space-y-6 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
            <div className="flex items-center gap-2 text-xl font-semibold text-white mb-4">
              <Settings className="w-5 h-5" /> Parameters
            </div>

            {/* Slider Component */}
            <div className="space-y-6">
              <ControlSlider 
                label="Output Elasticity (α)" 
                val={alpha} set={setAlpha} min={0.1} max={0.9} step={0.01} 
                desc="Returns to capital. Higher α means capital is more productive."
              />
              <ControlSlider 
                label="Savings Rate (s)" 
                val={s} set={setS} min={0.01} max={0.90} step={0.01} 
                desc="Fraction of output invested."
                mark={goldenRuleS} markLabel="Golden Rule"
              />
              <ControlSlider 
                label="Depreciation Rate (δ)" 
                val={delta} set={setDelta} min={0.01} max={0.20} step={0.005} 
                desc="Rate at which capital wears out."
              />
              <div className="pt-4 border-t border-slate-700">
                <ControlSlider 
                  label="Current Capital (k_t)" 
                  val={kCurrent} set={setKCurrent} min={1} max={200} step={1} 
                  desc="Starting capital stock."
                />
              </div>
            </div>

            {/* Reset Button */}
            <button 
              onClick={() => { setAlpha(0.5); setS(0.2); setDelta(0.06); setKCurrent(50); }}
              className="w-full mt-4 py-2 px-4 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg flex items-center justify-center gap-2 transition-colors text-sm"
            >
              <RefreshCw size={14} /> Reset to Defaults
            </button>
          </div>

          {/* Middle Column: Charts */}
          <div className="lg:col-span-6 space-y-4">
            
            {/* Tabs */}
            <div className="flex gap-2 bg-slate-800 p-1 rounded-lg w-fit mb-2">
              <TabButton active={activeTab === 'diagram'} onClick={() => setActiveTab('diagram')}>
                Phase Diagram (k vs y)
              </TabButton>
              <TabButton active={activeTab === 'time-series'} onClick={() => setActiveTab('time-series')}>
                Convergence over Time
              </TabButton>
            </div>

            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-xl h-[500px]">
              <ResponsiveContainer width="100%" height="100%">
                {activeTab === 'diagram' ? (
                  <LineChart data={phaseData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="k" 
                      label={{ value: 'Capital Stock (k)', position: 'bottom', fill: '#94a3b8' }} 
                      stroke="#94a3b8"
                      type="number"
                      domain={[0, 'auto']}
                    />
                    <YAxis 
                      stroke="#94a3b8"
                      label={{ value: 'Output / Investment', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    
                    {/* The Curves */}
                    <Line type="monotone" dataKey="output" name="Output f(k)" stroke="#3b82f6" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="investment" name="Investment s*f(k)" stroke="#10b981" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="depreciation" name="Depreciation δ*k" stroke="#ef4444" strokeWidth={3} dot={false} />
                    
                    {/* Current K Line */}
                    <ReferenceLine x={kCurrent} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'Current k', position: 'insideTopRight', fill: '#f59e0b' }} />
                    
                    {/* Steady State Line */}
                    <ReferenceLine x={steadyStateK} stroke="#a855f7" strokeDasharray="3 3" label={{ value: 'Steady State k*', position: 'insideTopRight', fill: '#a855f7' }} />

                  </LineChart>
                ) : (
                  <AreaChart data={timeSeriesData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                    <defs>
                      <linearGradient id="colorOutput" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="t" stroke="#94a3b8" label={{ value: 'Time Periods', position: 'bottom', fill: '#94a3b8' }} />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }} />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Area type="monotone" dataKey="output" stroke="#3b82f6" fillOpacity={1} fill="url(#colorOutput)" name="Output (y)" />
                    <Area type="monotone" dataKey="consumption" stroke="#10b981" fill="transparent" name="Consumption (c)" />
                    {/* Steady State Ref Line for Output */}
                    <ReferenceLine y={steadyStateY} stroke="#a855f7" strokeDasharray="3 3" label="Steady State Output" />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>

            {/* Legend / Key Analysis */}
            <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 text-sm text-slate-300">
              <p>
                <span className="text-purple-400 font-bold">Steady State:</span> The economy converges to 
                <span className="font-mono text-white ml-1">k* = {fmt(steadyStateK)}</span>. 
                Currently at <span className="font-mono text-white">k = {kCurrent}</span>, 
                capital is <span className={changeInK > 0 ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
                  {changeInK > 0 ? "growing" : "shrinking"}
                </span>.
              </p>
            </div>
          </div>

          {/* Right Column: Math Breakdown */}
          <div className="lg:col-span-3 space-y-6">
            
            {/* Dynamic Math Box */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
              <div className="flex items-center gap-2 text-xl font-semibold text-white mb-4">
                <Calculator className="w-5 h-5" /> Math Engine
              </div>
              
              <div className="space-y-6 text-slate-300 text-sm font-mono">
                
                {/* Step 1: Output */}
                <div className="bg-slate-900/50 p-3 rounded border-l-4 border-blue-500">
                  <p className="text-xs text-blue-400 mb-1">1. Output Production</p>
                  <p>y = k<sup>α</sup></p>
                  <p>= {kCurrent}<sup>{alpha}</sup></p>
                  <p className="text-white text-lg font-bold">= {fmt(yCurrent)}</p>
                </div>

                {/* Step 2: Savings */}
                <div className="bg-slate-900/50 p-3 rounded border-l-4 border-green-500">
                  <p className="text-xs text-green-400 mb-1">2. Savings (Investment)</p>
                  <p>i = s × y</p>
                  <p>= {s} × {fmt(yCurrent)}</p>
                  <p className="text-white text-lg font-bold">= {fmt(investmentCurrent)}</p>
                </div>

                {/* Step 3: Depreciation */}
                <div className="bg-slate-900/50 p-3 rounded border-l-4 border-red-500">
                  <p className="text-xs text-red-400 mb-1">3. Depreciation</p>
                  <p>δk = δ × k</p>
                  <p>= {delta} × {kCurrent}</p>
                  <p className="text-white text-lg font-bold">= {fmt(depreciationCurrent)}</p>
                </div>

                {/* Step 4: Accumulation */}
                <div className="bg-slate-900/50 p-3 rounded border-l-4 border-yellow-500">
                  <p className="text-xs text-yellow-400 mb-1">4. Next Period (k_t+1)</p>
                  <p>k' = k + i - δk</p>
                  <p>= {kCurrent} + {fmt(investmentCurrent)} - {fmt(depreciationCurrent)}</p>
                  <p className="text-white text-xl font-bold">= {fmt(kNext)}</p>
                  <p className={`text-xs mt-1 ${changeInK > 0 ? 'text-green-400' : 'text-red-400'}`}>
                     (Net Change: {changeInK > 0 ? '+' : ''}{fmt(changeInK)})
                  </p>
                </div>

              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

// --- Helper Components ---

const ControlSlider = ({ label, val, set, min, max, step, desc, mark, markLabel }) => (
  <div className="space-y-2">
    <div className="flex justify-between items-baseline">
      <label className="text-sm font-bold text-slate-200">{label}</label>
      <span className="font-mono text-blue-400">{val.toFixed(2)}</span>
    </div>
    <div className="relative">
        <input 
        type="range" 
        min={min} max={max} step={step} 
        value={val} 
        onChange={(e) => set(parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400"
        />
        {mark && (
            <div 
                className="absolute top-4 text-[10px] text-yellow-500 flex flex-col items-center -ml-2 pointer-events-none"
                style={{ left: `${((mark - min) / (max - min)) * 100}%` }}
            >
                <span>▲</span>
                <span>{markLabel}</span>
            </div>
        )}
    </div>
    <p className="text-xs text-slate-500">{desc}</p>
  </div>
);

const TabButton = ({ active, children, onClick }) => (
  <button 
    onClick={onClick}
    className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
      active 
      ? 'bg-blue-600 text-white shadow-lg' 
      : 'text-slate-400 hover:text-white hover:bg-slate-700'
    }`}
  >
    {children}
  </button>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-lg">
        <p className="text-slate-400 text-xs mb-2">Capital (k): {label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color }} className="text-sm font-mono">
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default SolowModel;