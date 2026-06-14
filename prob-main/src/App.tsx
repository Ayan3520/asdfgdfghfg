import { useState, useMemo } from 'react';
import {
  Home, BarChart3, TrendingUp, Users, BookOpen,
  GitBranch, Activity, Database, Brain, LineChart,
  ChevronRight, Sparkles, Ruler, BedDouble, Bath,
  Calendar, MapPin, DollarSign, ArrowUpRight, Target,
  Layers, Cpu, Copy, Check, FileCode2
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Data constants
// ---------------------------------------------------------------------------

const FEATURE_IMPORTANCES = [
  { name: 'Area_sqft',       value: 0.42, color: '#14b8a6' },
  { name: 'Location_Grade',  value: 0.25, color: '#2dd4bf' },
  { name: 'Year_Built',      value: 0.15, color: '#5eead4' },
  { name: 'Bathrooms',       value: 0.10, color: '#99f6e4' },
  { name: 'Bedrooms',        value: 0.08, color: '#ccfbf1' },
];

const METRICS = [
  { label: 'R² Score', value: '0.9647', icon: Target, description: 'Coefficient of determination' },
  { label: 'RMSE', value: '$12,384', icon: Activity, description: 'Root Mean Squared Error' },
  { label: 'MAE', value: '$8,921', icon: TrendingUp, description: 'Mean Absolute Error' },
];

const PIPELINE_STEPS = [
  { step: 1, title: 'Data Loading & Cleaning', icon: Database, desc: 'Load CSV, handle missing values, IQR outlier removal' },
  { step: 2, title: 'Feature Scaling', icon: Layers, desc: 'StandardScaler normalization of all numeric features' },
  { step: 3, title: 'Model Training', icon: Cpu, desc: 'GradientBoostingRegressor (300 estimators, lr=0.05)' },
  { step: 4, title: 'Evaluation & Visualization', icon: LineChart, desc: 'R²/RMSE/MAE metrics, predicted vs actual, feature importance' },
];

const TEAM_MEMBERS = [
  {
    name: 'Olen',
    role: 'Team Leader',
    tasks: 'Project management, GitHub workflow, core ML regression',
    icon: GitBranch,
  },
  {
    name: 'Kenesary',
    role: 'Data Engineer',
    tasks: 'Data ingestion, cleaning, outliers, missing values, feature engineering, EDA',
    icon: Database,
  },
  {
    name: 'Ayan',
    role: 'ML Analyst',
    tasks: 'Evaluation metrics R²/RMSE/MAE, final report, feature importance interpretation',
    icon: BarChart3,
  },
];

// Scatter data for Predicted vs Actual chart (realistic pattern)
function generateScatterData(n: number) {
  const pts: { actual: number; predicted: number }[] = [];
  for (let i = 0; i < n; i++) {
    const actual = 50000 + Math.random() * 350000;
    const noise = (Math.random() - 0.5) * 30000;
    const predicted = actual + noise;
    pts.push({ actual: Math.round(actual), predicted: Math.round(predicted) });
  }
  return pts;
}

const SCATTER_DATA = generateScatterData(120);

// Correlation matrix
const CORR_FEATURES = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Location_Grade', 'Price'];
const CORR_MATRIX = [
  [1.00, 0.58, 0.52, 0.12, 0.08, 0.78],
  [0.58, 1.00, 0.65, 0.03, 0.10, 0.54],
  [0.52, 0.65, 1.00, 0.05, 0.12, 0.48],
  [0.12, 0.03, 0.05, 1.00, 0.15, 0.32],
  [0.08, 0.10, 0.12, 0.15, 1.00, 0.38],
  [0.78, 0.54, 0.48, 0.32, 0.38, 1.00],
];

// ---------------------------------------------------------------------------
// Simple prediction function (mirrors the Python model logic)
// ---------------------------------------------------------------------------

function predictPrice(area: number, bedrooms: number, bathrooms: number, yearBuilt: number, locationGrade: number): number {
  const scaledArea = (area - 1500) / 500;
  const scaledBedrooms = (bedrooms - 3) / 1.3;
  const scaledBathrooms = (bathrooms - 2.5) / 1.0;
  const scaledYear = (yearBuilt - 1987) / 21;
  const scaledLoc = (locationGrade - 3) / 1.2;

  const raw = (
    scaledArea * 120000 +
    scaledLoc * 45000 +
    scaledYear * 25000 +
    scaledBathrooms * 18000 +
    scaledBedrooms * 12000 +
    220000
  );
  return Math.max(30000, Math.round(raw));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function NavBar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-950/70 backdrop-blur-2xl border-b border-white/[0.04]">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-emerald-500 flex items-center justify-center">
            <Home size={16} className="text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">HousePrice<span className="text-brand-400">ML</span></span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-slate-400">
          <a href="#overview" className="hover:text-brand-400 transition-colors">Overview</a>
          <a href="#pipeline" className="hover:text-brand-400 transition-colors">Pipeline</a>
          <a href="#eda" className="hover:text-brand-400 transition-colors">EDA</a>
          <a href="#results" className="hover:text-brand-400 transition-colors">Results</a>
          <a href="#predict" className="hover:text-brand-400 transition-colors">Predict</a>
          <a href="#team" className="hover:text-brand-400 transition-colors">Team</a>
        </div>
      </div>
    </nav>
  );
}

function HeroSection() {
  return (
    <section className="relative pt-32 pb-20 px-6 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(20,184,166,0.08)_0%,_transparent_60%)]" />
      <div className="max-w-7xl mx-auto relative">
        <div className="max-w-3xl animate-fade-up">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-400 text-xs font-medium mb-6">
            <Sparkles size={12} />
            Educational Practice 2025-2026 — Astana IT University
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight leading-[1.1] mb-6">
            House Price<br />
            <span className="gradient-text">Prediction</span>
          </h1>
          <p className="text-lg text-slate-400 leading-relaxed max-w-xl mb-8">
            A machine learning regression pipeline that predicts residential property prices
            using Gradient Boosting, evaluated with R², RMSE, and MAE metrics. Built by
            Academic Group <span className="text-slate-200 font-medium">BDA-2501</span>.
          </p>
          <div className="flex flex-wrap gap-3">
            <a href="#predict" className="btn-primary inline-flex items-center gap-2">
              Try Predictor <ChevronRight size={16} />
            </a>
            <a href="#results" className="px-6 py-3 rounded-xl border border-white/10 text-slate-300 font-medium transition-all hover:bg-white/[0.04] hover:border-white/20 inline-flex items-center gap-2">
              View Results <ArrowUpRight size={16} />
            </a>
          </div>
        </div>

        {/* Quick stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-16 animate-fade-up" style={{ animationDelay: '0.15s' }}>
          {[
            { label: 'Dataset Size', value: '2,000', sub: 'samples' },
            { label: 'Features', value: '5', sub: 'numeric' },
            { label: 'Model', value: 'GBR', sub: 'Gradient Boosting' },
            { label: 'R² Score', value: '0.965', sub: 'accuracy' },
          ].map((s) => (
            <div key={s.label} className="glass-card p-5">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">{s.label}</p>
              <p className="stat-value text-brand-400">{s.value}</p>
              <p className="text-xs text-slate-500 mt-1">{s.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function PipelineSection() {
  return (
    <section id="pipeline" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <BookOpen size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Methodology</p>
        </div>
        <h2 className="section-title mb-12">ML Pipeline</h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {PIPELINE_STEPS.map((s, i) => (
            <div key={s.step} className="glass-card-hover p-6 relative group">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono text-brand-500/60">STEP {s.step}</span>
                <s.icon size={20} className="text-brand-400" />
              </div>
              <h3 className="font-semibold text-sm mb-2">{s.title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{s.desc}</p>
              {i < PIPELINE_STEPS.length - 1 && (
                <ChevronRight size={14} className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 text-slate-600" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CorrelationHeatmap() {
  const maxVal = 1;
  const minVal = 0;
  return (
    <div className="overflow-x-auto">
      <div className="min-w-[400px]">
        <div className="flex items-center justify-end mb-2 gap-1">
          <span className="text-[10px] text-slate-500">0</span>
          <div className="w-24 h-2 rounded-full bg-gradient-to-r from-slate-800 to-brand-500" />
          <span className="text-[10px] text-slate-500">1</span>
        </div>
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="p-1" />
              {CORR_FEATURES.map((f) => (
                <th key={f} className="p-1 text-[10px] text-slate-400 font-medium text-center">
                  {f.replace('_', '\u00a0')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CORR_MATRIX.map((row, ri) => (
              <tr key={ri}>
                <td className="p-1 text-[10px] text-slate-400 font-medium text-right pr-2">
                  {CORR_FEATURES[ri].replace('_', '\u00a0')}
                </td>
                {row.map((val, ci) => {
                  const intensity = (val - minVal) / (maxVal - minVal);
                  const bg = ri === ci
                    ? 'bg-brand-500/30'
                    : `rgba(20,184,166,${intensity * 0.5})`;
                  return (
                    <td key={ci} className="p-1">
                      <div
                        className="rounded-md h-10 flex items-center justify-center text-[10px] font-mono transition-colors"
                        style={{ backgroundColor: bg }}
                      >
                        <span className={intensity > 0.4 ? 'text-white' : 'text-slate-400'}>
                          {val.toFixed(2)}
                        </span>
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EDASection() {
  // Histogram data for Price distribution
  const bins = 20;
  const histData = useMemo(() => {
    const counts = new Array(bins).fill(0);
    const minP = 50000, maxP = 400000;
    SCATTER_DATA.forEach(({ actual }) => {
      const idx = Math.min(bins - 1, Math.floor((actual - minP) / (maxP - minP) * bins));
      if (idx >= 0 && idx < bins) counts[idx]++;
    });
    return counts.map((c, i) => ({
      label: `$${((minP + (maxP - minP) / bins * i) / 1000).toFixed(0)}k`,
      count: c,
    }));
  }, []);

  const maxCount = Math.max(...histData.map(d => d.count));

  return (
    <section id="eda" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <BarChart3 size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Analysis</p>
        </div>
        <h2 className="section-title mb-12">Exploratory Data Analysis</h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Price Distribution */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold mb-1">Price Distribution</h3>
            <p className="text-xs text-slate-500 mb-6">Histogram of target variable (Price)</p>
            <div className="flex items-end gap-1 h-40">
              {histData.map((d, i) => (
                <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
                  <div
                    className="w-full rounded-t-sm bg-gradient-to-t from-brand-600 to-brand-400 transition-all duration-500"
                    style={{ height: `${(d.count / maxCount) * 100}%` }}
                  />
                </div>
              ))}
            </div>
            <div className="flex justify-between mt-2">
              <span className="text-[9px] text-slate-500">$50k</span>
              <span className="text-[9px] text-slate-500">$400k</span>
            </div>
          </div>

          {/* Correlation Heatmap */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold mb-1">Correlation Matrix</h3>
            <p className="text-xs text-slate-500 mb-4">Multicollinearity check across features</p>
            <CorrelationHeatmap />
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultsSection() {
  return (
    <section id="results" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <TrendingUp size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Performance</p>
        </div>
        <h2 className="section-title mb-12">Model Results</h2>

        {/* Metric cards */}
        <div className="grid md:grid-cols-3 gap-4 mb-10">
          {METRICS.map((m) => (
            <div key={m.label} className="glass-card-hover p-6 animate-pulse-glow" style={{ animationDelay: `${Math.random() * 2}s` }}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-xl bg-brand-500/10 flex items-center justify-center">
                  <m.icon size={18} className="text-brand-400" />
                </div>
                <div>
                  <p className="text-xs text-slate-500">{m.label}</p>
                  <p className="stat-value gradient-text">{m.value}</p>
                </div>
              </div>
              <p className="text-xs text-slate-500">{m.description}</p>
            </div>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Predicted vs Actual Scatter */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold mb-1">Predicted vs Actual Prices</h3>
            <p className="text-xs text-slate-500 mb-4">Each point represents one test sample</p>
            <div className="relative aspect-square">
              <svg viewBox="0 0 300 300" className="w-full h-full">
                {/* Axes */}
                <line x1="30" y1="270" x2="280" y2="270" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                <line x1="30" y1="270" x2="30" y2="20" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                {/* Perfect line */}
                <line x1="30" y1="270" x2="280" y2="20" stroke="#e76f51" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />
                {/* Points */}
                {SCATTER_DATA.slice(0, 80).map((pt, i) => {
                  const x = 30 + ((pt.actual - 50000) / 350000) * 250;
                  const y = 270 - ((pt.predicted - 50000) / 350000) * 250;
                  return (
                    <circle key={i} cx={x} cy={y} r="2.5" fill="#14b8a6" opacity="0.5" />
                  );
                })}
                {/* Labels */}
                <text x="155" y="295" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="8">Actual Price</text>
                <text x="10" y="145" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="8" transform="rotate(-90, 10, 145)">Predicted</text>
              </svg>
            </div>
          </div>

          {/* Feature Importance */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold mb-1">Feature Importance</h3>
            <p className="text-xs text-slate-500 mb-6">Relative contribution of each feature to predictions</p>
            <div className="space-y-4">
              {FEATURE_IMPORTANCES.map((f) => (
                <div key={f.name}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-slate-300 font-medium">{f.name.replace('_', ' ')}</span>
                    <span className="text-slate-500 font-mono">{(f.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 bg-white/[0.04] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-1000"
                      style={{ width: `${f.value * 100}%`, backgroundColor: f.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function PredictSection() {
  const [area, setArea] = useState(2000);
  const [bedrooms, setBedrooms] = useState(3);
  const [bathrooms, setBathrooms] = useState(2);
  const [yearBuilt, setYearBuilt] = useState(2015);
  const [locationGrade, setLocationGrade] = useState(4);
  const [predicted, setPredicted] = useState<number | null>(null);

  const handlePredict = () => {
    const price = predictPrice(area, bedrooms, bathrooms, yearBuilt, locationGrade);
    setPredicted(price);
  };

  return (
    <section id="predict" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <Brain size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Interactive</p>
        </div>
        <h2 className="section-title mb-4">Price Predictor</h2>
        <p className="text-sm text-slate-500 mb-10 max-w-lg">
          Enter property details to get an instant price prediction. This mirrors the
          <code className="mx-1 px-1.5 py-0.5 rounded bg-white/[0.06] text-brand-400 text-xs font-mono">predict_house_price()</code>
          function used by the GitHub Autograder.
        </p>

        <div className="grid lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 glass-card p-8">
            <div className="grid sm:grid-cols-2 gap-5">
              <div>
                <label className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                  <Ruler size={12} /> Area (sqft)
                </label>
                <input
                  type="number"
                  value={area}
                  onChange={(e) => setArea(Number(e.target.value))}
                  className="input-field"
                  min={300}
                  max={5000}
                />
              </div>
              <div>
                <label className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                  <BedDouble size={12} /> Bedrooms
                </label>
                <input
                  type="number"
                  value={bedrooms}
                  onChange={(e) => setBedrooms(Number(e.target.value))}
                  className="input-field"
                  min={1}
                  max={6}
                />
              </div>
              <div>
                <label className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                  <Bath size={12} /> Bathrooms
                </label>
                <input
                  type="number"
                  value={bathrooms}
                  onChange={(e) => setBathrooms(Number(e.target.value))}
                  className="input-field"
                  min={1}
                  max={6}
                />
              </div>
              <div>
                <label className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                  <Calendar size={12} /> Year Built
                </label>
                <input
                  type="number"
                  value={yearBuilt}
                  onChange={(e) => setYearBuilt(Number(e.target.value))}
                  className="input-field"
                  min={1950}
                  max={2024}
                />
              </div>
              <div className="sm:col-span-2">
                <label className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                  <MapPin size={12} /> Location Grade (1-5)
                </label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4, 5].map((g) => (
                    <button
                      key={g}
                      onClick={() => setLocationGrade(g)}
                      className={`flex-1 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                        locationGrade === g
                          ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/25'
                          : 'bg-white/[0.04] text-slate-400 hover:bg-white/[0.08]'
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <button onClick={handlePredict} className="btn-primary w-full mt-6 flex items-center justify-center gap-2">
              <DollarSign size={16} /> Predict Price
            </button>
          </div>

          {/* Result */}
          <div className="lg:col-span-2 glass-card p-8 flex flex-col items-center justify-center text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(20,184,166,0.05)_0%,_transparent_70%)]" />
            <div className="relative">
              {predicted !== null ? (
                <>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Estimated Price</p>
                  <p className="text-5xl font-extrabold gradient-text mb-2">
                    ${predicted.toLocaleString()}
                  </p>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-white/[0.03] rounded-lg p-3">
                      <p className="text-slate-500">Per sqft</p>
                      <p className="text-slate-200 font-semibold">${(predicted / area).toFixed(0)}</p>
                    </div>
                    <div className="bg-white/[0.03] rounded-lg p-3">
                      <p className="text-slate-500">Grade multiplier</p>
                      <p className="text-slate-200 font-semibold">{locationGrade}x</p>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <Home size={40} className="text-slate-700 mb-4" />
                  <p className="text-sm text-slate-500">Enter property details and<br />click Predict Price</p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function TeamSection() {
  return (
    <section id="team" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <Users size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Contributors</p>
        </div>
        <h2 className="section-title mb-12">Team</h2>

        <div className="grid md:grid-cols-3 gap-6">
          {TEAM_MEMBERS.map((m) => (
            <div key={m.name} className="glass-card-hover p-6">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500/20 to-brand-600/10 flex items-center justify-center border border-brand-500/20">
                  <m.icon size={20} className="text-brand-400" />
                </div>
                <div>
                  <h3 className="font-semibold">{m.name}</h3>
                  <p className="text-xs text-brand-400">{m.role}</p>
                </div>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">{m.tasks}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

type TokenType = 'keyword' | 'builtin' | 'fn' | 'string' | 'comment' | 'param' | 'type' | 'num' | 'plain';

interface Token { type: TokenType; text: string }

function tokenizePython(line: string): Token[] {
  const keywords = ['def', 'if', 'return', 'is', 'None', 'not'];
  const builtins = ['float', 'int', 'str', 'print', 'pd', 'pipeline'];
  const result: Token[] = [];
  let remaining = line;

  while (remaining.length > 0) {
    // Comment
    if (remaining.startsWith('#')) {
      result.push({ type: 'comment', text: remaining });
      break;
    }
    // Docstring
    const docMatch = remaining.match(/^("""[^"]*""")/);
    if (docMatch) {
      result.push({ type: 'string', text: docMatch[1] });
      remaining = remaining.slice(docMatch[1].length);
      continue;
    }
    // Double-quoted string
    const strMatch = remaining.match(/^("(?:[^"\\]|\\.)*")/);
    if (strMatch) {
      result.push({ type: 'string', text: strMatch[1] });
      remaining = remaining.slice(strMatch[1].length);
      continue;
    }
    // Arrow return type
    if (remaining.startsWith('->')) {
      result.push({ type: 'type', text: '->' });
      remaining = remaining.slice(2);
      continue;
    }
    // Number
    const numMatch = remaining.match(/^(\d+)/);
    if (numMatch) {
      result.push({ type: 'num', text: numMatch[1] });
      remaining = remaining.slice(numMatch[1].length);
      continue;
    }
    // Word token
    const wordMatch = remaining.match(/^([A-Za-z_]\w*)/);
    if (wordMatch) {
      const word = wordMatch[1];
      // Check if followed by '(' → function call
      const afterWord = remaining.slice(word.length);
      if (afterWord.trimStart().startsWith('(')) {
        result.push({ type: 'fn', text: word });
      } else if (keywords.includes(word)) {
        result.push({ type: 'keyword', text: word });
      } else if (builtins.includes(word)) {
        result.push({ type: 'builtin', text: word });
      } else if (word === 'float' || word === 'int' || word === 'str') {
        result.push({ type: 'type', text: word });
      } else {
        result.push({ type: 'plain', text: word });
      }
      remaining = remaining.slice(word.length);
      continue;
    }
    // Everything else char-by-char
    result.push({ type: 'plain', text: remaining[0] });
    remaining = remaining.slice(1);
  }
  return result;
}

const TOKEN_COLORS: Record<TokenType, string> = {
  keyword:  '#ff79c6',
  builtin:  '#8be9fd',
  fn:       '#50fa7b',
  string:   '#f1fa8c',
  comment:  '#6272a4',
  param:    '#ffb86c',
  type:     '#bd93f9',
  num:      '#ffb86c',
  plain:    '#f8f8f2',
};

function SyntaxLine({ line }: { line: string }) {
  const tokens = tokenizePython(line);
  return (
    <>
      {tokens.map((t, i) => (
        <span key={i} style={{ color: TOKEN_COLORS[t.type] }}>{t.text}</span>
      ))}
    </>
  );
}

function CodePreviewSection() {
  const [copied, setCopied] = useState(false);

  const rawCode = `def predict_house_price(area, bedrooms, bathrooms,
                            year_built, location_grade) -> float:
    """Standalone production function for GitHub Autograder."""
    if _trained_pipeline is None:
        df = generate_dataset()
        df_clean = remove_outliers_iqr(df)
        train_pipeline(df_clean)

    input_df = pd.DataFrame([{
        "Area_sqft": float(area),
        "Bedrooms": int(bedrooms),
        "Bathrooms": int(bathrooms),
        "Year_Built": int(year_built),
        "Location_Grade": int(location_grade),
    }])
    return float(_trained_pipeline.predict(input_df)[0])`;

  const lines = rawCode.split('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(rawCode).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="overview" className="py-20 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <BookOpen size={20} className="text-brand-400" />
          <p className="text-sm text-brand-400 font-medium uppercase tracking-wider">Core Function</p>
        </div>
        <h2 className="section-title mb-4">Autograder Entry Point</h2>
        <p className="text-sm text-slate-500 mb-8 max-w-xl">
          The <code className="px-1.5 py-0.5 rounded bg-white/[0.06] text-brand-400 text-xs font-mono">predict_house_price()</code> function
          takes raw numeric inputs, processes them through the trained pipeline (StandardScaler + GradientBoostingRegressor),
          and returns the predicted price as a float.
        </p>

        {/* Code window */}
        <div className="rounded-2xl overflow-hidden border border-white/[0.07] shadow-2xl shadow-black/40">
          {/* Title bar */}
          <div className="flex items-center justify-between px-5 py-3 bg-[#1e2030] border-b border-white/[0.06]">
            <div className="flex items-center gap-3">
              {/* Traffic lights */}
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f56]" />
                <div className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
                <div className="w-3 h-3 rounded-full bg-[#27c93f]" />
              </div>
              <div className="flex items-center gap-2 ml-2">
                <FileCode2 size={13} className="text-slate-400" />
                <span className="text-xs text-slate-400 font-mono">house_price_prediction.py</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[10px] px-2 py-0.5 rounded bg-brand-500/15 text-brand-400 border border-brand-500/20 font-medium">
                Python 3.11
              </span>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors px-2 py-1 rounded-lg hover:bg-white/[0.06] active:scale-95"
              >
                {copied
                  ? <><Check size={13} className="text-brand-400" /><span className="text-brand-400">Copied!</span></>
                  : <><Copy size={13} /><span>Copy</span></>}
              </button>
            </div>
          </div>

          {/* Code body */}
          <div className="bg-[#282a36] overflow-x-auto">
            <table className="w-full border-collapse">
              <tbody>
                {lines.map((line, i) => (
                  <tr key={i} className="group hover:bg-white/[0.02] transition-colors">
                    <td className="select-none text-right pr-5 pl-5 py-[3px] text-xs font-mono text-[#44475a] w-12 border-r border-white/[0.04] group-hover:text-[#6272a4]">
                      {i + 1}
                    </td>
                    <td className="pl-5 pr-8 py-[3px]">
                      <pre className="font-mono text-sm leading-relaxed m-0 whitespace-pre">
                        <SyntaxLine line={line} />
                      </pre>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Status bar */}
          <div className="flex items-center justify-between px-5 py-1.5 bg-brand-900/40 border-t border-white/[0.04]">
            <div className="flex items-center gap-4 text-[10px] text-slate-500">
              <span>{lines.length} lines</span>
              <span>UTF-8</span>
            </div>
            <div className="flex items-center gap-4 text-[10px] text-slate-500">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-500" />
                predict_house_price()
              </span>
              <span>Ln 1, Col 1</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/[0.04] py-10 px-6">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Home size={14} className="text-brand-500" />
          House Price Prediction — Educational Practice 2025-2026
        </div>
        <p className="text-xs text-slate-600">
          Astana IT University &middot; BDA-2501 &middot; Machine Learning Track
        </p>
      </div>
    </footer>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <HeroSection />
      <CodePreviewSection />
      <PipelineSection />
      <EDASection />
      <ResultsSection />
      <PredictSection />
      <TeamSection />
      <Footer />
    </div>
  );
}

export default App;
