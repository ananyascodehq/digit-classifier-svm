import React, { useRef, useState, useEffect } from 'react'
import axios from 'axios'
import { 
  Eraser, Brain, Play, Sparkles, Activity, 
  ShieldCheck, Trash2, Cpu, BarChart3, Fingerprint, 
  Settings, Info, LayoutDashboard, Target, Layers, 
  TrendingUp, Grid
} from 'lucide-react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, AreaChart, Area, Cell,
  ScatterChart, Scatter, ZAxis, LineChart, Line, ComposedChart
} from 'recharts'
import './index.css'

// --- Components ---

const NavBar = ({ activePage, setActivePage }) => (
  <nav className="nav-bar">
    <div 
      className={`nav-link ${activePage === 'recognizer' ? 'active' : ''}`}
      onClick={() => setActivePage('recognizer')}
    >
      <Fingerprint size={18} /> Recognizer
    </div>
    <div 
      className={`nav-link ${activePage === 'analytics' ? 'active' : ''}`}
      onClick={() => setActivePage('analytics')}
    >
      <LayoutDashboard size={18} /> Analytics
    </div>
  </nav>
)

const Recognizer = () => {
  const canvasRef = useRef(null)
  const [ctx, setCtx] = useState(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState("System Ready")

  useEffect(() => {
    const canvas = canvasRef.current
    canvas.width = 300
    canvas.height = 300
    const context = canvas.getContext('2d')
    context.fillStyle = 'white'
    context.fillRect(0, 0, 300, 300)
    context.strokeStyle = '#0f172a'
    context.lineWidth = 18
    context.lineCap = 'round'
    context.lineJoin = 'round'
    setCtx(context)
  }, [])

  const startDrawing = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    ctx.beginPath()
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top)
    setIsDrawing(true)
  }

  const draw = (e) => {
    if (!isDrawing) return
    const rect = canvasRef.current.getBoundingClientRect()
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top)
    ctx.stroke()
  }

  const endDrawing = () => setIsDrawing(false)

  const clearCanvas = () => {
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 300, 300)
    setPrediction(null)
  }

  const handlePredict = async () => {
    setIsLoading(true)
    try {
      const canvas = canvasRef.current;
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      
      const formData = new FormData();
      formData.append('file', blob, 'digit.png');
      
      const response = await axios.post('api/predict-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Prediction Error:", error);
      setStatus("Error: Prediction failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="card-container">
      <header>
        <h1>Digit Intelligence</h1>
        <p className="subtitle">High-precision handwritten digit recognition using SVM</p>
        <div className="system-status">
          <div className="status-dot"></div>
          <span>{status}</span>
        </div>
      </header>

      <main className="main-grid">
        <section className="canvas-section">
          <div className="canvas-wrapper">
            <canvas ref={canvasRef} onMouseDown={startDrawing} onMouseMove={draw} onMouseUp={endDrawing} onMouseLeave={endDrawing} />
          </div>
          <p className="canvas-hint">Draw a single digit (0-9) inside the area above</p>
          <div className="action-buttons">
            <button onClick={clearCanvas} className="btn-secondary" title="Clear Canvas"><Eraser size={18} /> Clear</button>
            <button onClick={handlePredict} className="btn-primary" disabled={isLoading}>
              {isLoading ? <Brain className="spinning" size={18} /> : <Sparkles size={18} />} 
              {isLoading ? 'Analyzing...' : 'Recognize'}
            </button>
          </div>
        </section>

        <section className="results-section">
          <div className="prediction-card">
            <h2 className="prediction-label">Prediction Result</h2>
            <div className="prediction-value">{prediction ?? '?'}</div>
            <div className="output-info">{prediction !== null ? 'Confidence: High' : 'Awaiting Input'}</div>
            <div className="feature-list">
              <div className="feature-item"><span>Model</span><span>SVC (StandardScaler)</span></div>
              <div className="feature-item"><span>Accuracy</span><span>~98.6%</span></div>
              <div className="feature-item"><span>Kernel</span><span>RBF</span></div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

const Dashboard = () => {
  const [eda, setEda] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    Promise.all([
      axios.get('api/eda'),
      axios.get('api/analytics')
    ]).then(([edaRes, analyticsRes]) => {
      setEda(edaRes.data)
      setAnalytics(analyticsRes.data)
      setLoading(false)
    }).catch(err => {
      console.error(err)
      setLoading(false)
    })
  }, [])

  if (loading) return <div className="card-container">Loading Analytics Suite...</div>
  if (!eda || !analytics) return <div className="card-container">Failed to initialize dashboard assets.</div>

  const colors = [
    '#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#f97316', '#71717a', '#1e293b'
  ]

  return (
    <div className="analytics-container" style={{ animation: 'fadeIn 0.5s' }}>
      <header style={{ textAlign: 'left', marginBottom: '2rem' }}>
        <h1 style={{ textAlign: 'left' }}>Dataset Intelligence Page</h1>
        <p className="subtitle">Exploring Support Vector separation and error distribution metrics</p>
      </header>

      {/* Summary Metrics */}
      <div className="dashboard-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: '2rem' }}>
         <div className="chart-card" style={{ textAlign: 'center' }}>
            <Target size={24} color="#4f46e5" style={{ marginBottom: '0.4rem' }} />
            <div className="stat-label">Model Accuracy</div>
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{(analytics.accuracy * 100).toFixed(1)}%</div>
         </div>
         <div className="chart-card" style={{ textAlign: 'center' }}>
            <Layers size={24} color="#10b981" style={{ marginBottom: '0.4rem' }} />
            <div className="stat-label">Dataset Size</div>
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{eda.total_samples}</div>
         </div>
         <div className="chart-card" style={{ textAlign: 'center' }}>
            <Activity size={24} color="#f59e0b" style={{ marginBottom: '0.4rem' }} />
            <div className="stat-label">Resolution</div>
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>{eda.resolution}</div>
         </div>
         <div className="chart-card" style={{ textAlign: 'center' }}>
            <TrendingUp size={24} color="#ec4899" style={{ marginBottom: '0.4rem' }} />
            <div className="stat-label">Optimization</div>
            <div className="stat-value" style={{ fontSize: '1.4rem' }}>Kernel RBF</div>
         </div>
      </div>

      <div className="dashboard-grid" style={{ gridTemplateColumns: '2fr 1.5fr' }}>
        {/* PCA Clustering with Fitted Line */}
        <div className="chart-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
             <TrendingUp size={18} /> Support Vector Analysis
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
            Projection of 64D digit features into 2D space using PCA. Points clustered together share similar pixel distributions, while distance between clusters shows how distinct the SVM separates different digit classes.
          </p>
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis type="number" dataKey="x" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 10}} />
                <YAxis type="number" dataKey="y" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 10}} />
                <Tooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                  cursor={{ strokeDasharray: '3 3' }}
                />
                

                {/* PCA Points */}
                {Array.from({ length: 10 }).map((_, i) => (
                  <Scatter 
                    key={i} 
                    name={`Digit ${i}`} 
                    data={analytics.pca_data.filter(p => p.label === i)} 
                    fill={colors[i]} 
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix with Dimensions */}
        <div className="chart-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
             <Grid size={18} /> Error Confusion Matrix (10x10)
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '1.5rem' }}>Visualizes predictive accuracy. Diagonal values (blue) show correct classifications, while off-diagonal values (red) pinpoint exactly which digits the SVM is confusing during the testing phase.</p>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(1, 1fr)', 
            width: '100%', 
            aspectRatio: '1',
            gap: '2px',
            background: '#f1f5f9',
            padding: '2px',
            borderRadius: '8px'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gridTemplateRows: 'repeat(10, 1fr)', width: '100%', height: '100%' }}>
              {analytics.confusion_matrix.map((cell, idx) => {
                const intensity = Math.min(cell.value / 35, 1)
                const isCorrect = cell.actual === cell.predicted
                return (
                  <div 
                    key={idx}
                    title={`Actual: ${cell.actual}, Predicted: ${cell.predicted}, Freq: ${cell.value}`}
                    style={{
                      background: isCorrect ? `rgba(79, 70, 229, ${0.1 + intensity * 0.9})` : cell.value > 0 ? `rgba(239, 68, 68, ${0.2 + intensity * 0.8})` : '#fff',
                      color: isCorrect && intensity > 0.4 ? '#fff' : '#475569',
                      fontSize: '0.65rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 700
                    }}
                  >
                    {cell.value > 0 ? cell.value : ''}
                  </div>
                )
              })}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#94a3b8', marginTop: '1rem' }}>
             <span>Predicted Classes (0-9) →</span>
             <span>Actual Digit ↓</span>
          </div>
        </div>
      </div>

      <div className="dashboard-grid" style={{ marginTop: '2rem' }}>
        {/* Class Distribution (Simplified) */}
        <div className="chart-card" style={{ gridColumn: 'span 2' }}>
          <h3 style={{ fontSize: '1rem' }}>Support Count (Dataset Balance)</h3>
          <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '1rem' }}>Ensures the training set isn't biased. A balanced distribution (roughly 180 samples per digit) ensures the SVM learns unique features for all numbers equally.</p>
          <div style={{ width: '100%', height: 180 }}>
            <ResponsiveContainer>
              <BarChart data={eda.class_distribution}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="digit" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <YAxis hide />
                <Tooltip 
                  cursor={{fill: '#f8fafc'}}
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Bar dataKey="count" fill="#334155" radius={[4, 4, 0, 0]} barSize={34} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [activePage, setActivePage] = useState('recognizer')

  return (
    <>
      <NavBar activePage={activePage} setActivePage={setActivePage} />
      <div className="app-wrapper">
        {activePage === 'recognizer' ? <Recognizer /> : <Dashboard />}
      </div>
    </>
  )
}

export default App
