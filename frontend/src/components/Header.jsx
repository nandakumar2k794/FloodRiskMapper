import React from 'react';
import { Shield, CloudRain, AlertTriangle, Video, Navigation, CheckCircle2 } from 'lucide-react';

export default function Header({ rainfall, stats, activeTab, setActiveTab }) {
  return (
    <header className="header-bar">
      {/* Brand & Project Title */}
      <div className="brand-section">
        <div className="brand-icon">
          <Shield style={{ width: 20, height: 20 }} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <span className="brand-title">Chennai Flood Access Risk Mapper</span>
            <span className="brand-tag">DrainWatch AI</span>
          </div>
          <div className="brand-sub">
            Road Accessibility & CCTV Drainage Intelligence System
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          onClick={() => setActiveTab('diagnostics')}
          className={`tab-button ${activeTab === 'diagnostics' ? 'active' : ''}`}
        >
          <AlertTriangle style={{ width: 14, height: 14, color: '#f59e0b' }} />
          Road Diagnostics
        </button>

        <button
          onClick={() => setActiveTab('rainfall')}
          className={`tab-button ${activeTab === 'rainfall' ? 'active' : ''}`}
        >
          <CloudRain style={{ width: 14, height: 14, color: '#38bdf8' }} />
          Rain Simulator ({rainfall} mm)
        </button>

        <button
          onClick={() => setActiveTab('camera')}
          className={`tab-button ${activeTab === 'camera' ? 'active' : ''}`}
        >
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
          <Video style={{ width: 14, height: 14, color: '#10b981' }} />
          Live CCTV Demo
        </button>

        <button
          onClick={() => setActiveTab('routing')}
          className={`tab-button ${activeTab === 'routing' ? 'active' : ''}`}
        >
          <Navigation style={{ width: 14, height: 14, color: '#06b6d4' }} />
          Safe Routing
        </button>

        <button
          onClick={() => setActiveTab('critical_paths')}
          className={`tab-button ${activeTab === 'critical_paths' ? 'active' : ''}`}
        >
          <Shield style={{ width: 14, height: 14, color: '#10b981' }} />
          Critical Paths (MongoDB)
        </button>
      </nav>

      {/* Header Quick KPIs */}
      <div className="header-kpis">
        <div className="kpi-chip">
          <CheckCircle2 style={{ width: 14, height: 14, color: '#10b981' }} />
          <span>Clearance:</span>
          <b style={{ color: '#10b981' }}>{stats?.clearance_percentage ?? 78.5}%</b>
        </div>
        <div className="kpi-chip">
          <AlertTriangle style={{ width: 14, height: 14, color: '#ef4444' }} />
          <span>Critical Roads:</span>
          <b style={{ color: '#ef4444' }}>{stats?.critical ?? 6}</b>
        </div>
        <div className="kpi-chip" style={{ border: 'none', background: 'transparent' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
          <span style={{ color: '#94a3b8' }}>FastAPI Live</span>
        </div>
      </div>
    </header>
  );
}
