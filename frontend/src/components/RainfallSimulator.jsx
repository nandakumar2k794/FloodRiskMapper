import React from 'react';
import { CloudRain, AlertCircle } from 'lucide-react';

const SCENARIOS = [
  { id: 'dry', label: '☀️ Dry Season', rainfall: 0, desc: 'Normal conditions, all roads completely passable' },
  { id: 'light', label: '🌦️ Light Monsoon', rainfall: 35, desc: 'Standard showers; localized puddling on low roads' },
  { id: 'heavy', label: '🌧️ Heavy Downpour', rainfall: 85, desc: 'Monsoonal surge; storm drains run at 80% capacity' },
  { id: 'cyclone', label: '🌀 Cyclone Michaung (2023)', rainfall: 185, desc: 'Severe inundation; Velachery & Adyar corridors submerge' },
  { id: 'catastrophic', label: '🌊 December 2015 Floods', rainfall: 300, desc: 'Historic disaster; major subways and trunk routes impassable' },
];

export default function RainfallSimulator({ rainfall, onRainfallChange, stats }) {
  return (
    <div className="sidebar-content">
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8, color: '#f8fafc' }}>
          <CloudRain style={{ width: 18, height: 18, color: '#38bdf8' }} />
          Rainfall Scenario Simulator
        </h2>
        <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 4, lineHeight: 1.4 }}>
          Adjust Chennai 24-hour rainfall accumulation. Re-evaluates all 42 road segments dynamically via the trained XGBoost model.
        </p>
      </div>

      {/* Interactive Slider Card */}
      <div className="card-box" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#e2e8f0' }}>24-Hour Rainfall</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>
            {rainfall} <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 400 }}>mm</span>
          </span>
        </div>

        <input
          type="range"
          min="0"
          max="300"
          step="5"
          value={rainfall}
          onChange={(e) => onRainfallChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer', accentColor: '#0284c7' }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
          <span>0mm (Dry)</span>
          <span>85mm (Heavy)</span>
          <span>185mm (Michaung)</span>
          <span>300mm (2015)</span>
        </div>
      </div>

      {/* Scenario Presets */}
      <div>
        <div className="card-title">Historical Benchmark Presets</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {SCENARIOS.map((sc) => {
            const isActive = rainfall === sc.rainfall;
            return (
              <button
                key={sc.id}
                onClick={() => onRainfallChange(sc.rainfall)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 3,
                  padding: '10px 12px',
                  borderRadius: 8,
                  border: isActive ? '1px solid #38bdf8' : '1px solid var(--border)',
                  background: isActive ? 'rgba(56, 189, 248, 0.12)' : 'var(--bg-elevated)',
                  cursor: 'pointer',
                  color: 'inherit',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: isActive ? '#38bdf8' : '#f1f5f9' }}>
                    {sc.label}
                  </span>
                  <span style={{ fontSize: 11, fontFamily: 'monospace', fontWeight: 600, color: '#94a3b8' }}>
                    {sc.rainfall} mm
                  </span>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.3 }}>
                  {sc.desc}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Dynamic Forecast Summary */}
      <div className="card-box">
        <div className="card-title" style={{ color: '#fbbf24' }}>
          <AlertCircle style={{ width: 14, height: 14, color: '#fbbf24' }} />
          Forecast Impact Summary
        </div>
        <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>
          <div>• <b>Network Accessibility:</b> {stats?.clearance_percentage ?? 78.5}% of arterial network passable.</div>
          <div>• <b>Impassable Segments:</b> {stats?.critical ?? 6} corridors exceed vehicle clearance thresholds.</div>
          <div>• <b>Drain Behavior:</b> {rainfall >= 180 ? 'Severe drain surcharging. Backflow into low-elevation carriageways.' : rainfall >= 70 ? 'Arterial drains operating at 80% capacity.' : 'Normal gravity discharge into Buckingham Canal.'}</div>
        </div>
      </div>
    </div>
  );
}
