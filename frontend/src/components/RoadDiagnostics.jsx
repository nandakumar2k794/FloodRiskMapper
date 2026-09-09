import React from 'react';
import { Info, TrendingUp, Compass, Waves, CheckCircle2, XCircle, Car, Bus, Truck, Bike } from 'lucide-react';

const VEHICLE_THRESHOLDS = [
  { type: 'Standard Car', limit: 25, icon: Car },
  { type: 'SUV / 4x4', limit: 40, icon: Truck },
  { type: 'Public Bus', limit: 50, icon: Bus },
  { type: 'Two-Wheeler', limit: 15, icon: Bike },
];

export default function RoadDiagnostics({ road, stats, onSelectSample }) {
  if (!road) {
    return (
      <div className="sidebar-content">
        <div className="card-box">
          <div className="card-title">
            <Info style={{ width: 14, height: 14, color: '#38bdf8' }} />
            Network Flood Risk Diagnostics
          </div>
          <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
            Click any road segment on the interactive map to inspect real-time ML flood predictions, terrain slope, elevation, and drainage condition.
          </p>
        </div>

        <div>
          <div className="card-title">Current Network State</div>
          <div className="grid-2">
            <div className="metric-cell">
              <span className="label">Total Segments</span>
              <span className="value">{stats?.total_segments || 42}</span>
              <span className="sub" style={{ color: '#34d399' }}>
                {stats?.clearance_percentage || 78.5}% Accessible
              </span>
            </div>

            <div className="metric-cell">
              <span className="label">Critical Roads</span>
              <span className="value" style={{ color: '#f87171' }}>{stats?.critical || 6}</span>
              <span className="sub" style={{ color: '#f87171' }}>Immediate Hazard</span>
            </div>

            <div className="metric-cell">
              <span className="label">High Risk</span>
              <span className="value" style={{ color: '#fb923c' }}>{stats?.high || 8}</span>
              <span className="sub" style={{ color: '#fb923c' }}>Inundation Likely</span>
            </div>

            <div className="metric-cell">
              <span className="label">Mean Risk Score</span>
              <span className="value" style={{ color: '#38bdf8' }}>{stats?.avg_risk || 0.42}</span>
              <span className="sub">Composite 0.0 - 1.0</span>
            </div>
          </div>
        </div>

        <div>
          <div className="card-title">Sample Key Corridors</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { id: 'seg_anna_salai_n', name: 'EVR Periyar Salai (Central-Kilpauk)', risk: 'CRITICAL', depth: 42 },
              { id: 'seg_tnagar_velachery', name: 'Velachery Main Road (T.Nagar-Velachery)', risk: 'CRITICAL', depth: 55 },
              { id: 'seg_gst_airport', name: 'GST Road (Guindy-Airport)', risk: 'LOW', depth: 4 },
              { id: 'seg_sardar_patel', name: 'Sardar Patel Road (Guindy-Adyar)', risk: 'HIGH', depth: 32 },
            ].map((s) => (
              <button
                key={s.id}
                onClick={() => onSelectSample && onSelectSample(s.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  cursor: 'pointer',
                  color: 'inherit',
                  textAlign: 'left',
                }}
              >
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#f1f5f9' }}>{s.name}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>Est. Depth: {s.depth} cm</div>
                </div>
                <span className={`badge badge-${s.risk.toLowerCase()}`}>
                  {s.risk}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const depth = road.predicted_depth_cm || 0;
  const riskClass = road.risk_class || 'LOW';

  return (
    <div className="sidebar-content">
      {/* Title & Badge */}
      <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f8fafc', lineHeight: 1.3 }}>
            {road.name || 'Chennai Corridor'}
          </h2>
          <span className={`badge badge-${riskClass.toLowerCase()}`}>
            {riskClass}
          </span>
        </div>
        <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace', marginTop: 4 }}>
          ID: {road.segment_id}
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="grid-2">
        <div className="metric-cell">
          <span className="label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <TrendingUp style={{ width: 14, height: 14, color: '#38bdf8' }} /> Risk Score
          </span>
          <span className="value">{road.risk_score}</span>
          <span className="sub">
            XGBoost: {((road.flood_probability || road.risk_score) * 100).toFixed(1)}% prob
          </span>
        </div>

        <div className="metric-cell">
          <span className="label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Waves style={{ width: 14, height: 14, color: '#06b6d4' }} /> Water Depth
          </span>
          <span className="value" style={{ color: '#38bdf8' }}>
            {depth} <span style={{ fontSize: 13, fontWeight: 400, color: '#94a3b8' }}>cm</span>
          </span>
          <span className="sub" style={{ color: road.accessible ? '#34d399' : '#f87171', fontWeight: 600 }}>
            {road.accessible ? '✓ Passable' : '✗ Inundated'}
          </span>
        </div>
      </div>

      {/* Physical Terrain Metrics */}
      <div className="card-box">
        <div className="card-title">
          <Compass style={{ width: 14, height: 14, color: '#38bdf8' }} />
          Physical Terrain & Hydrology
        </div>
        <div className="grid-2" style={{ marginTop: 8 }}>
          <div style={{ padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 6 }}>
            <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Elevation</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
              {road.elevation_m !== undefined ? `${road.elevation_m} m` : '6.5 m'}
            </span>
          </div>
          <div style={{ padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 6 }}>
            <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Slope</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
              {road.slope_deg !== undefined ? `${road.slope_deg}°` : '1.2°'}
            </span>
          </div>
          <div style={{ padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 6 }}>
            <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Distance to River</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
              {road.dist_to_water_m !== undefined ? `${road.dist_to_water_m} m` : '200 m'}
            </span>
          </div>
          <div style={{ padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 6 }}>
            <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Distance to Drain</span>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
              {road.dist_to_drain_m !== undefined ? `${road.dist_to_drain_m} m` : '20 m'}
            </span>
          </div>
        </div>
      </div>

      {/* AI Explanation */}
      <div className="card-box">
        <div className="card-title">Why is this road at risk?</div>
        <p style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>
          {road.explanation ||
            `Road is situated at ${road.elevation_m || 6.2}m elevation. High cumulative rainfall combined with restricted hydraulic drain capacity increases pooling risk.`}
        </p>
        {road.cctv_monitored && (
          <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 11, color: '#fbbf24' }}>
            ★ Monitored by DrainWatch CCTV ({road.cctv_camera_id}). Visual inlet blockage factor applied.
          </div>
        )}
      </div>

      {/* Vehicle Accessibility Matrix */}
      <div>
        <div className="card-title">Vehicle Clearance Check</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {VEHICLE_THRESHOLDS.map((v) => {
            const Icon = v.icon;
            const isPassable = depth < v.limit;
            return (
              <div
                key={v.type}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  background: 'var(--bg-elevated)',
                  borderRadius: 6,
                  fontSize: 12,
                  border: '1px solid var(--border)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Icon style={{ width: 14, height: 14, color: '#94a3b8' }} />
                  <span>{v.type}</span>
                  <span style={{ fontSize: 10, color: '#64748b' }}>(Limit {v.limit}cm)</span>
                </div>
                {isPassable ? (
                  <span style={{ color: '#34d399', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <CheckCircle2 style={{ width: 14, height: 14 }} /> Passable
                  </span>
                ) : (
                  <span style={{ color: '#f87171', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <XCircle style={{ width: 14, height: 14 }} /> Blocked
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
