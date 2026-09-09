import React, { useState, useEffect } from 'react';
import { Database, AlertTriangle, RefreshCw, CheckCircle2, ShieldAlert, Waves, MapPin, ArrowRight } from 'lucide-react';
import { fetchCriticalPaths, fetchCriticalStats, syncCriticalPaths } from '../api/client';

export default function CriticalPathsMongo({ rainfall, onSelectRoad }) {
  const [criticalPaths, setCriticalPaths] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  const loadMongoData = async () => {
    try {
      setLoading(true);
      const res = await fetchCriticalPaths({ min_risk: 0.5, limit: 50 });
      setCriticalPaths(res.critical_paths || []);
      setStats(res.stats || null);
    } catch (err) {
      console.error('Error loading critical paths from MongoDB:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMongoData();
  }, [rainfall]);

  const handleSync = async () => {
    try {
      setSyncing(true);
      setStatusMessage('');
      const res = await syncCriticalPaths(rainfall);
      setStatusMessage(`✅ ${res.message || 'Successfully synced with MongoDB'}`);
      await loadMongoData();
    } catch (err) {
      console.error('Error syncing to MongoDB:', err);
      setStatusMessage('⚠️ Sync failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setSyncing(false);
    }
  };

  return (
    <div className="sidebar-content">
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8, color: '#f8fafc' }}>
            <Database style={{ width: 18, height: 18, color: '#10b981' }} />
            Critical Paths Storage
          </h2>
          <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
            MONGODB ACTIVE
          </span>
        </div>
        <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          Persisted high-risk road corridors, inundated bottlenecks, and emergency detour paths.
        </p>
      </div>

      {/* MongoDB Status Box */}
      <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
            Database: <code style={{ color: '#38bdf8' }}>chennai_flood_risk</code>
          </span>
          <span style={{ fontSize: 11, color: '#94a3b8' }}>Collection: <b>critical_paths</b></span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 4 }}>
          <div style={{ background: 'var(--bg-elevated)', padding: '8px 10px', borderRadius: 6, textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>Critical Roads</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#ef4444', marginTop: 2 }}>
              {stats?.total_critical_paths ?? criticalPaths.length}
            </div>
          </div>

          <div style={{ background: 'var(--bg-elevated)', padding: '8px 10px', borderRadius: 6, textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>Max Depth</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#06b6d4', marginTop: 2 }}>
              {stats?.max_inundation_depth_cm ?? 58} cm
            </div>
          </div>

          <div style={{ background: 'var(--bg-elevated)', padding: '8px 10px', borderRadius: 6, textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>Impassable</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#f59e0b', marginTop: 2 }}>
              {stats?.impassable_count ?? criticalPaths.filter(p => !p.accessible).length}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <button
            onClick={handleSync}
            disabled={syncing}
            className="btn-action btn-blue"
            style={{ flex: 1, background: '#10b981', borderColor: '#10b981' }}
          >
            <RefreshCw style={{ width: 14, height: 14 }} className={syncing ? 'spin' : ''} />
            {syncing ? 'Syncing to MongoDB...' : 'Sync Critical Paths to MongoDB'}
          </button>
          <button
            onClick={loadMongoData}
            disabled={loading}
            className="btn-action btn-secondary"
            style={{ width: 'auto' }}
            title="Refresh list"
          >
            <RefreshCw style={{ width: 14, height: 14 }} className={loading ? 'spin' : ''} />
          </button>
        </div>

        {statusMessage && (
          <div style={{ fontSize: 11, color: '#38bdf8', padding: '6px 8px', background: 'rgba(56, 189, 248, 0.1)', borderRadius: 4 }}>
            {statusMessage}
          </div>
        )}
      </div>

      {/* Stored Critical Paths List */}
      <div>
        <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span>Stored Critical Corridors ({criticalPaths.length})</span>
          <span style={{ fontSize: 10, color: '#94a3b8' }}>Rainfall: {rainfall} mm</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 380, overflowY: 'auto' }}>
          {criticalPaths.length > 0 ? (
            criticalPaths.map((path) => (
              <div
                key={path.segment_id}
                onClick={() => onSelectRoad && onSelectRoad(path)}
                style={{
                  padding: '10px 12px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderLeft: `4px solid ${path.risk_score >= 0.75 ? '#ef4444' : '#f97316'}`,
                  borderRadius: 6,
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#f1f5f9' }}>{path.name}</div>
                    <div style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace', marginTop: 1 }}>
                      ID: {path.segment_id}
                    </div>
                  </div>
                  <span className={`badge ${path.risk_score >= 0.75 ? 'badge-critical' : 'badge-high'}`}>
                    {(path.risk_score * 100).toFixed(0)}% RISK
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8, fontSize: 11 }}>
                  <span style={{ color: '#06b6d4', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Waves style={{ width: 12, height: 12 }} />
                    Depth: <b>{path.predicted_depth_cm} cm</b>
                  </span>
                  <span style={{ color: path.accessible ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                    {path.accessible ? '✓ Passable' : '✗ Impassable'}
                  </span>
                  {path.cctv_monitored && (
                    <span style={{ color: '#facc15', fontSize: 10, fontWeight: 600 }}>
                      ● CCTV Monitored
                    </span>
                  )}
                </div>

                {path.explanation && (
                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6, lineHeight: 1.3 }}>
                    {path.explanation}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div style={{ textAlign: 'center', padding: '24px 12px', color: '#94a3b8', fontSize: 12 }}>
              No critical paths found for current scenario. Click <b>Sync Critical Paths to MongoDB</b> to persist current roads.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
