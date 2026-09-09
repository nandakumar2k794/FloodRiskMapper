import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MapComponent from './components/MapComponent';
import RoadDiagnostics from './components/RoadDiagnostics';
import RainfallSimulator from './components/RainfallSimulator';
import DrainWatchCamera from './components/DrainWatchCamera';
import RoutePlanner from './components/RoutePlanner';
import CriticalPathsMongo from './components/CriticalPathsMongo';
import { fetchRiskMap, fetchLayers, fetchRiskGrid } from './api/client';

export default function App() {
  const [rainfall, setRainfall] = useState(120);
  const [roadFeatures, setRoadFeatures] = useState([]);
  const [layersData, setLayersData] = useState(null);
  const [gridFeatures, setGridFeatures] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedRoad, setSelectedRoad] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [activeTab, setActiveTab] = useState('camera'); // default to camera to showcase the camera feature!
  const [activeRoute, setActiveRoute] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load Environmental Layers & Grid
  useEffect(() => {
    fetchLayers()
      .then((data) => {
        setLayersData(data);
        if (data.cctv_stations?.length > 0) {
          setSelectedCamera(data.cctv_stations[0]);
        }
      })
      .catch((err) => console.error('Error fetching layers:', err));

    fetchRiskGrid(250)
      .then((data) => {
        if (data.features) setGridFeatures(data.features);
      })
      .catch((err) => console.error('Error fetching grid:', err));
  }, []);

  // Fetch / Re-score Road Network when Rainfall changes
  useEffect(() => {
    setLoading(true);
    fetchRiskMap({ rainfall })
      .then((data) => {
        const features = data.features || [];
        setRoadFeatures(features);
        setStats(data.meta || null);

        if (selectedRoad) {
          const updated = features.find((f) => f.properties?.segment_id === selectedRoad.segment_id);
          if (updated) setSelectedRoad(updated.properties);
        }
      })
      .catch((err) => console.error('Error fetching risk map:', err))
      .finally(() => setLoading(false));
  }, [rainfall]);

  const handleSelectRoad = (roadProperties) => {
    setSelectedRoad(roadProperties);
    setActiveTab('diagnostics');
  };

  const handleSelectCamera = (station) => {
    setSelectedCamera(station);
    setActiveTab('camera');
  };

  const handleSelectSample = (segmentId) => {
    const found = roadFeatures.find((f) => f.properties?.segment_id === segmentId);
    if (found) {
      setSelectedRoad(found.properties);
    }
  };

  const handleTelemetryUpdate = (telemetry) => {
    if (!telemetry) return;
    if (telemetry.blockage_percentage >= 50) {
      setRoadFeatures((prev) =>
        prev.map((f) => {
          if (f.properties?.segment_id === 'seg_tnagar_velachery') {
            return {
              ...f,
              properties: {
                ...f.properties,
                risk_score: Math.min(1.0, f.properties.risk_score + 0.15),
                risk_class: 'CRITICAL',
                predicted_depth_cm: Math.max(f.properties.predicted_depth_cm || 0, 52),
                accessible: false,
                explanation: `DrainWatch Alert: Debris inlet blockage (${telemetry.blockage_percentage}%). Severe overflow surge penalty applied.`,
              },
            };
          }
          return f;
        })
      );
    }
  };

  return (
    <div className="app-shell">
      {/* Top Navigation & Status Bar */}
      <Header
        rainfall={rainfall}
        stats={stats}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Main Workspace (Map + Inspector Panel) */}
      <div className="main-workspace">
        {/* Full-bleed Leaflet Map */}
        <MapComponent
          roadFeatures={roadFeatures}
          layersData={layersData}
          selectedRoad={selectedRoad}
          onSelectRoad={handleSelectRoad}
          onSelectCamera={handleSelectCamera}
          activeRoute={activeRoute}
          gridFeatures={gridFeatures}
        />

        {/* Focused Right Sidebar */}
        <aside className="sidebar-panel">
          {activeTab === 'diagnostics' && (
            <RoadDiagnostics
              road={selectedRoad}
              stats={stats}
              onSelectSample={handleSelectSample}
            />
          )}

          {activeTab === 'rainfall' && (
            <RainfallSimulator
              rainfall={rainfall}
              onRainfallChange={setRainfall}
              stats={stats}
              loading={loading}
            />
          )}

          {activeTab === 'camera' && (
            <DrainWatchCamera
              station={selectedCamera}
              onTelemetryUpdate={handleTelemetryUpdate}
            />
          )}

          {activeTab === 'routing' && (
            <RoutePlanner onRoutesComputed={setActiveRoute} />
          )}

          {activeTab === 'critical_paths' && (
            <CriticalPathsMongo
              rainfall={rainfall}
              onSelectRoad={handleSelectRoad}
            />
          )}

          <div className="sidebar-footer">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#38bdf8', display: 'inline-block' }} />
              XGBoost Model Active
            </span>
            <span>Chennai EPSG:4326</span>
          </div>
        </aside>
      </div>
    </div>
  );
}
