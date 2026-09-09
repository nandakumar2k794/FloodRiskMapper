import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Layers, Map as MapIcon } from 'lucide-react';

const RISK_COLORS = {
  LOW: '#10b981',
  MEDIUM: '#f59e0b',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
};

const BASEMAP_TILES = {
  osm: {
    name: '🗺️ Standard OpenStreetMap (No Filter)',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  },
  satellite: {
    name: '🛰️ Satellite Imagery (Esri World)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, Maxar, Earthstar Geographics',
    maxZoom: 19,
  },
  carto_light: {
    name: '🏙️ Clean Light (Carto Positron)',
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19,
  },
};

export default function MapComponent({
  roadFeatures = [],
  layersData = null,
  selectedRoad = null,
  onSelectRoad,
  onSelectCamera,
  activeRoute = null,
  gridFeatures = [],
}) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const tileLayerRef = useRef(null);
  const layersGroupRef = useRef({
    roads_casing: null,
    roads: null,
    waterways: null,
    waterbodies: null,
    drains: null,
    cctv: null,
    routes: null,
    grid: null,
  });

  const [currentBasemap, setCurrentBasemap] = useState('osm'); // Default to Standard OSM with no filter

  const [visibleLayers, setVisibleLayers] = useState({
    roads: true,
    waterways: true,
    waterbodies: true,
    drains: true,
    cctv: true,
    grid: false,
  });

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [13.045, 80.23],
      zoom: 12,
      minZoom: 10,
      maxZoom: 18,
      zoomControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Standard OpenStreetMap base layer (no dark filters)
    const base = BASEMAP_TILES[currentBasemap] || BASEMAP_TILES.osm;
    tileLayerRef.current = L.tileLayer(base.url, {
      attribution: base.attribution,
      maxZoom: base.maxZoom,
    }).addTo(map);

    // Create Layer Groups
    layersGroupRef.current.waterbodies = L.layerGroup().addTo(map);
    layersGroupRef.current.waterways = L.layerGroup().addTo(map);
    layersGroupRef.current.drains = L.layerGroup().addTo(map);
    layersGroupRef.current.grid = L.layerGroup().addTo(map);
    layersGroupRef.current.roads_casing = L.layerGroup().addTo(map);
    layersGroupRef.current.roads = L.layerGroup().addTo(map);
    layersGroupRef.current.cctv = L.layerGroup().addTo(map);
    layersGroupRef.current.routes = L.layerGroup().addTo(map);

    mapRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 100);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update Basemap Tiles when selected
  useEffect(() => {
    if (!mapRef.current) return;
    const base = BASEMAP_TILES[currentBasemap] || BASEMAP_TILES.osm;
    if (tileLayerRef.current) {
      mapRef.current.removeLayer(tileLayerRef.current);
    }
    tileLayerRef.current = L.tileLayer(base.url, {
      attribution: base.attribution,
      maxZoom: base.maxZoom,
    }).addTo(mapRef.current);
    tileLayerRef.current.bringToBack();
  }, [currentBasemap]);

  // Update Roads Layer with crisp dual-stroke outlines
  useEffect(() => {
    const casingGroup = layersGroupRef.current.roads_casing;
    const group = layersGroupRef.current.roads;
    if (!group || !casingGroup) return;

    group.clearLayers();
    casingGroup.clearLayers();

    if (!visibleLayers.roads || !roadFeatures.length) return;

    roadFeatures.forEach((feature) => {
      const p = feature.properties || {};
      const coords = feature.geometry?.coordinates;
      if (!coords || coords.length < 2) return;

      const latlngs = coords.map((c) => [c[1], c[0]]);
      const isSelected = selectedRoad && selectedRoad.segment_id === p.segment_id;
      const color = RISK_COLORS[p.risk_class] || '#94a3b8';

      // Outer outline for high visibility against any map
      const casing = L.polyline(latlngs, {
        color: isSelected ? '#0284c7' : '#0f172a',
        weight: isSelected ? 10 : 7,
        opacity: 0.7,
      });
      casingGroup.addLayer(casing);

      // Inner colored line
      const polyline = L.polyline(latlngs, {
        color: isSelected ? '#38bdf8' : color,
        weight: isSelected ? 7 : 5,
        opacity: 1.0,
      });

      polyline.bindTooltip(
        `<div style="font-family: sans-serif; font-size: 11px; padding: 4px; background: #0f172a; color: #fff; border-radius: 4px;">
          <div style="font-weight: 700; color: #f8fafc;">${p.name || 'Chennai Corridor'}</div>
          <div style="color: ${color}; font-weight: 600; margin-top: 2px;">ML Risk: ${p.risk_class} (${p.risk_score})</div>
          <div style="color: #94a3b8;">Est. Inundation: ${p.predicted_depth_cm || 0} cm</div>
          <div style="color: ${p.accessible ? '#4ade80' : '#f87171'}; font-size: 10px; font-weight: bold;">
            ${p.accessible ? '✓ Passable' : '✗ Impassable'}
          </div>
        </div>`,
        { sticky: true }
      );

      polyline.on('click', () => {
        if (onSelectRoad) onSelectRoad(p);
      });

      group.addLayer(polyline);
    });
  }, [roadFeatures, selectedRoad, visibleLayers.roads, onSelectRoad]);

  // Update Waterways, Waterbodies, and Drains
  useEffect(() => {
    if (!layersData) return;

    // 1. Waterbodies
    const wbGroup = layersGroupRef.current.waterbodies;
    wbGroup.clearLayers();
    if (visibleLayers.waterbodies && layersData.waterbodies?.features) {
      L.geoJSON(layersData.waterbodies, {
        style: {
          color: '#0284c7',
          weight: 2,
          fillColor: '#0ea5e9',
          fillOpacity: 0.45,
        },
        onEachFeature: (feature, layer) => {
          layer.bindTooltip(
            `<span style="font-size: 11px; font-weight: 700; color: #0369a1;">${feature.properties?.name || 'Water Body'}</span>`,
            { sticky: true }
          );
        },
      }).addTo(wbGroup);
    }

    // 2. Waterways (Rivers & Canals)
    const wwGroup = layersGroupRef.current.waterways;
    wwGroup.clearLayers();
    if (visibleLayers.waterways && layersData.waterways?.features) {
      L.geoJSON(layersData.waterways, {
        style: {
          color: '#1d4ed8',
          weight: 5,
          opacity: 0.9,
        },
        onEachFeature: (feature, layer) => {
          layer.bindTooltip(
            `<span style="font-size: 11px; font-weight: 700; color: #1d4ed8;">${feature.properties?.name || 'Waterway'}</span>`,
            { sticky: true }
          );
        },
      }).addTo(wwGroup);
    }

    // 3. Storm Drains
    const drGroup = layersGroupRef.current.drains;
    drGroup.clearLayers();
    if (visibleLayers.drains && layersData.drains?.features) {
      L.geoJSON(layersData.drains, {
        style: {
          color: '#0d9488',
          weight: 3.5,
          dashArray: '6, 6',
          opacity: 0.85,
        },
        onEachFeature: (feature, layer) => {
          layer.bindTooltip(
            `<span style="font-size: 11px; font-weight: 700; color: #0f766e;">Drain: ${feature.properties?.name || 'Drain'}</span>`,
            { sticky: true }
          );
        },
      }).addTo(drGroup);
    }

    // 4. CCTV Stations
    const cctvGroup = layersGroupRef.current.cctv;
    cctvGroup.clearLayers();
    if (visibleLayers.cctv && layersData.cctv_stations) {
      layersData.cctv_stations.forEach((station) => {
        const isDemo = station.camera_id === 'CAM_VELACHERY_01';
        const isOverflow = station.overflow_detected;

        const iconHtml = `
          <div style="position: relative; display: flex; align-items: center; justify-content: center;">
            ${
              isDemo
                ? '<div style="position: absolute; width: 34px; height: 34px; border-radius: 50%; background: rgba(239, 68, 68, 0.4); animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>'
                : ''
            }
            <div style="
              width: ${isDemo ? '28px' : '22px'};
              height: ${isDemo ? '28px' : '22px'};
              border-radius: 50%;
              background: ${isDemo ? '#ef4444' : isOverflow ? '#f97316' : '#0284c7'};
              border: 2px solid #ffffff;
              box-shadow: 0 0 8px rgba(0,0,0,0.6);
              display: flex;
              align-items: center;
              justify-content: center;
              cursor: pointer;
            ">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
                <circle cx="12" cy="13" r="3"/>
              </svg>
            </div>
          </div>
        `;

        const icon = L.divIcon({
          html: iconHtml,
          className: '',
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        });

        const marker = L.marker([station.lat, station.lon], { icon });

        marker.bindTooltip(
          `<div style="font-family: sans-serif; font-size: 11px; padding: 4px; background: #0f172a; color: #fff; border-radius: 4px;">
            <div style="font-weight: 700; color: #38bdf8;">${station.name}</div>
            <div style="color: #cbd5e1; font-size: 10px;">ID: ${station.camera_id}</div>
            <div style="color: ${station.blockage_percentage >= 50 ? '#f87171' : '#4ade80'}; font-weight: 600;">
              Blockage: ${station.blockage_percentage}%
            </div>
            ${isDemo ? '<div style="color: #facc15; font-size: 10px; margin-top:2px;">★ ACTIVE DEMO CAMERA</div>' : ''}
          </div>`,
          { sticky: true }
        );

        marker.on('click', () => {
          if (onSelectCamera) onSelectCamera(station);
        });

        cctvGroup.addLayer(marker);
      });
    }
  }, [layersData, visibleLayers, onSelectCamera]);

  // Update Routes Layer
  useEffect(() => {
    const routeGroup = layersGroupRef.current.routes;
    if (!routeGroup) return;
    routeGroup.clearLayers();

    if (!activeRoute) return;

    if (activeRoute.standard?.geometry?.coordinates) {
      const stdCoords = activeRoute.standard.geometry.coordinates.map((c) => [c[1], c[0]]);
      const stdLine = L.polyline(stdCoords, {
        color: '#f43f5e',
        weight: 5,
        dashArray: '8, 6',
        opacity: 0.9,
      });
      stdLine.bindTooltip(
        `<b style="color:#f43f5e;">Standard Shortest Route</b><br/>Inundated Risk Penalty Applied`,
        { sticky: true }
      );
      routeGroup.addLayer(stdLine);
    }

    if (activeRoute.safe?.geometry?.coordinates) {
      const safeCoords = activeRoute.safe.geometry.coordinates.map((c) => [c[1], c[0]]);
      const safeLine = L.polyline(safeCoords, {
        color: '#166534',
        weight: 6.5,
        opacity: 0.95,
      });
      safeLine.bindTooltip(
        `<b style="color:#166534;">Recommended Flood-Safe Route</b><br/>High-Ground Corridors`,
        { sticky: true }
      );
      routeGroup.addLayer(safeLine);

      if (mapRef.current) {
        mapRef.current.fitBounds(safeLine.getBounds(), { padding: [40, 40] });
      }
    }
  }, [activeRoute]);

  // Update Flood Risk Grid Layer
  useEffect(() => {
    const gridGroup = layersGroupRef.current.grid;
    if (!gridGroup) return;
    gridGroup.clearLayers();

    if (!visibleLayers.grid || !gridFeatures.length) return;

    gridFeatures.forEach((f) => {
      const coords = f.geometry?.coordinates;
      if (!coords) return;
      const prob = f.properties?.flood_probability || 0;
      const color = prob > 0.75 ? '#ef4444' : prob > 0.5 ? '#f97316' : prob > 0.25 ? '#f59e0b' : '#10b981';

      const circle = L.circleMarker([coords[1], coords[0]], {
        radius: 4.5,
        fillColor: color,
        color: '#ffffff',
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0.6,
      });

      circle.bindTooltip(
        `<div><b>XGBoost Model Cell</b><br/>Flood Prob: ${(prob * 100).toFixed(1)}%<br/>Risk: ${f.properties?.risk_class}</div>`,
        { sticky: true }
      );

      gridGroup.addLayer(circle);
    });
  }, [gridFeatures, visibleLayers.grid]);

  return (
    <div className="map-viewport">
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />

      {/* Floating Basemap & Layer Controls */}
      <div className="floating-widget">
        <div className="widget-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <MapIcon style={{ width: 14, height: 14, color: '#38bdf8' }} />
            Map View (No Filter)
          </span>
        </div>

        {/* Basemap Switcher */}
        <div style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
          <select
            value={currentBasemap}
            onChange={(e) => setCurrentBasemap(e.target.value)}
            className="input-field"
            style={{ fontSize: 11, padding: '4px 6px', width: '100%', background: 'var(--bg-darkest)' }}
          >
            <option value="osm">🗺️ OpenStreetMap (Standard)</option>
            <option value="satellite">🛰️ Satellite (Esri Imagery)</option>
            <option value="carto_light">🏙️ Carto Positron (Light)</option>
          </select>
        </div>

        <div className="widget-title" style={{ marginTop: 4, marginBottom: 4 }}>
          <Layers style={{ width: 13, height: 13, color: '#38bdf8' }} />
          <span>GIS Feature Layers</span>
        </div>

        <div>
          <label className="widget-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: '#10b981', display: 'inline-block' }} />
              Road Network (XGBoost)
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.roads}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, roads: e.target.checked })}
            />
          </label>

          <label className="widget-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: '#1d4ed8', display: 'inline-block' }} />
              Rivers & Canals
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.waterways}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, waterways: e.target.checked })}
            />
          </label>

          <label className="widget-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: '#0ea5e9', display: 'inline-block' }} />
              Lakes & Wetlands
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.waterbodies}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, waterbodies: e.target.checked })}
            />
          </label>

          <label className="widget-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: '#0d9488', display: 'inline-block' }} />
              Stormwater Drains
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.drains}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, drains: e.target.checked })}
            />
          </label>

          <label className="widget-row">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', display: 'inline-block' }} />
              CCTV Stations (AI)
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.cctv}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, cctv: e.target.checked })}
            />
          </label>

          <label className="widget-row" style={{ paddingTop: 6, borderTop: '1px solid var(--border)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#94a3b8' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#a855f7', display: 'inline-block' }} />
              XGBoost Risk Grid
            </span>
            <input
              type="checkbox"
              checked={visibleLayers.grid}
              onChange={(e) => setVisibleLayers({ ...visibleLayers, grid: e.target.checked })}
            />
          </label>
        </div>
      </div>

      {/* Floating Legend */}
      <div className="floating-legend">
        <span style={{ fontWeight: 600, color: '#f8fafc', fontSize: 11 }}>ML Risk:</span>
        <span className="legend-pill"><span className="legend-dot" style={{ background: '#10b981' }} /> Low</span>
        <span className="legend-pill"><span className="legend-dot" style={{ background: '#f59e0b' }} /> Med</span>
        <span className="legend-pill"><span className="legend-dot" style={{ background: '#f97316' }} /> High</span>
        <span className="legend-pill"><span className="legend-dot" style={{ background: '#ef4444' }} /> Critical</span>
      </div>
    </div>
  );
}
