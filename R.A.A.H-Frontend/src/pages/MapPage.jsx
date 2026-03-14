import React, { useState, useEffect, useRef } from 'react';
import { Map as MapIcon, Navigation, ShieldAlert, X, MapPin, Globe } from 'lucide-react';
import api from '../services/api';

// Leaflet CSS is loaded via CDN in index.html (added by vite config)
let L;

export default function MapPage() {
  const [activeTab, setActiveTab] = useState('highways');
  const [selectedPothole, setSelectedPothole] = useState(null);
  const [potholes, setPotholes] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanMessage, setScanMessage] = useState('');
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null);

  // Initialize map once
  useEffect(() => {
    // Small delay to ensure container size is stabilized
    const timer = setTimeout(() => {
      initMap();
    }, 100);

    // Refresh data every 10s
    const interval = setInterval(() => {
      fetchAndRender();
    }, 10000);

    return () => {
      clearTimeout(timer);
      clearInterval(interval);
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  function initMap() {
    if (mapInstanceRef.current || !mapRef.current || !window.L) return;
    try {
      const map = window.L.map(mapRef.current, {
        center: [20.5937, 78.9629], // India center
        zoom: 5,
        zoomControl: true,
      });
      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(map);
      mapInstanceRef.current = map;
      markersLayerRef.current = window.L.layerGroup().addTo(map);
      fetchAndRender();
    } catch (err) {
      console.error('Leaflet initialization failed:', err);
    }
  }

  async function fetchAndRender() {
    try {
      const [ptRes, predRes] = await Promise.all([
        api.get('/map/list'),
        api.get('/map/predictions'),
      ]);
      setPotholes(ptRes.data.potholes || []);
      setPredictions(predRes.data.features || []);
      renderMarkers(ptRes.data.potholes || [], predRes.data.features || [], activeTab);
    } catch (err) {
      console.error('Map data fetch error:', err);
    }
  }

  async function handleMapillaryScan() {
    if (!mapInstanceRef.current) return;
    setIsScanning(true);
    setScanMessage('Initiating scan...');
    try {
      const center = mapInstanceRef.current.getCenter();
      // Call the new backend endpoint
      const res = await api.post('/citizen/mapillary/scan', null, {
        params: {
          latitude: center.lat,
          longitude: center.lng,
          radius: 50 // Default scan radius
        }
      });
      setScanMessage(res.data.message || 'Scan queued successfully!');
      setTimeout(() => setScanMessage(''), 4000);
    } catch (err) {
      console.error('Mapillary scan error:', err);
      setScanMessage('Failed to queue scan. Ensure your token is set.');
      setTimeout(() => setScanMessage(''), 4000);
    } finally {
      setIsScanning(false);
    }
  }

  function renderMarkers(pts, preds, tab) {
    if (!markersLayerRef.current || !window.L || !mapInstanceRef.current) return;
    markersLayerRef.current.clearLayers();
    const bounds = [];

    if (tab === 'risk_zones') {
      preds.forEach(f => {
        const { lat, lng } = { lat: f.geometry.coordinates[1], lng: f.geometry.coordinates[0] };
        bounds.push([lat, lng]);
        const p = f.properties;
        const color = p.color || '#22c55e';
        window.L.circleMarker([lat, lng], {
          radius: 20,
          fillColor: color,
          color: color,
          weight: 2,
          opacity: 0.8,
          fillOpacity: 0.4,
        }).bindTooltip(
          `PVI: ${p.pvi_score} — ${p.risk_level.toUpperCase()}`,
          { permanent: false }
        ).addTo(markersLayerRef.current);
      });
    } else {
      pts.forEach(p => {
        const { latitude, longitude } = p;

        // Filter: Ignore null/undefined or (0,0) coordinates
        if (latitude == null || longitude == null || (latitude === 0 && longitude === 0)) return;
        if (p.status === 'test' || p.is_test === true) return;

        bounds.push([latitude, longitude]);
        const color = p.severity === 'high' ? '#ef4444' : p.severity === 'medium' ? '#f97316' : '#3b82f6';

        const marker = window.L.circleMarker([latitude, longitude], {
          radius: 12,
          fillColor: color,
          color: '#fff',
          weight: 3,
          opacity: 1,
          fillOpacity: 0.9,
        });

        const date = p.timestamp ? new Date(p.timestamp).toLocaleDateString('en-GB', {
          day: '2-digit',
          month: 'short',
          year: 'numeric'
        }) : 'Unknown';

        const popupContent = `
          <div style="font-family: 'Inter', sans-serif; padding: 12px; min-width: 180px; background: rgba(0,0,0,0.9); color: white; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 20px rgba(0,0,0,0.5);">
            <div style="font-weight: 900; color: #22d3ee; margin-bottom: 10px; font-size: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 6px;">${p.id}</div>
            <div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;">Severity: <span style="color: ${color === '#ef4444' ? '#f87171' : color === '#f97316' ? '#fb923c' : '#60a5fa'}">${(p.severity || 'medium').toUpperCase()}</span></div>
            <div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;">Confidence: <span style="color: #4ade80;">${Math.round((p.confidence || 0) * 100)}%</span></div>
            <div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;">Status: <span style="color: #22d3ee;">${(p.status || 'Reported').replace(/_/g, ' ').toUpperCase()}</span></div>
            <div style="font-size: 11px; color: #9ca3af; margin-top: 12px;">Detected: ${date}</div>
          </div>
        `;

        marker.bindPopup(popupContent, {
          closeButton: false,
          className: 'hackathon-leaflet-popup'
        });

        marker.on('click', () => setSelectedPothole(p));
        marker.addTo(markersLayerRef.current);
      });
    }

    // Goal 2: Automatically zoom to detected potholes (3-tier logic)
    if (bounds.length > 1) {
      // Multiple potholes: Fit to bounds
      mapInstanceRef.current.fitBounds(bounds, { padding: [50, 50] });
    } else if (bounds.length === 1) {
      // Single pothole: Detailed view (Zoom 14)
      mapInstanceRef.current.setView(bounds[0], 14);
    } else {
      // No potholes: Default view (India, Zoom 5)
      mapInstanceRef.current.setView([20.5937, 78.9629], 5);
    }
  }

  // Re-render markers when tab OR data changes
  useEffect(() => {
    renderMarkers(potholes, predictions, activeTab);
  }, [activeTab, potholes, predictions]);

  return (
    <div className="flex flex-col h-full space-y-8 relative z-10 w-full text-gray-100">
      {/* Top Header & Internal Tabs */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 border-b border-white/10 pb-6 mb-2">
        <h2 className="text-3xl font-black text-white tracking-[0.1em] uppercase drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">Geospatial Map</h2>
        <div className="flex gap-4">
          <button onClick={() => setActiveTab('highways')}
            className={`flex items-center gap-3 px-6 py-3 rounded-lg transition-all font-bold uppercase tracking-widest text-xs shadow-sm ${activeTab === 'highways' ? 'bg-blue-900/40 text-cyan-400 border border-cyan-400/50 shadow-[0_0_15px_rgba(34,211,238,0.3)]' : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-transparent'}`}>
            <Navigation size={18} className={activeTab === 'highways' ? 'drop-shadow-[0_0_8px_rgba(34,211,238,0.8)]' : ''} /> Highways
          </button>
          <button onClick={() => setActiveTab('cities')}
            className={`flex items-center gap-3 px-6 py-3 rounded-lg transition-all font-bold uppercase tracking-widest text-xs shadow-sm ${activeTab === 'cities' ? 'bg-blue-900/40 text-cyan-400 border border-cyan-400/50 shadow-[0_0_15px_rgba(34,211,238,0.3)]' : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-transparent'}`}>
            <MapIcon size={18} /> Cities
          </button>
          <button onClick={() => setActiveTab('risk_zones')}
            className={`flex items-center gap-3 px-6 py-3 rounded-lg transition-all font-bold uppercase tracking-widest text-xs shadow-sm ${activeTab === 'risk_zones' ? 'bg-red-900/40 text-red-400 border border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.3)]' : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-transparent'}`}>
            <ShieldAlert size={18} className={activeTab === 'risk_zones' ? 'drop-shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse' : ''} /> Risk Zones
          </button>
        </div>
      </div>

      {/* Map Content Section */}
      <div className="flex-1 relative flex flex-col md:flex-row gap-6">
        {/* Map Container */}
        <div className="flex-1 hackathon-glass rounded-3xl border border-white/5 overflow-hidden relative shadow-[inset_0_0_30px_rgba(0,0,0,0.5)]">
          {/* Leaflet Map */}
          <div ref={mapRef} className="absolute inset-0 w-full h-full z-0" style={{ minHeight: '400px' }} />

          {/* Mapillary Scan Button */}
          <div className="absolute top-6 left-16 z-10 hidden md:block">
            <button 
              onClick={handleMapillaryScan}
              disabled={isScanning}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl backdrop-blur-md border shadow-lg transition-all font-black tracking-widest text-xs uppercase ${
                isScanning 
                  ? 'bg-emerald-900/60 text-emerald-400 border-emerald-500/50 cursor-not-allowed' 
                  : 'bg-black/80 text-white border-white/20 hover:bg-white/10 hover:border-cyan-400 hover:text-cyan-400 hover:shadow-[0_0_20px_rgba(34,211,238,0.3)]'
              }`}
            >
              {isScanning ? (
                <>
                  <div className="w-3 h-3 rounded-full border-2 border-t-emerald-400 border-r-transparent border-b-emerald-400 border-l-transparent animate-spin" />
                  Scanning...
                </>
              ) : (
                <>
                  <Globe size={16} /> Scan Area (Mapillary)
                </>
              )}
            </button>
            
            {/* Scan Toast */}
            {scanMessage && (
              <div className="mt-3 px-4 py-2 bg-black/90 text-[10px] font-black tracking-widest uppercase text-cyan-400 border border-cyan-500/30 rounded-lg shadow-xl animate-in fade-in slide-in-from-top-4">
                {scanMessage}
              </div>
            )}
          </div>

        {/* Layer Legend Overlay */}
        <div className="absolute top-6 right-6 bg-black/60 backdrop-blur-xl p-5 rounded-2xl border border-white/10 shadow-[0_10px_30px_rgba(0,0,0,0.5)] space-y-4 z-10 w-56">
          <div className="text-[10px] font-black text-gray-400 mb-2 uppercase tracking-[0.2em] border-b border-white/10 pb-2">Layer Legend</div>
          <div className="flex items-center gap-4 text-xs tracking-wider text-white font-bold"><div className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.8)] animate-pulse"></div> High Severity</div>
          <div className="flex items-center gap-4 text-xs tracking-wider text-white font-bold"><div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_15px_rgba(249,115,22,0.8)]"></div> Medium Severity</div>
          <div className="flex items-center gap-4 text-xs tracking-wider text-white font-bold"><div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.8)]"></div> Low / Resolved</div>
          <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mt-2 pt-2 border-t border-white/10">
            {potholes.length} potholes loaded
          </div>
        </div>

        {/* Popup Detail Card */}
        {selectedPothole && (
          <div className="absolute bottom-8 left-8 w-80 bg-black/80 backdrop-blur-3xl border border-white/10 rounded-3xl p-6 shadow-[0_20px_50px_rgba(0,0,0,0.8)] z-20">
            <div className="flex justify-between items-start mb-5 border-b border-white/10 pb-4">
              <span className="text-[10px] uppercase font-mono tracking-widest font-black text-cyan-400 bg-cyan-900/40 border border-cyan-500/50 px-3 py-1.5 rounded-lg">{selectedPothole.id}</span>
              <button onClick={() => setSelectedPothole(null)} className="text-gray-400 hover:text-white transition-colors hover:bg-white/10 p-1.5 rounded-full"><X size={18} /></button>
            </div>
            <h3 className="font-bold text-lg text-white mb-6 leading-snug flex items-start gap-3">
              <MapPin size={24} className="text-cyan-400 shrink-0 mt-0.5" />
              {selectedPothole.road_name}
            </h3>
            <div className="grid grid-cols-2 gap-4 my-6">
              <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Severity</p>
                <p className={`font-black uppercase tracking-wider text-sm ${selectedPothole.severity === 'high' ? 'text-red-400' : selectedPothole.severity === 'medium' ? 'text-orange-400' : 'text-cyan-400'}`}>{selectedPothole.severity}</p>
              </div>
              <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Confidence</p>
                <p className="font-black text-white text-lg">{Math.round((selectedPothole.confidence || 0) * 100)}<span className="text-sm font-bold text-gray-500">%</span></p>
              </div>
            </div>
            <div className="mb-6 bg-white/5 p-4 rounded-xl border border-white/5">
              <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Status</p>
              <p className="text-sm font-bold text-white uppercase tracking-wider">{selectedPothole.status?.replace(/_/g, ' ')}</p>
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}