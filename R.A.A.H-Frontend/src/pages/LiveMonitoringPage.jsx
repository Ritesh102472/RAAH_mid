import React, { useState, useEffect, useRef } from 'react';
import { Camera, Activity, ScrollText, Crosshair, AlertTriangle, Upload, CheckCircle, LayoutDashboard, X, Globe, MapPin, Hash, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import api, { API_BASE } from '../services/api';
import { useWebSocketContext } from '../context/WebSocketContext';

export default function LiveMonitoringPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('live');
  const [logs, setLogs] = useState([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [selectedPotholeId, setSelectedPotholeId] = useState(null);
  const [potholeDetail, setPotholeDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (activeTab === 'logs') fetchLogs();
  }, [activeTab]);

  const { isLive, subscribe } = useWebSocketContext();

  useEffect(() => {
    const unsubscribe = subscribe((message) => {
      if (message.event === 'new_pothole' || message.event === 'discovery_complete') {
        if (activeTab === 'logs') fetchLogs();
      }
    });
    return unsubscribe;
  }, [subscribe, activeTab]);

  async function fetchLogs() {
    setLogsLoading(true);
    try {
      const res = await api.get('/map/potholes?limit=20');
      setLogs(res.data.features || []);
    } catch (err) {
      console.error('Failed to load logs:', err);
    } finally {
      setLogsLoading(false);
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    setUploadError('');
    try {
      if (!navigator.geolocation) {
        setUploadError('Geolocation is not supported by this browser.');
        setUploading(false);
        return;
      }

      const coords = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve(pos.coords),
          (err) => reject(err),
          { timeout: 10000, enableHighAccuracy: true }
        );
      }).catch(() => {
        alert('Location access is required to report potholes.');
        return null;
      });

      if (!coords) {
        setUploading(false);
        return;
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('latitude', coords.latitude);
      formData.append('longitude', coords.longitude);

      const res = await api.post('/citizen/upload', formData);
      setUploadResult(res.data);
      if (activeTab === 'logs') fetchLogs();
    } catch (err) {
      setUploadError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  }

  async function handleCardClick(potholeId) {
    setSelectedPotholeId(potholeId);
    setDetailLoading(true);
    try {
      const res = await api.get(`/citizen/potholes/${potholeId}`);
      setPotholeDetail(res.data);
    } catch (err) {
      console.error('Failed to fetch pothole details:', err);
    } finally {
      setDetailLoading(false);
    }
  }

  const severityColor = (sev) => {
    if (sev === 'high') return 'text-red-400 border-red-500/50 bg-red-500/10';
    if (sev === 'medium') return 'text-orange-400 border-orange-500/50 bg-orange-500/10';
    return 'text-cyan-400 border-cyan-500/50 bg-cyan-500/10';
  };

  return (
    <div className="flex flex-col h-full space-y-8 relative z-10 w-full text-gray-100">
      {/* Top Header & Internal Tabs */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 border-b border-white/10 pb-6 mb-2">
        <h2 className="text-3xl font-black text-white tracking-[0.1em] uppercase drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">Live Monitoring</h2>
        <div className="flex gap-4">
          {[
            { key: 'live', icon: <Activity size={18} />, label: 'Live Detection' },
            { key: 'cameras', icon: <Camera size={18} />, label: 'Camera Feeds' },
            { key: 'logs', icon: <ScrollText size={18} />, label: 'Detection Logs' },
          ].map(tab => (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-3 px-6 py-3 rounded-lg transition-all font-bold uppercase tracking-widest text-xs shadow-sm ${activeTab === tab.key ? 'bg-blue-900/40 text-cyan-400 border border-cyan-400/50 shadow-[0_0_15px_rgba(34,211,238,0.3)]' : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-transparent'}`}>
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>
        
        {/* Real-time Status */}
        <div className="flex items-center gap-3 px-4 py-2 bg-black/40 rounded-xl border border-white/5 backdrop-blur-md">
          <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 shadow-[0_0_10px_#10b981] animate-pulse' : 'bg-red-500'}`}></div>
          <span className={`text-[10px] font-black uppercase tracking-widest ${isLive ? 'text-emerald-400' : 'text-red-400'}`}>
            {isLive ? 'Live Stream Active' : 'Connecting...'}
          </span>
        </div>
      </div>

      <div className="flex-1 hackathon-glass rounded-3xl p-8 shadow-[inset_0_0_30px_rgba(0,0,0,0.5)] overflow-hidden relative border border-white/5">

        {/* TAB 1: LIVE DETECTION */}
        {activeTab === 'live' && (
          <div className="h-full flex flex-col md:flex-row gap-8">
            <div
              className={`flex-1 bg-black/60 rounded-2xl border border-cyan-500/30 relative overflow-hidden flex flex-col items-center ${uploadResult ? 'justify-start pt-8' : 'justify-center'} group shadow-[inset_0_0_50px_rgba(0,0,0,0.8)] cursor-pointer hover:border-cyan-400/60 transition-all`}
              onClick={() => !uploading && fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" accept="image/*,video/*" className="hidden" onChange={handleFileUpload} />
              {(uploading || (!uploadResult && !uploadError)) && (
                <div className="absolute top-0 left-0 w-full h-1 bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,1)] animate-scan opacity-80 z-50" />
              )}
              {uploading ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-full border-4 border-t-cyan-500 border-r-blue-600 border-b-purple-600 border-l-transparent animate-spin"></div>
                  <p className="text-cyan-400 font-mono text-sm tracking-widest font-bold animate-pulse">PROCESSING...</p>
                </div>
              ) : uploadResult ? (
                <div className="flex flex-col items-center justify-start gap-8 z-10 w-full h-full p-4 pb-20 overflow-y-auto custom-scrollbar">
                    {/* Tight wrapper that matches the physical image area */}
                    <div className="relative inline-block max-w-full flex-shrink-0">
                      {uploadResult.is_video ? (
                        <video
                          src={`${API_BASE}${uploadResult.file_url}`}
                          controls
                          className="max-h-[60vh] w-auto object-contain rounded-xl shadow-2xl border border-white/10"
                          onLoadedMetadata={(e) => {
                            setImageSize({
                              width: e.target.videoWidth,
                              height: e.target.videoHeight
                            });
                          }}
                          onTimeUpdate={(e) => setVideoCurrentTime(e.target.currentTime)}
                        />
                      ) : (
                        <img
                          ref={imageRef}
                          src={`${API_BASE}${uploadResult.file_url}`}
                          alt="Pothole detection"
                          className="max-h-[60vh] w-auto object-contain rounded-xl shadow-2xl border border-white/10"
                          onLoad={(e) => {
                            setImageSize({
                              width: e.target.naturalWidth,
                              height: e.target.naturalHeight
                            });
                          }}
                        />
                      )}

                      {/* SVG Overlay — Now pinned precisely to the image boundaries */}
                      {!uploadResult.is_video && imageSize.width > 0 && (
                        <svg
                          className="absolute top-0 left-0 w-full h-full pointer-events-none"
                          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                          preserveAspectRatio="none"
                        >
                          {uploadResult.potholes?.map((p, i) => {
                            const [x, y, w, h] = p.bbox || [0, 0, 0, 0];
                            const labelText = `Pothole ${p.confidence.toFixed(2)}`;
                            const fontSize = Math.max(10, imageSize.width / 50);
                            const paddingX = 6;
                            const paddingY = 4;
                            const labelWidth = labelText.length * (fontSize * 0.6) + (paddingX * 2);
                            const labelHeight = fontSize + (paddingY * 2);

                            return (
                              <g key={i}>
                                <rect
                                  x={x}
                                  y={y}
                                  width={w}
                                  height={h}
                                  fill="none"
                                  stroke="#3b82f6"
                                  strokeWidth={Math.max(2, imageSize.width / 400)}
                                  className="drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]"
                                />
                                <rect
                                  x={x}
                                  y={y - labelHeight}
                                  width={labelWidth}
                                  height={labelHeight}
                                  fill="#3b82f6"
                                />
                                <text
                                  x={x + paddingX}
                                  y={y - paddingY/2}
                                  fill="white"
                                  fontSize={fontSize}
                                  fontWeight="black"
                                  fontFamily="Inter, sans-serif"
                                  dominantBaseline="text-after-edge"
                                >
                                  {labelText}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      )}
                    </div>

                  <div className="flex flex-col items-center gap-4 bg-black/40 p-4 rounded-2xl border border-white/5 backdrop-blur-sm mb-4 flex-shrink-0">
                    <p className="text-emerald-400 font-black uppercase tracking-[0.2em] flex items-center gap-3 text-sm">
                      <CheckCircle size={20} /> Detection Complete
                    </p>
                    <div className="flex gap-4">
                      <button
                        onClick={(e) => { e.stopPropagation(); setUploadResult(null); setImageSize({ width: 0, height: 0 }); }}
                        className="text-[10px] text-gray-400 hover:text-white uppercase tracking-[0.2em] font-black border border-white/20 px-8 py-3 rounded-xl transition-all hover:bg-white/5 active:scale-95"
                      >
                        Scan Another
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); navigate('/dashboard'); }}
                        className="text-[10px] text-cyan-400 hover:text-cyan-300 uppercase tracking-[0.2em] font-black border border-cyan-400/30 px-8 py-3 rounded-xl flex items-center gap-2 transition-all hover:bg-cyan-400/5 shadow-[0_0_20px_rgba(34,211,238,0.1)] active:scale-95"
                      >
                        <LayoutDashboard size={14} /> View Dashboard
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 z-10">
                  <Upload size={48} className="text-gray-600 group-hover:text-cyan-400 transition-colors duration-500 opacity-60" />
                  <p className="text-gray-500 group-hover:text-cyan-400 font-mono text-sm tracking-widest font-bold transition-colors">
                    CLICK TO UPLOAD IMAGE OR VIDEO
                  </p>
                  {uploadError && <p className="text-red-400 font-bold text-xs tracking-widest">{uploadError}</p>}
                </div>
              )}
              {!uploadResult && !uploading && (
                <div className="absolute bottom-6 left-6 font-mono text-sm tracking-[0.2em] font-bold text-emerald-400 bg-black/80 border border-emerald-500/30 px-4 py-2 rounded-lg">
                  AI_SCANNER_ACTIVE <span className="animate-pulse">_</span>
                </div>
              )}
            </div>

            {/* Sidebar: recent detections */}
            <div className="w-full md:w-96 flex flex-col gap-6">
              <h3 className="font-black text-white uppercase tracking-[0.1em] border-b border-white/10 pb-4 flex items-center gap-3">
                <Activity size={20} className="text-red-500" /> Live Detections
              </h3>
              <div className="space-y-4 overflow-y-auto pr-2 custom-scrollbar">
                {uploadResult?.potholes?.length > 0 ? (
                  uploadResult.potholes.map((p, i) => (
                    <div
                      key={i}
                      onClick={() => handleCardClick(p.pothole_id)}
                      className="bg-white/5 p-5 rounded-xl border border-white/10 border-l-4 border-l-red-500 cursor-pointer hover:bg-white/10 transition-all hover:scale-[1.02] active:scale-95 group/card"
                    >
                      <div className="flex justify-between items-start mb-3">
                        <span className="text-xs font-mono font-bold text-gray-400">ID: PTH-{p.pothole_id?.toString().padStart(4, '0')}</span>
                        <span className={`text-[10px] font-black px-2.5 py-1 rounded uppercase tracking-widest border ${severityColor(p.severity)}`}>{p.severity}</span>
                      </div>
                      <p className="text-sm font-bold text-white uppercase tracking-wider">{uploadResult.road || 'Road detected'}</p>
                      <div className="flex justify-between items-center mt-3">
                        <p className="text-xs text-gray-400 font-bold uppercase tracking-widest flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-emerald-400" />
                          Conf: {Math.round(p.confidence * 100)}%
                        </p>
                        <Info size={14} className="text-gray-600 group-hover/card:text-cyan-400 transition-colors" />
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-600 font-bold text-xs uppercase tracking-widest text-center mt-8">Upload an image to see live detections</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: CAMERA FEEDS */}
        {activeTab === 'cameras' && (
          <div className="h-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[1, 2, 3, 4, 5, 6].map((cam) => (
              <div key={cam} className="bg-black/60 rounded-2xl border border-white/10 flex flex-col justify-between p-4 shadow-[inset_0_0_30px_rgba(0,0,0,0.8)] overflow-hidden relative group hover:border-cyan-500/50 transition-colors duration-500">
                <div className="flex justify-between text-xs text-cyan-400 font-mono font-bold tracking-[0.2em] mb-2 z-10 bg-black/80 border border-cyan-500/30 px-3 py-1.5 rounded-lg w-fit">
                  <span>CAM_0{cam}</span>
                </div>
                <div className="absolute top-5 right-5 z-10">
                  <span className="text-red-500 flex items-center gap-2 text-xs font-black tracking-widest bg-black/80 px-3 py-1.5 rounded-lg border border-red-500/30">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" /> REC
                  </span>
                </div>
                <div className="absolute inset-0 flex items-center justify-center text-gray-700 font-black tracking-[0.5em] text-xl group-hover:text-gray-600 transition-colors duration-700 mx-8 text-center">
                  NO SIGNAL DETECTED
                </div>
              </div>
            ))}
          </div>
        )}

        {/* TAB 3: DETECTION LOGS */}
        {activeTab === 'logs' && (
          <div className="h-full overflow-y-auto rounded-2xl border border-white/10 bg-black/40 custom-scrollbar">
            {logsLoading ? (
              <div className="flex items-center justify-center h-48">
                <div className="w-12 h-12 rounded-full border-4 border-t-cyan-500 border-r-blue-600 border-b-purple-600 border-l-transparent animate-spin"></div>
              </div>
            ) : (
              <table className="w-full text-left text-sm text-gray-300">
                <thead className="text-[10px] text-gray-400 uppercase tracking-[0.2em] bg-white/5 border-b border-white/10 sticky top-0 backdrop-blur-md">
                  <tr>
                    <th className="px-6 py-5 font-black">Date</th>
                    <th className="px-6 py-5 font-black">Anomaly ID</th>
                    <th className="px-6 py-5 font-black">Location</th>
                    <th className="px-6 py-5 font-black">Severity</th>
                    <th className="px-6 py-5 font-black text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 bg-transparent">
                  {logs.length === 0 ? (
                    <tr><td colSpan={5} className="px-6 py-12 text-center text-gray-600 font-bold uppercase tracking-widest">No detections yet. Upload your first pothole image.</td></tr>
                  ) : logs.map((f) => {
                    const p = f.properties;
                    return (
                      <tr key={p.id} className="hover:bg-white/5 transition-colors group" onClick={() => handleCardClick(p.id)}>
                        <td className="px-6 py-5 font-mono font-bold text-gray-500 group-hover:text-gray-300">{p.created_at}</td>
                        <td className="px-6 py-5 font-bold text-white tracking-widest">{p.pothole_id}</td>
                        <td className="px-6 py-5 font-mono text-xs tracking-wider">{p.road_name}</td>
                        <td className="px-6 py-5">
                          <span className={`font-black uppercase tracking-widest text-[10px] flex items-center gap-2 border px-3 py-1 rounded w-fit ${severityColor(p.severity)}`}>
                            <AlertTriangle size={12} /> {p.severity}
                          </span>
                        </td>
                        <td className="px-6 py-5 text-right font-bold text-cyan-400">{Math.round((p.confidence || 0) * 100)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>

      {/* Pothole Details Modal */}
      <AnimatePresence>
        {selectedPotholeId && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-xl bg-black/60"
            onClick={() => setSelectedPotholeId(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="w-full max-w-lg hackathon-glass rounded-3xl border border-white/20 overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)]"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-8">
                <div className="flex justify-between items-start mb-8">
                  <div>
                    <h2 className="text-2xl font-black text-white uppercase tracking-tighter mb-1">
                      Pothole Details
                    </h2>
                    <p className="text-cyan-400 font-mono text-xs font-bold tracking-[0.3em] uppercase">
                      PTH_ANALYTICS_V1.0
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedPotholeId(null)}
                    className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white"
                  >
                    <X size={24} />
                  </button>
                </div>

                {detailLoading ? (
                  <div className="py-20 flex flex-col items-center justify-center gap-4">
                    <div className="w-12 h-12 rounded-full border-4 border-t-cyan-500 border-r-blue-600 border-b-purple-600 border-l-transparent animate-spin"></div>
                    <p className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em]">Retrieving AI Geodata...</p>
                  </div>
                ) : potholeDetail ? (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                        <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <Hash size={10} className="text-cyan-500" /> Pothole ID
                        </p>
                        <p className="text-lg font-black text-white">{potholeDetail.id}</p>
                      </div>
                      <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                        <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <Activity size={10} className="text-red-500" /> Severity
                        </p>
                        <span className={`text-xs font-black uppercase tracking-widest px-2 py-0.5 rounded border ${severityColor(potholeDetail.severity)}`}>
                          {potholeDetail.severity}
                        </span>
                      </div>
                    </div>

                    <div className="bg-white/5 p-6 rounded-2xl border border-white/5 space-y-4">
                      <div className="flex items-start gap-4">
                        <div className="p-3 bg-cyan-500/10 rounded-xl border border-cyan-500/20 text-cyan-400">
                          <MapPin size={20} />
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between items-start mb-1">
                            <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Geolocation Data</p>
                            <div className="flex items-center gap-1.5 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                              <Globe size={10} className="text-cyan-400" />
                              <span className="text-[8px] font-black text-cyan-400 uppercase tracking-tighter">
                                {potholeDetail.location_source === 'image_exif' ? 'Image Metadata GPS' : 'Citizen Device GPS'}
                              </span>
                            </div>
                          </div>
                          <p className="text-sm font-bold text-white mb-2 leading-relaxed">{potholeDetail.address}</p>
                          <div className="grid grid-cols-2 gap-4 mt-4">
                            <div>
                              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">Latitude</p>
                              <p className="text-xs font-mono text-cyan-400">{potholeDetail.latitude?.toFixed(6)}</p>
                            </div>
                            <div>
                              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">Longitude</p>
                              <p className="text-xs font-mono text-cyan-400">{potholeDetail.longitude?.toFixed(6)}</p>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="border-t border-white/5 pt-4 grid grid-cols-2 gap-y-4">
                        <div>
                          <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1">Road</p>
                          <p className="text-xs font-bold text-white uppercase">{potholeDetail.road}</p>
                        </div>
                        <div>
                          <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1">City</p>
                          <p className="text-xs font-bold text-white uppercase">{potholeDetail.city}</p>
                        </div>
                        <div>
                          <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1">State</p>
                          <p className="text-xs font-bold text-white uppercase">{potholeDetail.state}</p>
                        </div>
                        <div>
                          <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1">Country</p>
                          <p className="text-xs font-bold text-white uppercase">{potholeDetail.country}</p>
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-between items-center bg-emerald-500/5 border border-emerald-500/20 p-4 rounded-2xl">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400 border border-emerald-500/20">
                          <CheckCircle size={20} />
                        </div>
                        <div>
                          <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest">AI Confidence</p>
                          <p className="text-lg font-black text-emerald-400">{Math.round((potholeDetail.confidence || 0) * 100)}%</p>
                        </div>
                      </div>
                      <div className="h-2 w-32 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(potholeDetail.confidence || 0) * 100}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-20 text-center">
                    <p className="text-red-400 font-bold text-xs tracking-widest uppercase">Error loading details</p>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}