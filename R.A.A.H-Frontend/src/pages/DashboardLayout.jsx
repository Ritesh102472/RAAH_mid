import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Home,
  LayoutDashboard,
  Activity,
  Map as MapIcon,
  AlertTriangle,
  BarChart3,
  Cpu,
  Settings
} from 'lucide-react';
import NotificationToast from '../components/NotificationToast';
import { useState, useCallback, useEffect } from 'react';
import { useWebSocketContext } from '../context/WebSocketContext';

export default function DashboardLayout() {
  const location = useLocation();
  const { user } = useAuth();

  const [notifications, setNotifications] = useState([]);

  const clearNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const addNotification = useCallback((notif) => {
    const id = Date.now();
    setNotifications((prev) => [{ ...notif, id }, ...prev].slice(0, 3)); // Keep last 3
    setTimeout(() => clearNotification(id), 6000);
  }, [clearNotification]);

  const { subscribe } = useWebSocketContext();

  useEffect(() => {
    const unsubscribe = subscribe((message) => {
      console.log('📡 Global message:', message);
      if (message.event === 'new_pothole') {
        addNotification({
          event: message.event,
          message: `${message.data?.count || 1} new pothole(s) detected. Updating system data...`,
          data: message.data
        });
      } else if (message.event === 'discovery_complete') {
        addNotification({
          event: message.event,
          message: message.data?.message || 'Global discovery scan complete.',
          data: message.data
        });
      }
    });
    return unsubscribe;
  }, [subscribe, addNotification]);

  const isAdmin = user?.role === 'admin' || user?.role === 'superadmin';

  const navItems = isAdmin
    ? [
      { name: 'Dashboard', path: '/dashboard', icon: <LayoutDashboard size={20} /> },
      { name: 'Complaints', path: '/complaints', icon: <AlertTriangle size={20} /> },
    ]
    : [
      { name: 'Dashboard', path: '/dashboard', icon: <LayoutDashboard size={20} /> },
      { name: 'Live Monitoring', path: '/monitoring', icon: <Activity size={20} /> },
      { name: 'Map', path: '/map', icon: <MapIcon size={20} /> },
      { name: 'Analytics', path: '/analytics', icon: <BarChart3 size={20} /> },
      { name: 'AI System', path: '/ai-system', icon: <Cpu size={20} /> },
      { name: 'Settings', path: '/settings', icon: <Settings size={20} /> },
    ];

  return (
    <div className="flex h-screen overflow-hidden hackathon-bg text-gray-100 font-sans">

      {/* Decorative Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20 z-0">
        <div className={`absolute top-1/4 left-1/4 w-96 h-96 ${isAdmin ? 'bg-red-600/20' : 'bg-blue-600/20'} rounded-full blur-[120px]`}></div>
        <div className={`absolute bottom-1/4 right-1/4 w-96 h-96 ${isAdmin ? 'bg-orange-600/20' : 'bg-cyan-600/20'} rounded-full blur-[120px]`}></div>
      </div>

      {/* Extreme Dark Glass Sidebar */}
      <aside className={`w-64 hackathon-glass ${isAdmin ? 'bg-black/80 border-red-500/20 shadow-[10px_0_40px_rgba(255,0,0,0.15)]' : 'border-white/10 shadow-[10px_0_30px_rgba(0,0,0,0.5)]'} border-r flex flex-col z-20`}>
        {/* Brand Logo Area */}
        <div className="h-28 flex flex-col items-center justify-center px-4 border-b border-white/5 pt-4">
          <img
            src="/logo.png"
            alt="RAAH Logo"
            className="w-56 object-contain drop-shadow-[0_2px_10px_rgba(0,0,0,0.1)] mb-1"
          />
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 py-8 px-4 space-y-3">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path || (location.pathname === '/' && item.path === '/dashboard');
            return (
              <Link
                key={item.name}
                to={item.path}
                className={`flex items-center gap-4 px-4 py-3 rounded-lg transition-all duration-300 group relative overflow-hidden ${isActive
                  ? 'bg-blue-900/40 border border-blue-400/30 text-white font-bold shadow-[0_0_20px_rgba(59,130,246,0.1)]'
                  : 'text-gray-400 hover:bg-white/5 hover:text-gray-100 font-semibold border border-transparent'
                  }`}
              >
                {isActive && (
                  <div className="absolute left-0 top-0 h-full w-1 bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
                )}
                <div className={`${isActive ? 'text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.4)]' : 'text-gray-500 group-hover:text-cyan-600 transition-colors'}`}>
                  {item.icon}
                </div>
                <span className="tracking-wide text-sm uppercase">{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Bottom Profile Area */}
        <div className="p-6 border-t border-white/5 bg-black/20 backdrop-blur-md">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white font-black text-lg shadow-[0_0_15px_rgba(56,189,248,0.4)] border border-cyan-400/50">
              {user?.name?.[0]?.toUpperCase() || 'U'}
            </div>
            <div>
              <p className="text-sm font-black text-gray-100 tracking-wider uppercase">{user?.name || 'User Portal'}</p>
              <p className="text-xs text-cyan-400 flex items-center gap-2 font-bold mt-1 tracking-widest uppercase">
                <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse"></span>
                {user?.role || 'citizen'}
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col relative overflow-y-auto z-10 w-full scroll-smooth">

        {/* Top Header (Glassmorphic) */}
        <header className={`h-28 hackathon-glass border-b ${isAdmin ? 'border-red-500/20 bg-black/60' : 'border-white/10'} flex items-center justify-between px-10 sticky top-0 z-30 shadow-[0_10px_30px_rgba(0,0,0,0.5)]`}>
          <div className="flex flex-col">
            <h2 className={`text-3xl font-black ${isAdmin ? 'text-red-500' : 'text-gray-800'} uppercase tracking-[0.2em] flex items-center gap-4 drop-shadow-[0_2px_4px_rgba(255,255,255,0.8)]`}>
              <div className={`w-2.5 h-10 ${isAdmin ? 'bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.6)]' : 'bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.6)]'} rounded-full`}></div>
              {location.pathname.replace('/', '').replace(/-/g, ' ') || 'Dashboard'}
            </h2>
            {isAdmin && <span className="text-[10px] font-black text-red-400 tracking-[0.4em] uppercase mt-1 pl-6">Admin Command Center</span>}
          </div>

          <Link to="/">
            <button className="flex items-center gap-2 bg-white/50 backdrop-blur-md hover:bg-white/70 text-gray-800 font-black px-5 py-2.5 rounded-xl transition-all duration-300 border border-orange-300/60 hover:border-orange-400 hover:shadow-[0_0_15px_rgba(251,146,60,0.3)] hover:scale-[1.03]">
              <Home size={17} className="text-orange-500" />
              <span className="uppercase tracking-widest text-xs">Return Home</span>
            </button>
          </Link>
        </header>

        {/* Page Content Container */}
        <div className="p-8 md:p-10 flex-1 relative w-full h-full">
          <Outlet />
        </div>
      </main>

      <NotificationToast 
        notifications={notifications} 
        clearNotification={clearNotification} 
      />
    </div>
  );
}