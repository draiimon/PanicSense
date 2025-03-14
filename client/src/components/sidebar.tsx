import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'

export function Sidebar() {
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const location = useLocation()

  // Close mobile sidebar when route changes
  useEffect(() => {
    setIsMobileOpen(false)
  }, [location])

  return (
    <>
      {/* Mobile Menu Button */}
      <button 
        onClick={() => setIsMobileOpen(true)}
        className="lg:hidden fixed z-50 top-4 left-4 p-2.5 rounded-lg bg-white/95 backdrop-blur-sm shadow-lg border border-slate-200/60 text-slate-700 hover:bg-slate-50 transition-all"
        aria-label="Open Menu"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Sidebar Container */}
      <aside className={cn(
        "fixed top-0 left-0 z-40 h-screen transition-transform lg:translate-x-0 bg-white/95 backdrop-blur-sm border-r border-slate-200/60",
        "w-64 lg:w-72",
        isMobileOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        {/* Logo Section */}
        <div className="flex items-center gap-2 px-6 h-16 border-b border-slate-200/60">
          <img src="/logo.png" alt="PanicSense PH" className="w-8 h-8" />
          <span className="font-semibold text-lg">PanicSense PH</span>
        </div>

        {/* Navigation Links */}
        <nav className="flex flex-col gap-1 p-4">
          {[
            { path: '/', label: 'Dashboard', icon: 'graph' },
            { path: '/emotion', label: 'Emotion Analysis', icon: 'heart' },
            { path: '/timeline', label: 'Timeline', icon: 'clock' },
            { path: '/comparison', label: 'Comparison', icon: 'compare' },
            { path: '/raw-data', label: 'Raw Data', icon: 'database' },
            { path: '/metrics', label: 'Evaluation Metrics', icon: 'chart' },
            { path: '/realtime', label: 'Real-Time Mode', icon: 'live' },
            { path: '/about', label: 'About', icon: 'info' },
          ].map(({ path, label, icon }) => (
            <Link
              key={path}
              to={path}
              className={cn(
                "flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors",
                "hover:bg-slate-100",
                location.pathname === path 
                  ? "bg-slate-100 text-blue-600" 
                  : "text-slate-600 hover:text-slate-900"
              )}
            >
              <span className="flex-shrink-0 w-5 h-5">
                {getIcon(icon)}
              </span>
              {label}
            </Link>
          ))}
        </nav>
      </aside>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-30 lg:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </>
  )
}

// Helper function to render icons -  needs proper implementation based on icon library used
function getIcon(name: string) {
  // Replace with your actual icons. This is a placeholder.  You'll need to integrate your icon library here.
  switch (name) {
    case 'graph': return <BarChart3 className="h-5 w-5" />;
    case 'heart': return <Activity className="h-5 w-5" />;
    case 'clock': return <Clock className="h-5 w-5" />;
    case 'compare': return <LineChart className="h-5 w-5" />;
    case 'database': return <Database className="h-5 w-5" />;
    case 'chart': return <FileText className="h-5 w-5" />;
    case 'live': return <Timer className="h-5 w-5" />;
    case 'info': return <Info className="h-5 w-5" />;
    default: return <svg className="w-5 h-5" viewBox="0 0 24 24" />;
  }
}