import { useState, useEffect } from 'react';
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { 
  BarChart3, 
  FileText, 
  Clock, 
  LineChart, 
  Database,
  Activity,
  Info,
  Timer
} from "lucide-react";

interface SidebarProps {
  className?: string;
}

interface NavItem {
  href: string;
  label: string;
  icon: JSX.Element;
}

export function Sidebar({ className }: SidebarProps) {
  const [location] = useLocation();
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  // Close sidebar on route change on mobile
  useEffect(() => {
    if (isMobileOpen) {
      setIsMobileOpen(false);
    }
  }, [location]);

  const navItems: NavItem[] = [
    {
      href: "/",
      label: "Dashboard",
      icon: <BarChart3 className="h-5 w-5" />,
    },
    {
      href: "/emotion-analysis",
      label: "Emotion Analysis",
      icon: <Activity className="h-5 w-5" />,
    },
    {
      href: "/timeline",
      label: "Timeline",
      icon: <Clock className="h-5 w-5" />, 
    },
    {
      href: "/comparison",
      label: "Comparison",
      icon: <LineChart className="h-5 w-5" />,
    },
    {
      href: "/raw-data",
      label: "Raw Data",
      icon: <Database className="h-5 w-5" />,
    },
    {
      href: "/evaluation",
      label: "Evaluation Metrics",
      icon: <FileText className="h-5 w-5" />,
    },
    {
      href: "/real-time",
      label: "Real-Time Mode",
      icon: <Timer className="h-5 w-5" />,
    },
    {
      href: "/about",
      label: "About",
      icon: <Info className="h-5 w-5" />,
    },
  ];

  return (
    <>
      {/* Mobile sidebar button */}
      <button 
        onClick={() => setIsMobileOpen(true)}
        className="lg:hidden fixed z-50 top-4 left-4 p-2 rounded-md bg-white/90 backdrop-blur-sm shadow-sm border text-slate-700 hover:bg-slate-100"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 bg-slate-800 text-white w-[280px] z-50 transform transition-transform duration-300 shadow-xl",
          isMobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          className
        )}
      >
        <div className="flex items-center justify-between h-16 px-4 border-b border-slate-700">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
            </div>
            <h1 className="text-lg font-bold">PanicSense PH</h1>
          </div>
          <button 
            onClick={() => setIsMobileOpen(false)}
            className="lg:hidden"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <nav className="pt-5 px-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center space-x-2 py-2 px-3 rounded-md transition-colors duration-200",
                location === item.href
                  ? "bg-slate-700 text-white"
                  : "text-slate-300 hover:bg-slate-700 hover:text-white"
              )}
            >
              {item.icon}
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
      </div>
    </>
  );
}