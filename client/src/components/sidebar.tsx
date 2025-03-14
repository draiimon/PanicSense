import { useState } from 'react';
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { 
  BarChart3, 
  BrainCircuit,
  Clock, 
  LineChart, 
  Database,
  FileText,
  Activity,
  Info,
  Menu,
  X
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: JSX.Element;
}

export function Sidebar() {
  const [location] = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems: NavItem[] = [
    {
      href: "/dashboard",
      label: "Dashboard",
      icon: <BarChart3 className="h-5 w-5" />,
    },
    {
      href: "/emotion-analysis",
      label: "Emotion Analysis",
      icon: <BrainCircuit className="h-5 w-5" />,
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
      label: "Evaluation",
      icon: <FileText className="h-5 w-5" />,
    },
    {
      href: "/real-time",
      label: "Real-Time",
      icon: <Activity className="h-5 w-5" />,
    },
    {
      href: "/about",
      label: "About",
      icon: <Info className="h-5 w-5" />,
    },
  ];

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Logo Section */}
      <div className="p-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-blue-500 to-indigo-500 flex items-center justify-center">
            <BrainCircuit className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">PanicSense PH</h1>
            <p className="text-xs text-slate-400">Sentiment Analysis</p>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-4 pb-4">
        <div className="space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
            >
              <a
                className={cn(
                  "flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200",
                  location === item.href
                    ? "bg-gradient-to-r from-blue-600/20 to-indigo-600/20 text-white"
                    : "text-slate-400 hover:text-white hover:bg-white/10"
                )}
              >
                {item.icon}
                <span>{item.label}</span>
                {location === item.href && (
                  <motion.div
                    layoutId="activeNav"
                    className="absolute left-0 w-1 h-8 bg-blue-500 rounded-r-full"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
              </a>
            </Link>
          ))}
        </div>
      </nav>

      {/* Mobile Menu Button */}
      <button
        className="lg:hidden fixed bottom-4 right-4 p-4 rounded-full bg-blue-600 text-white shadow-lg"
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
      >
        {isMobileMenuOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <Menu className="h-6 w-6" />
        )}
      </button>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 bg-slate-900/90 z-50">
          <div className="h-full w-64 bg-gradient-to-b from-slate-900 to-slate-800">
            {/* Mobile Navigation */}
            <div className="p-4">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <a className={cn(
                    "flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200",
                    location === item.href
                      ? "bg-gradient-to-r from-blue-600/20 to-indigo-600/20 text-white"
                      : "text-slate-400 hover:text-white hover:bg-white/10"
                  )}>
                    {item.icon}
                    <span>{item.label}</span>
                  </a>
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}