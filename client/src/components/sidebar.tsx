import { useState, useEffect } from 'react';
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BarChart3, 
  FileText, 
  Clock, 
  LineChart, 
  Database,
  Activity,
  Info,
  Timer,
  Menu,
  X
} from "lucide-react";

interface SidebarProps {
  className?: string;
}

interface NavItem {
  href: string;
  label: string;
  icon: JSX.Element;
}

const sidebarVariants = {
  open: {
    x: 0,
    opacity: 1,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  },
  closed: {
    x: "-100%",
    opacity: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  }
};

export function Sidebar({ className }: SidebarProps) {
  const [location] = useLocation();
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  useEffect(() => {
    if (isMobileOpen) {
      setIsMobileOpen(false);
    }
  }, [location]);

  const navItems: NavItem[] = [
    {
      href: "/dashboard",
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
      label: "Evaluation",
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
      {/* Mobile Menu Button - Always Visible */}
      <motion.button 
        onClick={() => setIsMobileOpen(true)}
        className="lg:hidden fixed z-50 top-4 left-4 p-2 rounded-md bg-white/90 backdrop-blur-sm shadow-lg border border-gray-200 text-slate-700 hover:bg-slate-100 transition-all duration-300"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Menu className="h-6 w-6" />
      </motion.button>

      {/* Backdrop */}
      <AnimatePresence>
        {isMobileOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
            onClick={() => setIsMobileOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        variants={sidebarVariants}
        initial={false}
        animate={isMobileOpen ? "open" : "closed"}
        className={cn(
          "fixed inset-y-0 left-0 bg-gradient-to-b from-slate-800 to-slate-900 text-white w-[280px] z-50 shadow-xl lg:translate-x-0",
          className
        )}
      >
        {/* Logo Section */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-slate-700/50">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-500 to-indigo-500 flex items-center justify-center shadow-lg">
              <motion.svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-5 w-5"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200 }}
                viewBox="0 0 20 20" 
                fill="currentColor"
              >
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </motion.svg>
            </div>
            <h1 className="text-lg font-bold bg-gradient-to-r from-blue-200 to-indigo-200 bg-clip-text text-transparent">
              PanicSense PH
            </h1>
          </div>

          {/* Close Button - Mobile Only */}
          <motion.button 
            onClick={() => setIsMobileOpen(false)}
            className="lg:hidden text-slate-400 hover:text-white transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <X className="h-6 w-6" />
          </motion.button>
        </div>

        {/* Navigation */}
        <nav className="pt-5 px-4 space-y-1">
          <AnimatePresence>
            {navItems.map((item, index) => (
              <motion.div
                key={item.href}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-2 py-2 px-3 rounded-md transition-all duration-200 relative",
                    location === item.href
                      ? "bg-gradient-to-r from-blue-600/20 to-indigo-600/20 text-white"
                      : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                  )}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {location === item.href && (
                    <motion.div
                      layoutId="activeNav"
                      className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-blue-500 rounded-r-full"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </Link>
              </motion.div>
            ))}
          </AnimatePresence>
        </nav>
      </motion.aside>
    </>
  );
}