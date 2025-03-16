import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/auth-context";
import { motion, AnimatePresence } from "framer-motion";
import {
  BrainCircuit,
  BarChart2,
  Clock,
  Layers,
  Database,
  ChartPie,
  Activity,
  HelpCircle,
  Globe,
  User,
  ChevronDown,
} from "lucide-react";
import { useState } from "react";

export function Header() {
  const [location] = useLocation();
  const { user, logout } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleLogout = () => {
    logout();
    window.location.assign("/login");
  };

  const menuItems = [
    {
      path: "/dashboard",
      label: "Dashboard",
      icon: <BarChart2 className="w-4 h-4" />,
    },
    {
      path: "/emotion-analysis",
      label: "Geographic Analysis",
      icon: <Globe className="w-4 h-4" />,
    },
    {
      path: "/timeline",
      label: "Timeline",
      icon: <Clock className="w-4 h-4" />,
    },
    {
      path: "/comparison",
      label: "Comparison",
      icon: <Layers className="w-4 h-4" />,
    },
    {
      path: "/raw-data",
      label: "Raw Data",
      icon: <Database className="w-4 h-4" />,
    },
    {
      path: "/evaluation",
      label: "Evaluation",
      icon: <ChartPie className="w-4 h-4" />,
    },
    {
      path: "/real-time",
      label: "Real-time",
      icon: <Activity className="w-4 h-4" />,
    },
    {
      path: "/about",
      label: "About",
      icon: <HelpCircle className="w-4 h-4" />,
    },
  ];

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed w-full top-0 bg-gradient-to-r from-white/95 via-blue-50/95 to-indigo-50/95 backdrop-blur-sm border-b border-slate-200 shadow-lg z-50"
    >
      <div className="max-w-[2000px] mx-auto">
        <div className="flex items-center justify-between px-3 py-3 sm:px-8 sm:py-5">
          {/* Left side - Logo and Title */}
          <div className="flex items-center gap-3 sm:gap-5">
            <div className="relative w-10 h-10 sm:w-12 sm:h-12">
              <motion.div
                className="absolute inset-0 bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 rounded-xl shadow-lg"
                animate={{
                  scale: [1, 1.1, 1],
                  rotate: [0, 5, -5, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
              />
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                <BrainCircuit className="absolute inset-0 w-full h-full text-white p-2 drop-shadow" />
              </motion.div>
            </div>
            <div>
              <motion.h1
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="text-lg sm:text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent"
              >
                DisasterSense AI
              </motion.h1>
            </div>
          </div>

          {/* Center - Navigation Dropdown */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              className="flex items-center gap-2 hover:bg-blue-50"
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            >
              Navigate
              <motion.div
                animate={{ rotate: isDropdownOpen ? 180 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronDown className="w-4 h-4" />
              </motion.div>
            </Button>

            <AnimatePresence>
              {isDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute top-full right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-100 py-2"
                >
                  {menuItems.map((item) => (
                    <motion.a
                      key={item.path}
                      href={item.path}
                      className={`flex items-center gap-3 px-4 py-2 text-sm transition-colors ${
                        location === item.path
                          ? "bg-blue-50 text-blue-600"
                          : "text-gray-600 hover:bg-gray-50"
                      }`}
                      whileHover={{ x: 5 }}
                    >
                      {item.icon}
                      {item.label}
                    </motion.a>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right side - Profile */}
          <div className="flex items-center gap-2 sm:gap-4">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center">
                <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
              </div>
              <span className="text-xs sm:text-sm font-medium text-slate-700 hidden sm:inline">
                {user?.name || "User"}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 sm:h-9 sm:w-auto rounded-full hover:bg-red-50 hover:text-red-600 transition-all duration-200"
              onClick={handleLogout}
            >
              <span className="hidden sm:inline ml-2">Logout</span>
            </Button>
          </div>
        </div>
      </div>
    </motion.header>
  );
}