import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { 
  BrainCircuit, 
  BarChart2, 
  Clock, 
  Layers, 
  Database, 
  ChartPie, 
  Activity, 
  HelpCircle,
  Menu, 
  User, 
  LogOut 
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Link, useLocation } from "wouter";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [location] = useLocation();

  const menuItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <BarChart2 className="w-4 h-4" /> },
    { path: '/emotion-analysis', label: 'Emotion Analysis', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/timeline', label: 'Timeline', icon: <Clock className="w-4 h-4" /> },
    { path: '/comparison', label: 'Comparison', icon: <Layers className="w-4 h-4" /> },
    { path: '/raw-data', label: 'Raw Data', icon: <Database className="w-4 h-4" /> },
    { path: '/evaluation', label: 'Evaluation', icon: <ChartPie className="w-4 h-4" /> },
    { path: '/real-time', label: 'Real-time', icon: <Activity className="w-4 h-4" /> },
    { path: '/about', label: 'About', icon: <HelpCircle className="w-4 h-4" /> }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 z-0">
        <div className="wave-animation"></div>
      </div>

      {/* Header with Dropdown Navigation */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed w-full top-0 bg-white/95 backdrop-blur-sm border-b border-slate-200 py-4 px-6 shadow-sm z-50"
      >
        <div className="max-w-[2000px] mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4">
              <div className="relative w-10 h-10">
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl"
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
                <BrainCircuit className="absolute inset-0 w-full h-full text-white p-2" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  PanicSense PH
                </h1>
                <p className="text-sm text-slate-500">
                  Real-time Analysis
                </p>
              </div>

              {/* Menu Button - Icon Only */}
              <Button
                variant="ghost"
                size="icon"
                className="w-10 h-10 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-all duration-200"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                <Menu className="w-5 h-5" />
              </Button>
            </div>

            {/* User Profile Section */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-blue-50 transition-colors cursor-pointer group">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <User className="w-4 h-4 text-white" />
                </div>
                <span className="text-sm font-medium text-slate-700 group-hover:text-blue-600">John Doe</span>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                className="flex items-center gap-2 hover:bg-red-50 hover:text-red-600 hover:border-red-200 transition-colors"
                onClick={() => console.log('Logout clicked')}
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </Button>
            </div>
          </div>

          {/* Dropdown Navigation Menu */}
          <AnimatePresence>
            {isMenuOpen && (
              <>
                <div 
                  className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
                  onClick={() => setIsMenuOpen(false)}
                />
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  className="absolute top-full left-0 right-0 bg-white/90 backdrop-blur-sm shadow-lg border-t border-slate-200 z-50"
                >
                  <nav className="max-w-7xl mx-auto p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {menuItems.map((item) => (
                        <Link 
                          key={item.path}
                          href={item.path}
                          onClick={() => setIsMenuOpen(false)}
                        >
                          <a className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                            location === item.path 
                              ? 'bg-blue-50 text-blue-600 shadow-sm' 
                              : 'hover:bg-blue-50/50 text-slate-600 hover:text-blue-600 hover:translate-x-1'
                          }`}>
                            {item.icon}
                            <span>{item.label}</span>
                          </a>
                        </Link>
                      ))}
                    </div>
                  </nav>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="relative z-10 flex-grow pt-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto bg-white/80 backdrop-blur-sm border-t border-slate-200 py-4 px-8 relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center text-sm text-slate-600">
          <div className="flex items-center space-x-2">
            <BrainCircuit className="h-5 w-5 text-blue-600" />
            <span>PanicSense PH © 2025</span>
          </div>
          <div className="mt-2 sm:mt-0">
            Advanced Disaster Sentiment Analysis Platform
          </div>
        </div>
      </footer>

      {/* Add wave animation styles */}
      <style>{`
        .wave-animation {
          position: absolute;
          width: 100%;
          height: 100%;
          background: linear-gradient(60deg, rgba(59, 130, 246, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
          animation: wave 8s ease-in-out infinite;
          background-size: 400% 400%;
        }

        .wave-animation::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(60deg, rgba(255, 255, 255, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
          animation: wave 8s ease-in-out infinite;
          animation-delay: -4s;
          background-size: 400% 400%;
        }

        @keyframes wave {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
      `}</style>
    </div>
  );
}