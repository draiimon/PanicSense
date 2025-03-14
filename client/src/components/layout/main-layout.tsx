import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { BrainCircuit, Menu, ChevronDown, User, LogOut } from "lucide-react";
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
    { path: '/dashboard', label: 'Dashboard', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/emotion-analysis', label: 'Emotion Analysis', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/timeline', label: 'Timeline', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/comparison', label: 'Comparison', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/raw-data', label: 'Raw Data', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/evaluation', label: 'Evaluation', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/real-time', label: 'Real-time', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/about', label: 'About', icon: <BrainCircuit className="w-4 h-4" /> }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header with Dropdown Navigation */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-0 bg-white/80 backdrop-blur-sm border-b border-slate-200 py-4 px-6 shadow-sm z-50"
      >
        <div className="max-w-[2000px] mx-auto">
          <div className="flex items-center justify-between">
            {/* Logo and Menu Button */}
            <div className="flex items-center gap-6">
              {/* Logo Section */}
              <div className="flex items-center gap-3">
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
              </div>

              {/* Menu Button */}
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-2 px-3 py-2 hover:bg-slate-100 transition-colors"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                <Menu className="w-5 h-5" />
                <span className="font-medium">Menu</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isMenuOpen ? 'rotate-180' : ''}`} />
              </Button>
            </div>

            {/* User Profile Section */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <span className="text-sm font-medium text-slate-700">John Doe</span>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                className="flex items-center gap-2"
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
                              : 'hover:bg-slate-50 text-slate-600 hover:text-slate-900'
                          }`}>
                            {item.icon}
                            <span className="font-medium">{item.label}</span>
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
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto bg-white/80 backdrop-blur-sm border-t border-slate-200 py-4 px-8">
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
    </div>
  );
}