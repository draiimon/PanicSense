import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { BrainCircuit, Menu, X } from "lucide-react";
import { motion } from "framer-motion";
import { useState } from "react";
import { Link, useLocation } from "wouter";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [location] = useLocation();

  const menuItems = [
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/emotion-analysis', label: 'Emotion Analysis' },
    { path: '/timeline', label: 'Timeline' },
    { path: '/comparison', label: 'Comparison' },
    { path: '/raw-data', label: 'Raw Data' },
    { path: '/evaluation', label: 'Evaluation' },
    { path: '/real-time', label: 'Real-time' },
    { path: '/about', label: 'About' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header with Dropdown Navigation */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-0 bg-white/80 backdrop-blur-sm border-b border-slate-200 px-8 py-4 shadow-sm z-50"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo and Menu Button */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="relative group"
            >
              <div className="relative w-10 h-10">
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl group-hover:scale-110 transition-transform"
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
            </button>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                PanicSense PH
              </h1>
              <p className="text-sm text-slate-500">
                Real-time Disaster Sentiment Analysis
              </p>
            </div>
          </div>

          {/* User Profile Section */}
          <div className="flex items-center space-x-4">
            <span className="text-sm text-slate-600">John Doe</span>
            <Button variant="outline" onClick={() => console.log('Logout clicked')}>
              Logout
            </Button>
          </div>
        </div>

        {/* Dropdown Navigation Menu */}
        {isMenuOpen && (
          <>
            <div 
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
              onClick={() => setIsMenuOpen(false)}
            />
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-full left-0 right-0 bg-white shadow-lg border-t border-slate-200 z-50"
            >
              <nav className="max-w-7xl mx-auto py-4 px-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
                  {menuItems.map((item) => (
                    <Link 
                      key={item.path}
                      href={item.path}
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <a className={`px-4 py-3 rounded-lg transition-colors ${
                        location === item.path 
                          ? 'bg-blue-50 text-blue-600' 
                          : 'hover:bg-slate-50 text-slate-600'
                      }`}>
                        {item.label}
                      </a>
                    </Link>
                  ))}
                </div>
              </nav>
            </motion.div>
          </>
        )}
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