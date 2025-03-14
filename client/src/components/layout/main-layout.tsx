import { ReactNode } from "react";
import { Sidebar } from "@/components/sidebar";
import { BrainCircuit } from "lucide-react";
import { motion } from "framer-motion";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Sidebar - Always visible on desktop */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-slate-200 shadow-lg hidden lg:block">
        <Sidebar />
      </aside>

      {/* Main Content - With padding for sidebar */}
      <div className="flex-1 lg:ml-64">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="sticky top-0 bg-white/80 backdrop-blur-sm border-b border-slate-200 px-8 py-4 shadow-sm z-20"
        >
          <div className="flex items-center space-x-4 max-w-7xl mx-auto">
            <div className="flex items-center">
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
              <div className="ml-3">
                <motion.h1 
                  className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  PanicSense PH
                </motion.h1>
                <motion.p 
                  className="text-sm text-slate-600"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  Real-time Disaster Sentiment Analysis
                </motion.p>
              </div>
            </div>
          </div>
        </motion.header>

        {/* Main Content Area */}
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
    </div>
  );
}