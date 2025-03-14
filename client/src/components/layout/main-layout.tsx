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
      {/* Sidebar - Fixed width to prevent content shifting */}
      <div className="w-[280px] flex-shrink-0 hidden lg:block">
        <Sidebar />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-screen">
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

        {/* Main Content Area - Fixed height and scrollable */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <main className="flex-1 overflow-y-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              {children}
            </div>
          </main>

          {/* Footer - Always at bottom */}
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
      {/* Global Styles */}
      <style>{`
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(100, 116, 139, 0.2) transparent;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 5px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: rgba(100, 116, 139, 0.2);
          border-radius: 20px;
          transition: background-color 0.2s ease;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: rgba(100, 116, 139, 0.4);
        }
      `}</style>
    </div>
  );
}