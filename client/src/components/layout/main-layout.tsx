
import { ReactNode, useState, useEffect } from "react";
import { Sidebar } from "@/components/sidebar";
import { motion, AnimatePresence } from "framer-motion";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden flex-col bg-gradient-to-br from-gray-50 to-blue-50">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white shadow-md z-10 flex justify-between items-center px-6 py-3"
      >
        <motion.h1 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600"
        >
          PanicSense PH
        </motion.h1>
        
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="flex items-center space-x-4"
        >
          <div className="hidden md:flex space-x-1">
            <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse"></span>
            <span className="text-sm text-gray-600">System Active</span>
          </div>
        </motion.div>
      </motion.div>
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <AnimatePresence>
          {isLoaded && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="flex-1 overflow-auto pl-0 lg:pl-64"
            >
              <main className="px-6 sm:px-8 lg:px-10 py-8 flex-grow">
                {children}
              </main>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
