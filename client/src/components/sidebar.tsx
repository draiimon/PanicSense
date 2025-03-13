
import { Link, useLocation } from "react-router-dom";
import { useTheme } from "@/context/theme-context";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface SidebarLinkProps {
  to: string;
  children: React.ReactNode;
  icon: React.ReactNode;
}

export function Sidebar() {
  const location = useLocation();
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const sidebarVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5,
        staggerChildren: 0.1
      }
    }
  };

  const linkVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: { 
      opacity: 1, 
      x: 0,
      transition: { 
        type: "spring", 
        stiffness: 100 
      }
    }
  };

  // Custom animated link component
  const SidebarLink = ({ to, children, icon }: SidebarLinkProps) => {
    const isActive = location.pathname === to;
    
    return (
      <motion.div variants={linkVariants}>
        <Link
          to={to}
          className={cn(
            "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all mb-1",
            isActive
              ? `${isDark 
                  ? 'bg-gradient-to-r from-blue-900/50 to-indigo-900/30 text-white border-r-4 border-blue-500' 
                  : 'bg-gradient-to-r from-blue-100 to-indigo-50 text-blue-900 border-r-4 border-blue-500'}`
              : `${isDark 
                  ? 'text-slate-300 hover:bg-slate-700/50 hover:text-white' 
                  : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'}`
          )}
        >
          <span className={cn(
            "flex h-6 w-6 items-center justify-center rounded-md",
            isActive 
              ? `${isDark ? 'text-blue-300' : 'text-blue-700'}` 
              : `${isDark ? 'text-slate-400' : 'text-slate-500'}`
          )}>
            {icon}
          </span>
          <span>{children}</span>
          
          {isActive && (
            <motion.div
              layoutId="sidebar-indicator"
              className="ml-auto h-1.5 w-1.5 rounded-full bg-blue-500"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            />
          )}
        </Link>
      </motion.div>
    );
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={sidebarVariants}
      className={cn(
        "fixed top-16 left-0 bottom-0 z-30 w-64 border-r p-4",
        isDark 
          ? "bg-slate-900 border-slate-800" 
          : "bg-white border-gray-200"
      )}
    >
      <div className="py-2">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mb-6 px-3"
        >
          <div className={`h-16 rounded-lg ${isDark ? 'bg-gradient-to-r from-blue-900/30 to-purple-900/20' : 'bg-gradient-to-r from-blue-50 to-indigo-50'} flex items-center justify-center mb-2`}>
            <h2 className={`text-xl font-bold ${isDark ? 'text-blue-300' : 'text-blue-700'} flex items-center`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              PanicSense PH
            </h2>
          </div>
          <div className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'} px-1`}>
            Advanced sentiment analysis
          </div>
        </motion.div>

        <div className="space-y-1">
          <SidebarLink to="/" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
            </svg>
          }>
            Dashboard
          </SidebarLink>
          
          <SidebarLink to="/timeline" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
            </svg>
          }>
            Timeline
          </SidebarLink>
          
          <SidebarLink to="/comparison" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
              <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
            </svg>
          }>
            Comparison
          </SidebarLink>
          
          <SidebarLink to="/metrics" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 0l-2 2a1 1 0 101.414 1.414L8 10.414l1.293 1.293a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          }>
            Metrics
          </SidebarLink>
          
          <SidebarLink to="/upload" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
          }>
            Upload Data
          </SidebarLink>
          
          <SidebarLink to="/config" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
          }>
            Settings
          </SidebarLink>
          
          <SidebarLink to="/about" icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          }>
            About
          </SidebarLink>
        </div>
      </div>

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className={`absolute bottom-4 left-4 right-4 p-3 rounded-lg ${
          isDark ? 'bg-blue-900/20 border border-blue-800/30' : 'bg-blue-50 border border-blue-100'
        }`}
      >
        <div className="flex items-center space-x-3">
          <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
            isDark ? 'bg-blue-700' : 'bg-blue-500'
          }`}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <h3 className={`text-xs font-semibold ${isDark ? 'text-blue-300' : 'text-blue-700'}`}>Model Status</h3>
            <p className={`text-xs ${isDark ? 'text-blue-200/70' : 'text-blue-600/80'}`}>
              Neural engine active
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
