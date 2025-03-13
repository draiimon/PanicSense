import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { 
  BarChart, 
  Globe, 
  Home, 
  Upload, 
  FileText, 
  MessageSquare, 
  AlertTriangle,
  Info,
  User
} from "lucide-react";

const menuItems = [
  { to: "/", icon: <Home className="w-5 h-5" />, label: "Dashboard" },
  { to: "/text-analysis", icon: <MessageSquare className="w-5 h-5" />, label: "Text Analysis" },
  { to: "/file-upload", icon: <Upload className="w-5 h-5" />, label: "File Upload" },
  { to: "/sentiment-data", icon: <FileText className="w-5 h-5" />, label: "Sentiment Data" },
  { to: "/emotion-analysis", icon: <BarChart className="w-5 h-5" />, label: "Emotion Analysis" },
  { to: "/disaster-tracking", icon: <AlertTriangle className="w-5 h-5" />, label: "Disaster Tracking" },
  { to: "/geo-mapping", icon: <Globe className="w-5 h-5" />, label: "Geo Mapping" },
  { to: "/about", icon: <Info className="w-5 h-5" />, label: "About" }
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  hidden: { opacity: 0, x: -20 },
  show: { opacity: 1, x: 0 }
};

export function Sidebar() {
  return (
    <motion.div 
      initial={{ x: -250 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="fixed left-0 top-0 pt-16 h-full w-64 hidden lg:block bg-gradient-to-b from-blue-800 to-indigo-900 text-white shadow-xl z-10 overflow-y-auto"
    >
      <div className="p-5">
        <motion.div 
          className="flex flex-col mb-6 items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full flex items-center justify-center shadow-lg mb-3">
            <User className="w-8 h-8 text-white" />
          </div>
          <div className="text-center">
            <p className="text-xs text-blue-200">Welcome to</p>
            <h3 className="font-bold text-lg">PanicSense PH</h3>
          </div>
        </motion.div>

        <div className="space-y-1">
          <p className="text-xs font-semibold text-blue-300 uppercase tracking-wider mb-3 px-3">Main Navigation</p>
          <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="space-y-1"
          >
            {menuItems.map((item, index) => (
              <motion.div key={item.to} variants={item}>
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center px-4 py-3 text-sm rounded-lg transition-all duration-200 ${
                      isActive 
                        ? "bg-gradient-to-r from-blue-600/40 to-indigo-500/40 text-white font-medium shadow-md" 
                        : "text-blue-100 hover:bg-white/10"
                    }`
                  }
                >
                  <span className="mr-3">{item.icon}</span>
                  <span>{item.label}</span>
                  {item.to === "/" && (
                    <span className="ml-auto bg-blue-500 text-xs rounded-full px-2 py-1">New</span>
                  )}
                </NavLink>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.5 }}
        className="absolute bottom-0 left-0 right-0 p-4"
      >
        <div className="bg-gradient-to-r from-blue-600/30 to-indigo-600/30 rounded-lg p-3 text-xs text-blue-100">
          <p className="font-medium mb-1">System Status: Active</p>
          <div className="w-full bg-blue-900/50 rounded-full h-1.5">
            <div className="bg-gradient-to-r from-blue-400 to-indigo-400 h-1.5 rounded-full" style={{ width: '93%' }}></div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}