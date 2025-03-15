import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/auth-context";
import { motion } from "framer-motion";
import { BrainCircuit, BarChart2, Clock, Layers, Database, ChartPie, Activity, HelpCircle } from "lucide-react";

export function Header() {
  const [location] = useLocation();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    window.location.assign('/login');
  };

  const menuItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <BarChart2 className="w-4 h-4" /> },
    { path: '/emotion-analysis', label: 'Geographic Analysis', icon: <BrainCircuit className="w-4 h-4" /> },
    { path: '/timeline', label: 'Timeline', icon: <Clock className="w-4 h-4" /> },
    { path: '/comparison', label: 'Comparison', icon: <Layers className="w-4 h-4" /> },
    { path: '/raw-data', label: 'Raw Data', icon: <Database className="w-4 h-4" /> },
    { path: '/evaluation', label: 'Evaluation', icon: <ChartPie className="w-4 h-4" /> },
    { path: '/real-time', label: 'Real-time', icon: <Activity className="w-4 h-4" /> },
    { path: '/about', label: 'About', icon: <HelpCircle className="w-4 h-4" /> },
  ];

  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="sticky top-0 bg-blue-50 border-b border-slate-200 py-4 px-6 shadow-md z-50"
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
                Real-time Sentiment Analysis
              </p>
            </div>
          </div>

          {user && (
            <nav className="flex items-center space-x-1 overflow-x-auto pb-2 md:pb-0 scrollbar-hide">
              {menuItems.map((item) => (
                <Button
                  key={item.path}
                  variant={location === item.path ? 'default' : 'ghost'}
                  className="flex items-center space-x-2 whitespace-nowrap"
                  onClick={() => location !== item.path && window.location.assign(item.path)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </Button>
              ))}
            </nav>
          )}

          <div className="flex items-center space-x-4">
            {user ? (
              <div className="flex items-center space-x-4">
                <span className="text-sm text-slate-600">
                  Welcome, {user.username}
                </span>
                <Button variant="outline" onClick={handleLogout}>
                  Logout
                </Button>
              </div>
            ) : (
              <div className="flex space-x-2">
                <Button 
                  variant="outline" 
                  onClick={() => window.location.assign('/login')}
                >
                  Login
                </Button>
                <Button 
                  onClick={() => window.location.assign('/signup')}
                >
                  Sign up
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add custom scrollbar styles */}
      <style>{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </motion.header>
  );
}