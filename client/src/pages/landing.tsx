import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'wouter';
import { motion, useAnimation, AnimatePresence } from 'framer-motion';
import { ChevronRight, X, FileText, BarChart3, AlertTriangle, MapPin, Clock, Database, ArrowRight, Info, ExternalLink, Shield, Users, BellRing, Star, Award, Heart, Globe, Activity, ChevronLeft, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// Import tutorial images directly
import uploadDataImg from '../assets/upload-disaster-data.png';
import analyzeSentimentImg from '../assets/analyze-sentiment.png';
import geographicAnalysisImg from '../assets/geographic-analysis.png';
import realTimeMonitoringImg from '../assets/real-time-monitoring.png';

// Create a twinkling stars effect
const TwinklingStars = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {Array.from({ length: 50 }).map((_, i) => {
        const top = Math.random() * 100;
        const left = Math.random() * 100;
        const delay = Math.random() * 10;
        const size = Math.random() * 3 + 1;
        const duration = Math.random() * 4 + 3;
        
        return (
          <div 
            key={i}
            className="absolute rounded-full bg-white"
            style={{
              top: `${top}%`,
              left: `${left}%`,
              width: `${size}px`,
              height: `${size}px`,
              opacity: Math.random() * 0.7 + 0.3,
              animation: `twinkling ${duration}s infinite ${delay}s`
            }}
          />
        );
      })}
    </div>
  );
};

// Interactive Philippines Map Animation
const AnimatedMap = () => {
  return (
    <div className="absolute inset-0 w-full h-full opacity-20 pointer-events-none overflow-hidden">
      {/* Philippines map outline */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
        <svg viewBox="0 0 800 800" width="1200" height="1200" className="opacity-30">
          <path 
            d="M400,120 C420,160 500,140 520,170 C540,200 550,250 530,270 C510,290 490,320 500,350 C510,380 540,400 520,430 C500,460 480,480 450,500 C420,520 400,550 380,540 C360,530 340,490 320,480 C300,470 280,450 270,420 C260,390 240,370 250,340 C260,310 280,280 310,260 C340,240 350,200 380,170 C410,140 380,80 400,120 Z" 
            fill="none" 
            stroke="url(#philippinesGradient)" 
            strokeWidth="6"
            className="animate-pulse-slow"
          />
          
          {/* Islands and regions */}
          <path 
            d="M350,140 C360,150 380,130 390,150 C400,170 380,190 370,180 C360,170 340,130 350,140 Z" 
            fill="rgba(59, 130, 246, 0.3)" 
            stroke="rgba(59, 130, 246, 0.8)" 
            strokeWidth="2"
            className="animate-float-1"
          />
          
          <path 
            d="M420,220 C430,210 450,220 460,230 C470,240 480,260 470,270 C460,280 440,270 430,260 C420,250 410,230 420,220 Z" 
            fill="rgba(99, 102, 241, 0.3)" 
            stroke="rgba(99, 102, 241, 0.8)" 
            strokeWidth="2"
            className="animate-float-2"
          />
          
          <path 
            d="M380,300 C400,290 420,310 430,330 C440,350 430,380 410,390 C390,400 370,380 360,360 C350,340 360,310 380,300 Z" 
            fill="rgba(79, 70, 229, 0.3)" 
            stroke="rgba(79, 70, 229, 0.8)" 
            strokeWidth="2"
            className="animate-float-3"
          />
          
          <path 
            d="M330,400 C340,380 370,390 380,410 C390,430 380,460 360,470 C340,480 310,470 300,450 C290,430 320,420 330,400 Z" 
            fill="rgba(147, 51, 234, 0.3)" 
            stroke="rgba(147, 51, 234, 0.8)" 
            strokeWidth="2"
            className="animate-float-4"
          />
          
          {/* Connection lines between islands */}
          <g className="connection-lines">
            <path 
              d="M370,170 C390,210 410,250 400,280" 
              fill="none" 
              stroke="rgba(255, 255, 255, 0.3)" 
              strokeWidth="1" 
              strokeDasharray="5,5"
              className="animate-dash"
            />
            
            <path 
              d="M440,240 C430,270 410,300 390,320" 
              fill="none" 
              stroke="rgba(255, 255, 255, 0.3)" 
              strokeWidth="1" 
              strokeDasharray="5,5"
              className="animate-dash-reverse"
            />
            
            <path 
              d="M380,350 C360,380 340,410 330,430" 
              fill="none" 
              stroke="rgba(255, 255, 255, 0.3)" 
              strokeWidth="1" 
              strokeDasharray="5,5"
              className="animate-dash"
            />
          </g>
          
          {/* Pulse locations (disaster areas) */}
          <circle cx="370" cy="170" r="8" fill="rgba(239, 68, 68, 0.7)" className="animate-ping-slow" />
          <circle cx="440" cy="240" r="6" fill="rgba(245, 158, 11, 0.7)" className="animate-ping-slow delay-300" />
          <circle cx="380" cy="350" r="7" fill="rgba(59, 130, 246, 0.7)" className="animate-ping-slow delay-600" />
          <circle cx="330" cy="430" r="5" fill="rgba(139, 92, 246, 0.7)" className="animate-ping-slow delay-900" />
          
          {/* Data flow animations */}
          <g className="data-flow">
            <path 
              d="M370,170 Q420,200 440,240" 
              fill="none" 
              stroke="url(#dataFlow1)" 
              strokeWidth="2" 
              className="animate-data-flow"
            />
            
            <path 
              d="M440,240 Q410,290 380,350" 
              fill="none" 
              stroke="url(#dataFlow2)" 
              strokeWidth="2" 
              className="animate-data-flow delay-300"
            />
            
            <path 
              d="M380,350 Q355,390 330,430" 
              fill="none" 
              stroke="url(#dataFlow3)" 
              strokeWidth="2" 
              className="animate-data-flow delay-600"
            />
          </g>
          
          {/* Gradients for various elements */}
          <defs>
            <linearGradient id="philippinesGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#2563eb" />
              <stop offset="50%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#db2777" />
            </linearGradient>
            
            <linearGradient id="dataFlow1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2563eb">
                <animate attributeName="offset" from="-1" to="1" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="transparent">
                <animate attributeName="offset" from="0" to="2" dur="3s" repeatCount="indefinite" />
              </stop>
            </linearGradient>
            
            <linearGradient id="dataFlow2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#7c3aed">
                <animate attributeName="offset" from="-1" to="1" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="transparent">
                <animate attributeName="offset" from="0" to="2" dur="3s" repeatCount="indefinite" />
              </stop>
            </linearGradient>
            
            <linearGradient id="dataFlow3" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#db2777">
                <animate attributeName="offset" from="-1" to="1" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="transparent">
                <animate attributeName="offset" from="0" to="2" dur="3s" repeatCount="indefinite" />
              </stop>
            </linearGradient>
          </defs>
        </svg>
      </div>
      
      {/* Grid overlay for the tech feeling */}
      <div className="absolute inset-0 bg-grid-pattern-white opacity-5"></div>
      
      {/* Floating data points */}
      <div className="absolute top-1/4 left-1/3 h-2 w-2 rounded-full bg-blue-500 shadow-glow-blue animate-float-1"></div>
      <div className="absolute top-1/3 right-1/4 h-3 w-3 rounded-full bg-indigo-500 shadow-glow-indigo animate-float-2"></div>
      <div className="absolute bottom-1/4 left-1/4 h-2 w-2 rounded-full bg-purple-500 shadow-glow-purple animate-float-3"></div>
      <div className="absolute bottom-1/3 right-1/3 h-2 w-2 rounded-full bg-cyan-500 shadow-glow-cyan animate-float-4"></div>
    </div>
  );
};

// Animated text effect
const AnimatedText = ({ 
  text, 
  className = "", 
  delay = 0 
}: { 
  text: string; 
  className?: string; 
  delay?: number 
}) => {
  const controls = useAnimation();
  
  useEffect(() => {
    controls.start(i => ({
      opacity: 1,
      y: 0,
      transition: { delay: delay + i * 0.05 }
    }));
  }, [controls, delay]);
  
  return (
    <span className={`inline-block ${className}`}>
      {text.split("").map((char: string, i: number) => (
        <motion.span
          key={i}
          custom={i}
          initial={{ opacity: 0, y: 20 }}
          animate={controls}
          className="inline-block"
        >
          {char === " " ? "\u00A0" : char}
        </motion.span>
      ))}
    </span>
  );
};

// Floating elements effect
const FloatingElement = ({ 
  delay = 0, 
  duration = 3, 
  children, 
  className = "" 
}: { 
  delay?: number; 
  duration?: number; 
  children: React.ReactNode; 
  className?: string 
}) => {
  return (
    <motion.div
      initial={{ y: 0 }}
      animate={{
        y: [0, -10, 0],
        transition: {
          delay,
          duration,
          repeat: Infinity,
          repeatType: "reverse",
          ease: "easeInOut",
        }
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

// Feature carousel component
// Feature Carousel has been removed

// Live Time Counter for video effect
const LiveTimeCounter = () => {
  const [time, setTime] = useState("00:00:00");
  
  useEffect(() => {
    // Update time every second
    const interval = setInterval(() => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      const seconds = String(now.getSeconds()).padStart(2, '0');
      setTime(`${hours}:${minutes}:${seconds}`);
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <span className="video-time-counter">{time}</span>
  );
};

// Interactive Tutorial Component with better animations
const Tutorial = ({ onClose }: { onClose: () => void }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [direction, setDirection] = useState(0);
  const slideVariants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 500 : -500,
      opacity: 0
    }),
    center: {
      x: 0,
      opacity: 1
    },
    exit: (direction: number) => ({
      x: direction < 0 ? 500 : -500,
      opacity: 0
    })
  };
  
  const steps = [
    {
      title: "Upload Disaster Data",
      description: "Upload CSV files containing social media posts or messages about disasters to begin analysis.",
      icon: <FileText size={24} />,
      image: uploadDataImg
    },
    {
      title: "Analyze Sentiment",
      description: "The system automatically analyzes emotions and classifies each message using advanced AI models.",
      icon: <BarChart3 size={24} />,
      image: analyzeSentimentImg
    },
    {
      title: "Geographic Analysis",
      description: "View disaster locations plotted on interactive maps to identify affected areas.",
      icon: <MapPin size={24} />,
      image: geographicAnalysisImg
    },
    {
      title: "Real-time Monitoring",
      description: "Monitor new disaster reports in real-time for faster emergency response and coordination.",
      icon: <Clock size={24} />,
      image: realTimeMonitoringImg
    }
  ];
  
  const nextStep = () => {
    setDirection(1);
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
    }
  };
  
  const prevStep = () => {
    setDirection(-1);
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="bg-card text-card-foreground rounded-xl shadow-2xl w-full max-w-sm sm:max-w-md md:max-w-3xl lg:max-w-5xl relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 w-full h-1">
          <div 
            className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 h-full"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
        
        {/* X button removed */}
        
        {/* Responsive layout - Single column for mobile, dual column for desktop */}
        <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-blue-600 p-4 md:p-6 lg:p-8">          
          <div className="flex flex-wrap gap-2 items-center justify-center md:justify-start mt-1 mb-4 px-2">
            {steps.map((_, index) => {
              // Randomize position styling for each number (for mobile)
              const positionClasses = [
                "self-start", // Top position
                "self-center", // Center position
                "self-end",    // Bottom position
                "mt-3"         // Different top margin
              ];
              const randomPosition = positionClasses[index % positionClasses.length];
              
              return (
                <motion.div 
                  key={index}
                  initial={{ scale: 0.8, opacity: 0.5 }}
                  animate={{ 
                    scale: currentStep === index ? 1.2 : 0.85,
                    opacity: currentStep === index ? 1 : 0.5
                  }}
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    setDirection(index > currentStep ? 1 : -1);
                    setCurrentStep(index);
                  }}
                  className={`flex items-center justify-center rounded-full w-8 h-8 md:w-10 md:h-10 cursor-pointer transition-all ${randomPosition} md:self-auto ${
                    currentStep === index 
                      ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white font-bold shadow-lg md:shadow-xl' 
                      : 'bg-white/20 text-white/70 hover:bg-white/30'
                  }`}
                >
                  <span className="text-sm md:text-base">{index + 1}</span>
                </motion.div>
              );
            })}
          </div>
          
          <div className="flex flex-col md:flex-row items-center md:items-start md:gap-8 lg:gap-12">
            {/* Tutorial Image - Column 1 - WIDER but not too tall */}
            <div 
              className="tutorial-image-container slide-up-animation video-style-effect w-full max-w-[300px] md:max-w-[450px] lg:max-w-[500px] mx-auto md:mx-0 md:aspect-video overflow-hidden"
            >
              <div className="tutorial-dynamic-content h-full">
                <img 
                  src={steps[currentStep].image} 
                  alt={steps[currentStep].title} 
                  className="tutorial-image object-contain md:object-cover object-center w-full h-full transform md:scale-110 transition-all duration-500"
                />
                <div className="video-moving-overlay-1"></div>
                <div className="video-moving-overlay-2"></div>
                <div className="video-moving-overlay-3"></div>
              </div>
              <div className="tutorial-image-blur"></div>
              <div className="tutorial-image-glow"></div>
              
              <div className="video-scanlines"></div>
              <div className="video-noise"></div>
              
              {/* Step-specific overlays based on content */}
              {currentStep === 3 && (
                <div className="realtime-alert-banner">
                  <span className="realtime-alert-text">LIVE MONITORING</span>
                </div>
              )}
              
              {/* Time counter */}
              <div className="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded-full z-20">
                <LiveTimeCounter />
              </div>
              
              {/* Enhanced Desktop Content - visible on md screens and above */}
              <div className="hidden md:block absolute -bottom-3 -right-3 transform rotate-3 z-30">
                <div className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 py-1 px-3 shadow-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    <span className="text-white/80 text-xs">Advanced Analytics</span>
                  </div>
                </div>
              </div>
              
              {/* Floating indicators - only for desktop */}
              <div className="hidden md:block absolute left-0 top-1/4 -translate-x-3 z-20">
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ repeat: Infinity, duration: 3 }}
                  className="bg-blue-500/30 backdrop-blur-sm rounded-lg border border-blue-500/20 p-2"
                >
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                </motion.div>
              </div>
            </div>
            
            {/* Content Column - desktop only */}
            <div className="hidden md:flex flex-col justify-center items-start mt-8 md:mt-0 text-left flex-1">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="mb-6 md:mb-8"
              >
                <Badge className="mb-4 py-1.5 px-4 text-sm bg-indigo-500/20 text-indigo-200">
                  {currentStep === 0 ? 'Data Collection' : 
                   currentStep === 1 ? 'AI Analysis' :
                   currentStep === 2 ? 'Geographic Mapping' : 'Real-time Monitoring'}
                </Badge>
                
                <h2 className="text-2xl lg:text-3xl font-bold text-white mb-4">{steps[currentStep].title}</h2>
                <p className="text-white/70 lg:text-lg mb-6">{steps[currentStep].description}</p>
                
                {/* Extra details for desktop view */}
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      {steps[currentStep].icon}
                      <span className="text-white font-medium">Key Features</span>
                    </div>
                    <ul className="text-white/60 text-sm space-y-1">
                      {currentStep === 0 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Bulk CSV Processing</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Format Recognition</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Data Validation</li>
                        </>
                      )}
                      {currentStep === 1 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Emotion Classification</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Confidence Scoring</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Multilingual Support</li>
                        </>
                      )}
                      {currentStep === 2 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Interactive Maps</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Location Clustering</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Hotspot Analysis</li>
                        </>
                      )}
                      {currentStep === 3 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Live Updates</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Alert System</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div> Trend Detection</li>
                        </>
                      )}
                    </ul>
                  </div>
                  <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                      <Award className="h-4 w-4 text-amber-300" />
                      <span className="text-white font-medium">Benefits</span>
                    </div>
                    <ul className="text-white/60 text-sm space-y-1">
                      {currentStep === 0 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Fast Processing</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Error Correction</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Automated Handling</li>
                        </>
                      )}
                      {currentStep === 1 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Accurate Classification</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Language Detection</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Detailed Explanations</li>
                        </>
                      )}
                      {currentStep === 2 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Visual Understanding</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Impacted Area Analysis</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Resource Planning</li>
                        </>
                      )}
                      {currentStep === 3 && (
                        <>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Real-time Response</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Proactive Detection</li>
                          <li className="flex items-center gap-1.5"><div className="w-1 h-1 bg-amber-400 rounded-full"></div> Improved Coordination</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
        
        {/* Description for mobile view only */}
        <div className="p-5 pt-4 pb-16 md:hidden">
          <div className="px-4">
            <h3 className="font-bold text-center text-lg">
              {steps[currentStep].title}
            </h3>
            <p className="text-sm text-center text-gray-600 mt-1">
              {steps[currentStep].description}
            </p>
          </div>
        </div>
        
        {/* Only one set of navigation buttons at the bottom */}
        <div className="fixed left-0 right-0 bottom-0 flex justify-between py-4 px-6 bg-gradient-to-r from-blue-600/90 via-indigo-600/90 to-purple-600/90 backdrop-blur-sm">
          <Button 
            className={`${currentStep === 0 ? 'opacity-0 pointer-events-none' : ''} bg-white/20 hover:bg-white/30 text-white rounded-full flex items-center px-4`}
            onClick={prevStep}
            disabled={currentStep === 0}
            size="sm"
          >
            <ChevronLeft className="mr-1.5 h-4 w-4" />
            <span>Previous</span>
          </Button>
          
          <Button 
            onClick={nextStep}
            className="bg-white text-blue-600 hover:bg-white/90 hover:scale-105 transform transition-all shadow-lg rounded-full flex items-center px-4"
            size="sm"
          >
            <span>{currentStep === steps.length - 1 ? 'Get Started' : 'Next'}</span>
            <ChevronRight className="ml-1.5 h-4 w-4" />
          </Button>
        </div>
      </motion.div>
    </div>
  );
};

export default function LandingPage() {
  const [showTutorial, setShowTutorial] = useState(false);
  const parallaxRef = useRef<HTMLDivElement>(null);
  
  // Parallax effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!parallaxRef.current) return;
      
      const elements = parallaxRef.current.querySelectorAll('.parallax-element');
      
      const x = e.clientX / window.innerWidth;
      const y = e.clientY / window.innerHeight;
      
      elements.forEach((el) => {
        const speed = parseFloat((el as HTMLElement).dataset.speed || '0');
        const moveX = (x - 0.5) * speed;
        const moveY = (y - 0.5) * speed;
        
        (el as HTMLElement).style.transform = `translate(${moveX}px, ${moveY}px)`;
      });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);
  
  return (
    <div className="min-h-screen overflow-hidden bg-slate-50">
      {/* Header with same style as main pages */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-slate-200 shadow-lg py-3 px-4">
        <div className="max-w-[2000px] mx-auto flex justify-between items-center">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3"
          >
            <div className="relative w-11 h-11 sm:w-14 sm:h-14">
              <motion.div
                className="absolute inset-0 bg-gradient-to-br from-blue-600/30 via-indigo-600/30 to-purple-600/30 rounded-xl shadow-lg"
                animate={{
                  scale: [1, 1.02, 1],
                }}
                transition={{
                  duration: 0.5,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
              />
              <div className="absolute inset-0 w-full h-full flex items-center justify-center">
                <img src="/favicon.png" alt="PanicSense PH Logo" className="w-7 h-7 sm:w-9 sm:h-9 drop-shadow" />
              </div>
            </div>
            
            <div>
              <h1 className="text-xl sm:text-3xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent drop-shadow-sm">
                PanicSense PH
              </h1>
              <p className="text-sm sm:text-base text-slate-600 font-medium">
                Real-time Disaster Analysis
              </p>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex items-center space-x-4"
          >
            <Button 
              onClick={() => setShowTutorial(true)}
              className="relative overflow-hidden bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 text-white shadow-lg hover:shadow-xl transition-all rounded-full px-5 py-2.5"
            >
              <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-30 -translate-x-full animate-shimmer"/>
              <span className="relative flex items-center">
                Tutorial
                <ChevronRight className="ml-1.5 h-4 w-4" />
              </span>
            </Button>
          </motion.div>
        </div>
      </header>
      
      {/* Hero Section with Animated Background */}
      <section className="relative min-h-screen flex items-center pt-20">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-50 to-white dark:from-gray-900 dark:to-slate-900 overflow-hidden">
          <TwinklingStars />
          
          {/* Background geometric elements */}
          <div className="absolute bottom-0 left-0 w-full h-1/3 bg-grid-dark-pattern opacity-[0.03] dark:opacity-[0.07]"></div>
          
          {/* Add the Philippines map animation */}
          <AnimatedMap />
          
          {/* Add more visual elements */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-[800px] h-[800px] border-[1px] border-blue-400/10 rounded-full animate-spin-slow"></div>
            <div className="absolute w-[600px] h-[600px] border-[1px] border-indigo-500/10 rounded-full animate-spin-slower"></div>
            <div className="absolute w-[400px] h-[400px] border-[1px] border-purple-500/10 rounded-full animate-spin-reverse"></div>
          </div>
          
          {/* Animated floating bubbles */}
          <motion.div 
            className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"
            animate={{ 
              x: [0, 50, 0],
              y: [0, 30, 0]
            }}
            transition={{ 
              repeat: Infinity,
              duration: 15,
              ease: "easeInOut" 
            }}
          />
          
          <motion.div 
            className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl"
            animate={{ 
              x: [0, -50, 0],
              y: [0, -30, 0]
            }}
            transition={{ 
              repeat: Infinity,
              duration: 18,
              ease: "easeInOut",
              delay: 1 
            }}
          />
          
          {/* New animated elements */}
          <motion.div 
            className="absolute top-1/3 right-1/3 w-40 h-40 bg-cyan-500/10 rounded-full blur-2xl"
            animate={{ 
              x: [0, -30, 0],
              y: [0, 20, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              repeat: Infinity,
              duration: 10,
              ease: "easeInOut",
              delay: 0.5
            }}
          />
          
          <motion.div 
            className="absolute bottom-1/3 left-1/3 w-56 h-56 bg-pink-500/10 rounded-full blur-2xl"
            animate={{ 
              x: [0, 40, 0],
              y: [0, -25, 0],
              scale: [1, 1.05, 1]
            }}
            transition={{ 
              repeat: Infinity,
              duration: 12,
              ease: "easeInOut",
              delay: 1.5
            }}
          />
          
          {/* Particle effects */}
          <div className="absolute inset-0">
            {Array.from({ length: 15 }).map((_, i) => {
              const size = Math.random() * 6 + 2;
              const top = Math.random() * 100;
              const left = Math.random() * 100;
              const delay = Math.random() * 5;
              const duration = Math.random() * 15 + 10;
              
              return (
                <motion.div 
                  key={i}
                  className="absolute rounded-full bg-white/50 shadow-glow z-0"
                  style={{ 
                    width: size, 
                    height: size, 
                    top: `${top}%`, 
                    left: `${left}%` 
                  }}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ 
                    y: [0, -100, 0],
                    x: [0, Math.random() * 50 - 25, 0],
                    scale: [0, 1, 0],
                    opacity: [0, 0.5, 0]
                  }}
                  transition={{ 
                    repeat: Infinity,
                    duration,
                    delay,
                    ease: "easeInOut"
                  }}
                />
              );
            })}
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-6 relative z-10" ref={parallaxRef}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <motion.div 
                className="mb-6 inline-block"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <Badge className="py-1.5 px-4 text-sm bg-gradient-to-r from-blue-100 to-indigo-100 dark:from-blue-900/40 dark:to-indigo-900/40 text-blue-800 dark:text-blue-300 shadow-sm">
                  <Star className="h-3.5 w-3.5 mr-1" />
                  Next-Gen Disaster Intelligence
                </Badge>
              </motion.div>
              
              <motion.h1 
                className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white leading-none mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <span className="block">Advanced Disaster</span>
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-500 to-pink-500">
                  Monitoring & Analysis
                </span>
              </motion.h1>
              
              <motion.p 
                className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Real-time disaster monitoring for the Philippines using advanced NLP and sentiment analysis for faster emergency response and coordination.
              </motion.p>
              
              <motion.div 
                className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <motion.div
                  whileHover={{ 
                    scale: 1.05,
                    transition: { duration: 0.3 }
                  }}
                >
                  <Link href="/dashboard">
                    <Button 
                      size="lg"
                      className="relative overflow-hidden bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 text-white px-12 py-8 text-xl font-bold tracking-wide rounded-full"
                    >
                      <span className="relative flex items-center">
                        Get Started Now
                        <ArrowRight className="ml-3 h-5 w-5" />
                      </span>
                    </Button>
                  </Link>
                </motion.div>
              </motion.div>
              

            </div>
            
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3, type: "spring" }}
              className="rounded-2xl shadow-2xl overflow-hidden relative border border-gray-200 dark:border-gray-800 bg-gradient-to-br from-blue-50 to-indigo-50 p-8"
            >
              <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/10 via-transparent to-purple-500/10"></div>
              
              <motion.div 
                className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-200 dark:border-gray-700 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                  <h4 className="font-semibold">Live Disaster Monitoring</h4>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">Real-time sentiment tracking for faster emergency response and coordination in the Philippines.</p>
              </motion.div>
              
              <motion.div 
                className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-200 dark:border-gray-700 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
              >
                <div className="flex items-center mb-2">
                  <Globe className="h-4 w-4 text-blue-500 mr-2" />
                  <h4 className="font-semibold">Geographic Analysis</h4>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">Visual mapping of disaster events across the Philippine archipelago for strategic response.</p>
              </motion.div>
              
              <motion.div 
                className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-200 dark:border-gray-700"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9 }}
              >
                <div className="flex items-center mb-2">
                  <Activity className="h-4 w-4 text-purple-500 mr-2" />
                  <h4 className="font-semibold">Advanced Analytics</h4>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">Advanced sentiment analysis using modern language processing with Filipino language support.</p>
              </motion.div>
              
              <div className="absolute -bottom-2 -right-2 z-20">
                <motion.div
                  initial={{ scale: 0, rotate: -20 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ delay: 1.2, type: "spring" }}
                >
                  <Badge className="py-1.5 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg">
                    <Award className="h-3.5 w-3.5 mr-1.5" />
                    Premium Technology
                  </Badge>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>
      
      {/* Removed carousel as requested */}
      
      {/* Features Section with Animated Cards */}
      <section className="py-24 bg-white dark:bg-gray-900 relative overflow-hidden">
        <motion.div 
          className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
          initial={{ scaleX: 0, originX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, delay: 0.2 }}
        />
        
        <div className="max-w-7xl mx-auto px-6">
          <motion.div 
            className="text-center mb-20"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                Powerful
              </span> Features
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              PanicSense PH provides a comprehensive suite of tools to better understand and monitor disasters in the Philippines.
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Sentiment Analysis",
                description: "Automatically detect emotions and sentiment in disaster-related text using advanced processing models.",
                icon: <BarChart3 />,
                color: "blue",
                delay: 0
              },
              {
                title: "Disaster Classification",
                description: "Automatically identify and classify different types of disasters and emergencies.",
                icon: <AlertTriangle />,
                color: "red",
                delay: 0.1
              },
              {
                title: "Geographic Mapping",
                description: "Visual representation of disaster locations plotted on interactive maps.",
                icon: <MapPin />,
                color: "green",
                delay: 0.2
              },
              {
                title: "Real-time Monitoring",
                description: "Live monitoring of disaster reports from various sources for immediate response.",
                icon: <Clock />,
                color: "purple",
                delay: 0.3
              },
              {
                title: "Secure Data Storage",
                description: "Secure and scalable storage of disaster data with advanced search capabilities.",
                icon: <Database />,
                color: "orange",
                delay: 0.4
              },
              {
                title: "Multilingual Support",
                description: "Support for Filipino, English, and other regional languages used in the Philippines.",
                icon: <Info />,
                color: "indigo",
                delay: 0.5
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: feature.delay, duration: 0.5 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
                className="group"
              >
                <Card className="border-0 h-full shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden relative bg-gradient-to-b from-white to-gray-50 dark:from-gray-800 dark:to-gray-900">
                  <div className={`absolute top-0 left-0 w-full h-1 bg-${feature.color}-500 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300`}></div>
                  
                  <CardHeader className="pb-2">
                    <div className={`p-3 bg-${feature.color}-100 dark:bg-${feature.color}-900/30 rounded-xl w-fit mb-4 group-hover:scale-110 transition-transform duration-300 text-${feature.color}-600 dark:text-${feature.color}-400`}>
                      {feature.icon}
                    </div>
                    <CardTitle className="text-xl font-bold">{feature.title}</CardTitle>
                  </CardHeader>
                  
                  <CardContent>
                    <p className="text-gray-600 dark:text-gray-400">{feature.description}</p>
                    
                    <div className="mt-4 flex items-center text-sm font-medium text-blue-600 dark:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <span>Learn more</span>
                      <ArrowRight className="ml-1 h-4 w-4" />
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Statistics Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 via-indigo-700 to-purple-800 text-white relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-0 bg-grid-pattern-white opacity-10"></div>
          <TwinklingStars />
        </div>
        
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <motion.div 
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl font-bold mb-4">Powerful Impact Metrics</h2>
            <p className="text-xl text-white/80 max-w-2xl mx-auto">
              PanicSense PH is making a real difference in disaster response and preparedness
            </p>
          </motion.div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { value: "95%", label: "Accuracy in Disaster Classification", icon: <Shield /> },
              { value: "Fast", label: "Data Processing Speed", icon: <Users /> },
              { value: "24/7", label: "Real-time Monitoring Coverage", icon: <Clock /> },
              { value: "15min", label: "Average Response Time Reduction", icon: <BellRing /> }
            ].map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 hover:bg-white/20 transition-colors duration-300"
              >
                <div className="p-3 bg-white/10 rounded-full w-fit mb-4">
                  {stat.icon}
                </div>
                <h3 className="text-4xl font-bold mb-2">{stat.value}</h3>
                <p className="text-white/80">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      

      
      {/* Footer - Matching main layout style */}
      <footer className="bg-white border-t border-slate-200 py-2 sm:py-4 z-50 relative">
        <div className="max-w-[2000px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center text-xs sm:text-sm text-slate-600">
          <div className="flex items-center gap-1 sm:gap-2">
            <img src="/favicon.png" alt="PanicSense PH Logo" className="h-5 w-5 sm:h-6 sm:w-6" />
            <span>PanicSense PH © {new Date().getFullYear()}</span>
          </div>
          <div className="mt-1 sm:mt-0 flex flex-col sm:flex-row items-center gap-1 sm:gap-4">
            <span>Advanced Disaster Sentiment Analysis Platform</span>
            <div className="flex space-x-3">
              {["Dashboard", "Geographic Analysis", "About"].map((item, index) => (
                <Link 
                  key={index} 
                  href={item === "Dashboard" ? "/dashboard" : `/${item.toLowerCase().replace(/\s+/g, '-')}`}
                  className="text-slate-500 hover:text-blue-600 transition-colors text-xs"
                >
                  {item}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </footer>
      
      {/* Tutorial Modal */}
      <AnimatePresence>
        {showTutorial && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Tutorial onClose={() => setShowTutorial(false)} />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Inject CSS for animations */}
      <style>{`
        @keyframes shimmer {
          100% {
            transform: translateX(100%);
          }
        }
        
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        
        .skew-x-30 {
          transform: skewX(30deg);
        }
        @keyframes twinkling {
          0% { opacity: 0.3; }
          50% { opacity: 1; }
          100% { opacity: 0.3; }
        }
        
        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-15px); }
          100% { transform: translateY(0px); }
        }
        
        .animate-spin-slow {
          animation: spin 35s linear infinite;
        }
        
        .animate-spin-slower {
          animation: spin 45s linear infinite;
        }
        
        .animate-spin-reverse {
          animation: spin 30s linear infinite reverse;
        }
        
        .shadow-glow-blue {
          box-shadow: 0 0 15px 5px rgba(59, 130, 246, 0.5);
        }
        
        .shadow-glow-indigo {
          box-shadow: 0 0 15px 5px rgba(99, 102, 241, 0.5);
        }
        
        .shadow-glow-cyan {
          box-shadow: 0 0 15px 5px rgba(6, 182, 212, 0.5);
        }
        
        .shadow-glow-purple {
          box-shadow: 0 0 15px 5px rgba(168, 85, 247, 0.5);
        }
        
        .bg-grid-pattern {
          background-image: 
            linear-gradient(to right, #e5e7eb 1px, transparent 1px),
            linear-gradient(to bottom, #e5e7eb 1px, transparent 1px);
          background-size: 40px 40px;
        }
        
        .bg-grid-dark-pattern {
          background-image: 
            linear-gradient(to right, #4b5563 1px, transparent 1px),
            linear-gradient(to bottom, #4b5563 1px, transparent 1px);
          background-size: 40px 40px;
        }
        
        .bg-grid-pattern-white {
          background-image: 
            linear-gradient(to right, rgba(255,255,255,0.2) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.2) 1px, transparent 1px);
          background-size: 40px 40px;
        }
      `}</style>
    </div>
  );
}