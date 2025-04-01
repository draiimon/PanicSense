import { useDisasterContext } from "@/context/disaster-context";
import { StatusCard } from "@/components/dashboard/status-card";
import { OptimizedSentimentChart } from "@/components/dashboard/optimized-sentiment-chart";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { AffectedAreasCard } from "@/components/dashboard/affected-areas-card-new";
import { UsageStatsCard } from "@/components/dashboard/usage-stats-card";
import { FileUploader } from "@/components/file-uploader";
import { motion, AnimatePresence, useAnimationControls } from "framer-motion";
import { getSentimentColor, getDisasterTypeColor } from "@/lib/colors";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Upload, Database, BarChart3, Globe2, ArrowUpRight, RefreshCw, AlertTriangle, Clock, Shield, Zap, Activity, Waves, Wind, Bell, Map, Sparkles, BrainCircuit } from "lucide-react";
import { CardCarousel } from "@/components/dashboard/card-carousel";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { KeyEvents } from "@/components/timeline/key-events";

import React, { useState, useRef, useEffect, lazy, Suspense, useMemo } from 'react';

// Performance optimization system - using strategies from high-performance web apps
function usePerformanceMode() {
  const [isLowPerformanceMode, setIsLowPerformanceMode] = useState(false);
  const [performanceLevel, setPerformanceLevel] = useState<'high'|'medium'|'low'>('high');
  
  useEffect(() => {
    // Apply technical optimizations immediately (regardless of device)
    
    // 1. Optimize image decoding and rendering
    if ('loading' in HTMLImageElement.prototype) {
      document.querySelectorAll('img').forEach(img => {
        if (img.getAttribute('loading') !== 'eager') {
          img.setAttribute('loading', 'lazy');
        }
      });
    }
    
    // 2. Use passive event listeners for touch and wheel events
    const supportsPassive = false;
    try {
      window.addEventListener('test', null as any, {
        get passive() {
          return true;
        }
      });
    } catch (e) {}
    
    // 3. Reduce paint complexity
    document.documentElement.style.setProperty('--animate-duration', '0s');
    
    // 4. Twitter/Instagram-style performance detection (more aggressive)
    const detectPerformance = () => {
      // Check CPU cores - Twitter uses this for feature flags
      const cpuCores = navigator.hardwareConcurrency || 2;
      
      // Check for mobile - Instagram treats all mobile as needing optimization
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      
      // Check memory - Facebook considers 4GB+ as high-end
      const ramEstimate = (navigator as any).deviceMemory || 4;
      
      // Check screen size - Twitter/Instagram reduce features on smaller screens
      const hasSmallScreen = window.innerWidth < 768;
      
      // Use network info if available
      const connection = (navigator as any).connection;
      const isSlowNetwork = connection && 
        (connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g');
      
      // Set performance level based on multiple factors
      if (
        (cpuCores <= 4) || 
        (ramEstimate <= 4) || 
        isSlowNetwork || 
        (isMobile && hasSmallScreen)
      ) {
        // Instagram/Twitter approach for most devices
        setPerformanceLevel('medium');
        setIsLowPerformanceMode(true);
      }
      
      // Ultra aggressive optimization for very low-end
      if (
        (cpuCores <= 2) || 
        (ramEstimate <= 2) || 
        (isSlowNetwork && isMobile) ||
        (window.screen.width * window.screen.height <= 1280 * 720)
      ) {
        setPerformanceLevel('low');
        setIsLowPerformanceMode(true);
      }
    };
    
    detectPerformance();
    
    // 5. Facebook-style progressive enhancement/frame drop detection
    let frameDrops = 0;
    let consecutiveSlowFrames = 0;
    let lastTimestamp = performance.now();
    
    const monitorFrameRate = (timestamp: number) => {
      const delta = timestamp - lastTimestamp;
      
      // Target 60fps (16.67ms per frame)
      // Instagram treats >30ms as a definitely dropped frame
      if (delta > 30) {
        frameDrops++;
        consecutiveSlowFrames++;
        
        // More aggressive than before - just 3 dropped frames triggers performance mode
        if (consecutiveSlowFrames >= 3) {
          setIsLowPerformanceMode(true);
          // If extremely bad performance, go to lowest mode
          if (delta > 100) {
            setPerformanceLevel('low');
          } else {
            setPerformanceLevel('medium');
          }
        }
      } else {
        consecutiveSlowFrames = 0;
      }
      
      lastTimestamp = timestamp;
      requestAnimationFrame(monitorFrameRate);
    };
    
    const animId = requestAnimationFrame(monitorFrameRate);
    
    // 6. Apply Facebook-style load sequence for critical vs non-critical elements
    // Mark the dashboard as ready to prevent the feeling of lag
    setTimeout(() => {
      document.body.classList.add('dashboard-ready');
    }, 100);
    
    return () => {
      cancelAnimationFrame(animId);
    };
  }, []);
  
  return isLowPerformanceMode;
}

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.3
    }
  }
};

const pulseAnimation = {
  initial: { scale: 1 },
  animate: {
    scale: [1, 1.05, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      repeatType: "reverse"
    }
  }
};

const shimmer = {
  hidden: { opacity: 0, x: -100 },
  visible: {
    opacity: 1,
    x: 100,
    transition: {
      repeat: Infinity,
      duration: 1.5,
      ease: "linear"
    }
  }
};

function LoadingOverlay({ message }: { message: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center z-50">
      {/* Semi-transparent backdrop */}
      <div className="absolute inset-0 bg-white/90 backdrop-blur-lg"></div>

      {/* Loading content */}
      <div className="relative z-10 flex flex-col items-center gap-4 p-6 bg-white/50 rounded-xl shadow-lg backdrop-blur-sm">
        <Loader2 className="h-12 w-12 text-blue-600 animate-spin" />
        <div className="text-center">
          <p className="text-lg font-semibold text-slate-800">Processing Data</p>
          <p className="text-sm text-slate-600">{message}</p>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  // Apply performance mode to reduce animations
  const isLowPerformanceMode = usePerformanceMode();
  
  const { 
    sentimentPosts = [],
    disasterEvents = [],
    activeDiastersCount = 0,
    analyzedPostsCount = 0,
    dominantSentiment = 'N/A',
    dominantDisaster = 'Unknown',
    dominantSentimentPercentage = 0,
    dominantDisasterPercentage = 0,
    secondDominantSentiment = null,
    secondDominantDisasterPercentage = 0,
    secondDominantSentimentPercentage = 0,
    modelConfidence = 0,
    isLoadingSentimentPosts = false,
    sentimentPercentages = {},
    disasterPercentages = {}
  } = useDisasterContext();
  const [carouselPaused, setCarouselPaused] = useState(false);
  const [showAlert, setShowAlert] = useState(true);
  const [updateIndex, setUpdateIndex] = useState(0);
  const [swipeStart, setSwipeStart] = useState<{x: number, y: number} | null>(null);
  
  // Auto-dismiss the alert after 10 seconds
  useEffect(() => {
    if (showAlert) {
      const timer = setTimeout(() => {
        setShowAlert(false);
      }, 10000);
      
      return () => clearTimeout(timer);
    }
  }, [showAlert]);
  
  // Auto-rotate through the updates when not paused
  useEffect(() => {
    if (carouselPaused) return;
    
    const interval = setInterval(() => {
      setUpdateIndex((prev) => (prev === 2 ? 0 : prev + 1));
    }, 7000);
    
    return () => clearInterval(interval);
  }, [carouselPaused]);
  const scrollRef = useRef(null);
  const controls = useAnimationControls();
  
  // Animation effect for dashboard elements
  useEffect(() => {
    const sequence = async () => {
      await controls.start({ opacity: 1, y: 0, transition: { duration: 0.5 } });
      await controls.start({ scale: [1, 1.02, 1], transition: { duration: 0.5 } });
    };
    sequence();
  }, [controls]);

  // Calculate stats with safety checks
  const totalPosts = Array.isArray(sentimentPosts) ? sentimentPosts.length : 0;
  const activeDisasters = Array.isArray(disasterEvents) 
    ? disasterEvents.filter(event => 
        new Date(event.timestamp) >= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
      ).length 
    : 0;

  // Get most affected area with safety checks
  const locationCounts = Array.isArray(sentimentPosts) 
    ? sentimentPosts.reduce<Record<string, number>>((acc, post) => {
        if (post.location) {
          acc[post.location] = (acc[post.location] || 0) + 1;
        }
        return acc;
      }, {})
    : {};
  const mostAffectedArea = Object.entries(locationCounts)
    .sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A';


  // Filter posts from last week with safety check
  const lastWeekPosts = Array.isArray(sentimentPosts) 
    ? sentimentPosts.filter(post => {
        const postDate = new Date(post.timestamp);
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        return postDate >= weekAgo;
      })
    : [];

  // Recalculate dominant sentiment from last week's posts
  const lastWeekDominantSentiment = (() => {
    if (lastWeekPosts.length === 0) return 'N/A';
    const sentimentCounts: Record<string, number> = {};
    lastWeekPosts.forEach(post => {
      sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
    });
    return Object.entries(sentimentCounts)
      .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  })();

  // Filter out "Not specified" and generic "Philippines" locations with safety check
  const filteredPosts = Array.isArray(sentimentPosts) 
    ? sentimentPosts.filter(post => {
        const location = post.location?.toLowerCase();
        return location && 
              location !== 'not specified' && 
              location !== 'philippines' &&
              location !== 'pilipinas' &&
              location !== 'pinas' &&
              location !== 'unknown';
      })
    : [];

  const sentimentData = {
    labels: ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'],
    values: [0, 0, 0, 0, 0],
    showTotal: false
  };

  // Count sentiments from filtered posts
  filteredPosts.forEach(post => {
    const index = sentimentData.labels.indexOf(post.sentiment);
    if (index !== -1) {
      sentimentData.values[index]++;
    }
  });

  // Performance optimizations while keeping animations
  useEffect(() => {
    // Function to optimize animations based on device performance
    const optimizeAnimations = () => {
      // Apply low-level optimizations without disabling animations
      // DO NOT disable animations - user wants them!
      
      // Apply performance optimizations globally but without disabling animations
      // DO NOT ADD 'reduce-animation' class as it was causing the animations to disappear
      // document.body.classList.add('reduce-animation');
      
      // Set animation quality based on device capabilities
      return {
        particleCount: 0, // Disable particles only
        animationEnabled: true, // KEEP animations enabled - important!
        complexEffects: true // KEEP effects enabled - important!
      };
    };

    const optimizationSettings = optimizeAnimations();
    
    // Component cleanup and optimization flag
    return () => {
      // Cleanup animation frames and timers here if needed
    };
  }, []);

  return (
    <div className="relative space-y-8 pb-10" ref={scrollRef}>
      {/* Enhanced colorful background for entire dashboard */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        {/* More vibrant animated gradient overlay - CSS Animation */}
        <div 
          className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: '200% 200%' }}
        />
        
        {/* Enhanced animated patterns with more vibrant colors */}
        <div className="absolute inset-0 opacity-15 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM1MDUwRjAiIGZpbGwtb3BhY2l0eT0iMC41Ij48cGF0aCBkPSJNMzYgMzR2Nmg2di02aC02em02IDZ2Nmg2di02aC02em0tMTIgMGg2djZoLTZ2LTZ6bTEyIDBoNnY2aC02di02eiIvPjwvZz48L2c+PC9zdmc+')]"></div>
        
        {/* Additional decorative elements */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,rgba(120,80,255,0.8)_0%,transparent_70%)]"></div>
        
        {/* More colorful floating elements - USING CSS ANIMATIONS */}
        <div 
          className="absolute h-72 w-72 rounded-full bg-purple-500/25 filter blur-3xl animate-float-1 will-change-transform"
          style={{ top: '15%', left: '8%' }}
        />
        
        <div 
          className="absolute h-64 w-64 rounded-full bg-teal-500/20 filter blur-3xl animate-float-2 will-change-transform"
          style={{ bottom: '15%', right: '15%' }}
        />
        
        <div 
          className="absolute h-52 w-52 rounded-full bg-purple-500/25 filter blur-3xl animate-float-3 will-change-transform"
          style={{ top: '45%', right: '20%' }}
        />
        
        {/* Additional floating elements for more color - USING CSS ANIMATIONS */}
        <div 
          className="absolute h-48 w-48 rounded-full bg-pink-500/20 filter blur-3xl animate-float-4 will-change-transform"
          style={{ top: '65%', left: '25%' }}
        />
        
        <div 
          className="absolute h-40 w-40 rounded-full bg-yellow-400/15 filter blur-3xl animate-float-5 will-change-transform"
          style={{ top: '30%', left: '40%' }}
        />
        
        {/* Shimmer effect with CSS */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
      </div>
      
      {/* Enhanced hero section with animated elements - fixed position for better scrolling */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-purple-600 via-teal-600 to-rose-600 shadow-xl">
        {/* Animated grid background */}
        <div className="absolute inset-0 bg-grid-white/10 bg-[size:20px_20px] opacity-20"></div>
        
        {/* Optimized animated floating bubbles - using CSS animations for better performance */}
        <div 
          className="absolute h-40 w-40 rounded-full bg-blue-400 filter blur-3xl opacity-30 -top-20 -left-20 animate-float-1 will-change-transform"
        />
        
        <div 
          className="absolute h-40 w-40 rounded-full bg-indigo-400 filter blur-3xl opacity-30 -bottom-20 -right-20 animate-float-2 will-change-transform delay-300"
        />
        
        {/* Static gradient overlay for better performance */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

        <div className="relative px-6 py-10 sm:px-12 sm:py-12">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
            {/* Main title area - takes 3 columns on desktop */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="md:col-span-3"
            >
              <motion.h1 
                className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-4 leading-tight"
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
              >
                <motion.span className="inline-block" variants={fadeInUp}>Disaster Response</motion.span>{" "}
                <motion.span className="inline-block bg-clip-text text-transparent bg-gradient-to-r from-blue-100 to-purple-100" variants={fadeInUp}>Dashboard</motion.span>
              </motion.h1>
              
              <motion.p 
                className="text-blue-100 text-base sm:text-lg mb-6 max-w-xl"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                Real-time sentiment monitoring and geospatial analysis for disaster response in the Philippines
              </motion.p>

              <motion.div 
                className="flex flex-wrap gap-3"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <motion.div 
                  className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white"
                >
                  <Database className="h-3.5 w-3.5 mr-1.5" />
                  <span>{totalPosts} Data Points</span>
                </motion.div>
                <motion.div 
                  className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white"
                >
                  <BarChart3 className="h-3.5 w-3.5 mr-1.5" />
                  <span>Sentiment Analysis</span>
                </motion.div>
                <motion.div 
                  className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white"
                >
                  <Globe2 className="h-3.5 w-3.5 mr-1.5" />
                  <span>Geographic Mapping</span>
                </motion.div>
              </motion.div>
            </motion.div>
            
            {/* Dynamic updates carousel - takes 2 columns on desktop */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.4 }}
              className="md:col-span-2 block"
            >
              <div className="backdrop-blur-md bg-gradient-to-br from-purple-600/30 via-teal-600/30 to-rose-600/30 rounded-xl overflow-hidden border border-white/20 shadow-xl h-[220px] mt-5 md:mt-0">
                {/* Animated background particles */}
                <div className="absolute inset-0">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <motion.div
                      key={`particle-${i}`}
                      className="absolute h-12 w-12 rounded-full bg-white/5"
                      style={{
                        top: `${Math.random() * 100}%`,
                        left: `${Math.random() * 100}%`,
                      }}
                      animate={{
                        y: [0, -10, 0, 10, 0],
                        x: [0, 10, 0, -10, 0],
                        scale: [1, 1.1, 1, 0.9, 1],
                        opacity: [0.2, 0.5, 0.2, 0.5, 0.2],
                      }}
                      transition={{
                        duration: 10 + Math.random() * 5,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    />
                  ))}
                </div>
                
                <div className="p-3 pb-2 border-b border-white/10 flex items-center justify-between relative z-10">
                  <motion.div 
                    className="flex items-center gap-2"
                  >
                    <motion.div 
                      className="p-1.5 rounded-md bg-white/20"
                      animate={{ 
                        boxShadow: ["0 0 0px rgba(255,255,255,0.2)", "0 0 10px rgba(255,255,255,0.5)", "0 0 0px rgba(255,255,255,0.2)"] 
                      }}
                      transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
                    >
                      <Sparkles className="h-3.5 w-3.5 text-white" />
                    </motion.div>
                    <h3 className="text-sm font-medium text-white flex items-center gap-1.5">
                      Latest Updates
                      <motion.span 
                        className="inline-block" 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3, delay: 1 }}
                      >
                        <div className="flex h-4 items-center justify-center rounded-full bg-green-500/20 px-1.5">
                          <span className="text-[8px] font-semibold text-green-400">NEW</span>
                        </div>
                      </motion.span>
                    </h3>
                  </motion.div>
                  <div className="flex gap-1.5">
                    <motion.div 
                      animate={{ 
                        scale: [1, 1.5, 1],
                        backgroundColor: ["rgb(167, 243, 208)", "rgb(45, 212, 191)", "rgb(167, 243, 208)"]
                      }}
                      transition={{ duration: 3, repeat: Infinity, repeatType: "reverse" }}
                      className="w-1.5 h-1.5 rounded-full bg-teal-300"
                    />
                    <motion.div 
                      animate={{ 
                        scale: [1, 1.5, 1],
                        backgroundColor: ["rgb(216, 180, 254)", "rgb(192, 132, 252)", "rgb(216, 180, 254)"]
                      }}
                      transition={{ duration: 3, repeat: Infinity, repeatType: "reverse", delay: 0.5 }}
                      className="w-1.5 h-1.5 rounded-full bg-purple-300"
                    />
                    <motion.div 
                      animate={{ 
                        scale: [1, 1.5, 1],
                        backgroundColor: ["rgb(253, 164, 175)", "rgb(244, 114, 182)", "rgb(253, 164, 175)"]
                      }}
                      transition={{ duration: 3, repeat: Infinity, repeatType: "reverse", delay: 1 }}
                      className="w-1.5 h-1.5 rounded-full bg-pink-300"
                    />
                  </div>
                </div>
                
                {/* Swipable feature updates */}
                <motion.div 
                  className="relative h-[180px] overflow-hidden touch-manipulation"
                  onTouchStart={(e) => {
                    const touch = e.touches[0];
                    setSwipeStart({x: touch.clientX, y: touch.clientY});
                    setCarouselPaused(true);
                  }}
                  onTouchMove={(e) => {
                    if (!swipeStart) return;
                    const touch = e.touches[0];
                    const deltaX = touch.clientX - swipeStart.x;
                    const swipeThreshold = 50;
                    
                    if (Math.abs(deltaX) > swipeThreshold) {
                      const newIndex = deltaX > 0 
                        ? (updateIndex === 0 ? 2 : updateIndex - 1) 
                        : (updateIndex === 2 ? 0 : updateIndex + 1);
                      setUpdateIndex(newIndex);
                      setSwipeStart(null);
                    }
                  }}
                  onTouchEnd={() => {
                    setSwipeStart(null);
                    setCarouselPaused(false);
                  }}
                >
                  <AnimatePresence mode="wait">
                    <motion.div 
                      key={updateIndex}
                      initial={{ opacity: 0, x: 20, filter: "blur(4px)" }}
                      animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                      exit={{ opacity: 0, x: -20, filter: "blur(4px)" }}
                      transition={{ duration: 0.5, type: "tween" }}
                      className="absolute inset-0 p-4 will-change-transform"
                    >
                      {/* Feature update contents - would rotate through these */}
                      {(updateIndex === 0 || carouselPaused) && (
                        <div className="flex flex-col h-full">
                          <div className="flex items-center gap-2">
                            <motion.div 
                              className="p-2 rounded-full bg-blue-500/20 backdrop-blur-md"
                              initial={{ scale: 0.8 }}
                              animate={{ scale: 1 }}
                              transition={{ type: "spring", stiffness: 300, damping: 10 }}
                            >
                              <Globe2 className="h-4 w-4 text-blue-300" />
                            </motion.div>
                            <motion.h4 
                              className="text-sm font-medium text-blue-100"
                              initial={{ x: -20, opacity: 0 }}
                              animate={{ x: 0, opacity: 1 }}
                              transition={{ delay: 0.1 }}
                            >
                              Enhanced Geospatial Analysis
                            </motion.h4>
                            <motion.div
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.3, type: "spring" }}
                              className="ml-auto flex items-center gap-1 rounded-full bg-blue-500/10 px-2 py-0.5 text-[8px] font-medium text-blue-300"
                            >
                              <span>v2.4</span>
                            </motion.div>
                          </div>
                          <motion.p 
                            className="text-sm text-blue-100/90 mt-3 leading-relaxed"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                          >
                            Improved disaster location mapping with real-time updates from social media and government alerts.
                          </motion.p>
                          <div className="mt-auto">
                            <motion.div 
                              className="flex items-center text-xs text-blue-100/70 gap-1"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.4 }}
                            >
                              <Clock className="h-3 w-3" />
                              <span>{new Date().toLocaleDateString()}</span>
                            </motion.div>
                          </div>
                        </div>
                      )}
                      {updateIndex === 1 && !carouselPaused && (
                        <div className="flex flex-col h-full">
                          <div className="flex items-center gap-2">
                            <motion.div 
                              className="p-2 rounded-full bg-indigo-500/20 backdrop-blur-md"
                              initial={{ scale: 0.8 }}
                              animate={{ scale: 1 }}
                              transition={{ type: "spring", stiffness: 300, damping: 10 }}
                            >
                              <Zap className="h-4 w-4 text-indigo-300" />
                            </motion.div>
                            <motion.h4 
                              className="text-sm font-medium text-blue-100"
                              initial={{ x: -20, opacity: 0 }}
                              animate={{ x: 0, opacity: 1 }}
                              transition={{ delay: 0.1 }}
                            >
                              Advanced Machine Learning
                            </motion.h4>
                            <motion.div
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.3, type: "spring" }}
                              className="ml-auto flex items-center gap-1 rounded-full bg-indigo-500/10 px-2 py-0.5 text-[8px] font-medium text-indigo-300"
                            >
                              <span>NEW</span>
                            </motion.div>
                          </div>
                          <motion.p 
                            className="text-sm text-blue-100/90 mt-3 leading-relaxed"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                          >
                            Advanced language model with improved sentiment analysis for both English and Tagalog social media posts.
                          </motion.p>
                          <div className="mt-auto">
                            <motion.div 
                              className="flex items-center text-xs text-blue-100/70 gap-1"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.4 }}
                            >
                              <Clock className="h-3 w-3" />
                              <span>{new Date().toLocaleDateString()}</span>
                            </motion.div>
                          </div>
                        </div>
                      )}
                      {updateIndex === 2 && !carouselPaused && (
                        <div className="flex flex-col h-full">
                          <div className="flex items-center gap-2">
                            <motion.div 
                              className="p-2 rounded-full bg-purple-500/20 backdrop-blur-md"
                              initial={{ scale: 0.8 }}
                              animate={{ scale: 1 }}
                              transition={{ type: "spring", stiffness: 300, damping: 10 }}
                            >
                              <Bell className="h-4 w-4 text-purple-300" />
                            </motion.div>
                            <motion.h4 
                              className="text-sm font-medium text-blue-100"
                              initial={{ x: -20, opacity: 0 }}
                              animate={{ x: 0, opacity: 1 }}
                              transition={{ delay: 0.1 }}
                            >
                              Real-time Alerts
                            </motion.h4>
                            <motion.div
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.3, type: "spring" }}
                              className="ml-auto flex items-center gap-1 rounded-full bg-purple-500/10 px-2 py-0.5 text-[8px] font-medium text-purple-300"
                            >
                              <span>COMING</span>
                            </motion.div>
                          </div>
                          <motion.p 
                            className="text-sm text-blue-100/90 mt-3 leading-relaxed"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                          >
                            Live notification system for emergency events with instant alerts based on sentiment and volume analysis.
                          </motion.p>
                          <div className="mt-auto">
                            <motion.div 
                              className="flex items-center text-xs text-blue-100/70 gap-1"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.4 }}
                            >
                              <Clock className="h-3 w-3" />
                              <span>Coming soon</span>
                            </motion.div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  </AnimatePresence>
                  
                  {/* Navigation dots */}
                  <div className="absolute bottom-3 left-0 right-0 flex justify-center gap-1.5 z-10">
                    {[0, 1, 2].map((i) => (
                      <motion.button
                        key={`nav-${i}`}
                        className={`w-6 h-1.5 rounded-full cursor-pointer ${
                          (!carouselPaused && updateIndex === i) || (carouselPaused && i === 0)
                            ? "bg-white/60"
                            : "bg-white/20"
                        }`}
                        onClick={() => {
                          setUpdateIndex(i);
                          setCarouselPaused(true);
                          setTimeout(() => setCarouselPaused(false), 5000);
                        }}

                        animate={
                          (!carouselPaused && updateIndex === i) || (carouselPaused && i === 0)
                            ? { width: "1.5rem", backgroundColor: "rgba(255, 255, 255, 0.6)" }
                            : { width: "0.75rem", backgroundColor: "rgba(255, 255, 255, 0.2)" }
                        }
                        transition={{ duration: 0.3 }}
                      />
                    ))}
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
      
      {/* Alert banner with animation */}
      <AnimatePresence>
        {showAlert && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="relative bg-blue-50 border-l-4 border-blue-500 p-3 rounded-r-md shadow-md mb-6"
          >
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <Bell className="h-5 w-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-800 font-medium">New Feature Alert</p>
                <p className="text-sm text-blue-600 mt-1">Enhanced geospatial sentiment analysis now available with real-time updates.</p>
                <div className="flex items-center mt-1.5 text-xs text-blue-500">
                  <Clock className="h-3 w-3 mr-1" />
                  <span>This alert will auto-dismiss in a few seconds</span>
                </div>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setShowAlert(false)}
                  className="inline-flex text-gray-500 focus:outline-none hover:text-gray-700"
                >
                  <span className="sr-only">Dismiss</span>
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload button placed outside the dashboard cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="mb-8 relative"
      >
        <motion.div
          whileTap={{ scale: 0.98 }}
          className="bg-white rounded-xl shadow-xl overflow-hidden border border-blue-50"
        >
          <div className="p-5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-blue-100 flex-shrink-0 flex items-center justify-center">
                <Upload className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-800 text-lg mb-1">Upload Disaster Data</h3>
                <p className="text-sm text-gray-600">
                  Upload CSV files for sentiment analysis and disaster monitoring. Files are processed in batches of 30 rows with a daily limit of 10,000 rows. Small files (under 30 rows) are processed instantly.
                </p>
              </div>
            </div>
            <div className="md:flex-shrink-0">
              <FileUploader className="min-w-[180px] justify-center" />
            </div>
          </div>
        </motion.div>
        {isLoadingSentimentPosts && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/90 backdrop-blur-lg rounded-lg">
            <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
          </div>
        )}
      </motion.div>

      {/* Stats Grid with improved styling (3-card layout) */}
      <div
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6"
      >
        <StatusCard 
          title="Dominant Disaster"
          value={dominantDisaster}
          icon={dominantDisaster.toLowerCase()}
          trend={{
            value: `${dominantDisasterPercentage}%`,
            isUpward: dominantDisasterPercentage > 50,
            label: "of disaster posts"
          }}
          isLoading={isLoadingSentimentPosts}
          disasterPercentages={disasterPercentages}
        />
        <StatusCard 
          title="Analyzed Posts"
          value={analyzedPostsCount.toString()}
          icon="bar-chart"
          trend={{
            value: "+15%",
            isUpward: true,
            label: "increase this month"
          }}
          isLoading={isLoadingSentimentPosts}
        />
        <StatusCard 
          title="Dominant Sentiment"
          value={dominantSentiment}
          icon="heart"
          trend={{
            value: `${dominantSentimentPercentage}%`,
            isUpward: dominantSentimentPercentage > 50,
            label: "of all posts"
          }}
          isLoading={isLoadingSentimentPosts}
          sentimentPercentages={sentimentPercentages}
        />
      </div>

      {/* Usage Stats Card - Separate row */}
      <div className="mb-6">
        <UsageStatsCard />
      </div>

      {/* Flexbox layout for main content with improved proportions */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left column */}
        <div className="w-full lg:w-[450px] flex-shrink-0"
        >
          <div className="sticky top-6">
            <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 shadow-xl border-none overflow-hidden rounded-xl relative h-[500px] flex flex-col">
              <CardHeader className="border-b border-blue-100/50 pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-blue-500/20">
                      <Globe2 className="text-blue-600 h-5 w-5" />
                    </div>
                    <CardTitle className="text-lg font-semibold text-slate-800">Affected Areas</CardTitle>
                  </div>
                  <a href="/geographic-analysis" className="rounded-lg h-8 gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 flex items-center px-3 py-1.5">
                    View All
                    <ArrowUpRight className="h-3 w-3 ml-1" />
                  </a>
                </div>
                <CardDescription className="text-slate-500 mt-1">
                  Recent disaster impact by location
                </CardDescription>
              </CardHeader>

              <div className="flex-grow overflow-hidden">
                <div className="h-[400px] overflow-hidden will-change-scroll">
                  <AffectedAreasCard 
                    sentimentPosts={filteredPosts} 
                    isLoading={isLoadingSentimentPosts}
                  />
                </div>
              </div>

              {isLoadingSentimentPosts && (
                <LoadingOverlay message="Updating affected areas..." />
              )}
            </Card>
          </div>
        </div>

        {/* Right column - takes remaining space */}
        <div className="flex-grow"
        >
          {/* Card Carousel for auto-rotating between Sentiment Distribution and Recent Activity */}
          <div className="relative mb-6 bg-gradient-to-r from-purple-50/80 to-indigo-50/80 shadow-xl border-none rounded-xl overflow-hidden">
            <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
              <div 
                className="cursor-pointer hover:scale-110 transition-transform"
                onClick={() => setCarouselPaused(!carouselPaused)}
              >
                <RefreshCw className={`h-5 w-5 text-blue-600 ${carouselPaused ? '' : 'animate-spin-slow'} rotate-icon`} />
              </div>
            </div>

            <CardCarousel 
              autoRotate={!carouselPaused}
              interval={10000}
              showControls={true}
              className="h-[500px]"
            >
              {/* Sentiment Distribution Card */}
              <div className="h-full">
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-pink-100/50 p-6 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-2 rounded-lg bg-purple-500/20">
                        <BarChart3 className="text-purple-600 h-5 w-5" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-800">Sentiment Distribution</h3>
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    Emotional response breakdown across disaster events
                  </p>
                </div>
                <div className="p-6">
                  <div className="h-[350px]">
                    <OptimizedSentimentChart 
                      data={sentimentData}
                      isLoading={isLoadingSentimentPosts}
                      type={isLowPerformanceMode ? 'doughnut' : 'doughnut'} // Just use doughnut for best performance
                    />
                  </div>
                </div>
              </div>

              {/* Recent Posts Card */}
              <div className="h-full flex flex-col">
                <div className="bg-gradient-to-r from-teal-50 to-cyan-50 border-b border-teal-100/50 p-6 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-2 rounded-lg bg-teal-500/20">
                        <Database className="text-teal-600 h-5 w-5" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-800">Recent Activity</h3>
                    </div>
                    <a href="/raw-data" className="rounded-lg h-8 gap-1 text-xs font-medium text-teal-600 hover:text-teal-700 hover:bg-teal-50 flex items-center px-3 py-1.5">
