import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'wouter';
import { motion, useAnimation, AnimatePresence } from 'framer-motion';
import { ChevronRight, X, FileText, BarChart3, AlertTriangle, MapPin, Clock, Database, ArrowRight, Info, ExternalLink, Shield, Users, BellRing, Star, Award, Heart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

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

// Animated globe component for background
const AnimatedGlobe = () => {
  return (
    <div className="absolute right-[-250px] top-[-150px] opacity-30 pointer-events-none">
      <div className="w-[500px] h-[500px] rounded-full border-2 border-blue-400/20 border-dashed relative animate-spin-slow">
        <div className="w-[450px] h-[450px] rounded-full border-2 border-indigo-500/20 border-dashed absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-spin-slower"></div>
        <div className="w-[400px] h-[400px] rounded-full border-2 border-cyan-400/20 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-spin-reverse"></div>
        <div className="w-4 h-4 rounded-full bg-blue-500 absolute top-0 left-1/2 transform -translate-x-1/2 shadow-glow-blue"></div>
        <div className="w-4 h-4 rounded-full bg-indigo-500 absolute bottom-0 left-1/2 transform -translate-x-1/2 shadow-glow-indigo"></div>
        <div className="w-4 h-4 rounded-full bg-cyan-500 absolute left-0 top-1/2 transform -translate-y-1/2 shadow-glow-cyan"></div>
        <div className="w-4 h-4 rounded-full bg-purple-500 absolute right-0 top-1/2 transform -translate-y-1/2 shadow-glow-purple"></div>
      </div>
    </div>
  );
};

// Animated text effect
const AnimatedText = ({ text, className = "", delay = 0 }) => {
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
      {text.split("").map((char, i) => (
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
const FloatingElement = ({ delay = 0, duration = 3, children, className = "" }) => {
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
      image: "/images/PANICSENSE PH.png"
    },
    {
      title: "Analyze Sentiment",
      description: "The system automatically analyzes emotions and classifies each message using advanced AI models.",
      icon: <BarChart3 size={24} />,
      image: "/images/PANICSENSE PH.png"
    },
    {
      title: "Geographic Analysis",
      description: "View disaster locations plotted on interactive maps to identify affected areas.",
      icon: <MapPin size={24} />,
      image: "/images/PANICSENSE PH.png"
    },
    {
      title: "Real-time Monitoring",
      description: "Monitor new disaster reports in real-time for faster emergency response and coordination.",
      icon: <Clock size={24} />,
      image: "/images/PANICSENSE PH.png"
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
        className="bg-card text-card-foreground rounded-xl shadow-2xl max-w-4xl w-full relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 w-full h-1">
          <div 
            className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 h-full"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
        
        <motion.button 
          onClick={onClose}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          whileHover={{ scale: 1.1, rotate: 90 }}
          whileTap={{ scale: 0.9 }}
          transition={{ type: "spring", stiffness: 500, damping: 25 }}
          className="absolute top-4 right-4 p-1 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10 text-white"
        >
          <X size={24} />
        </motion.button>
        
        <div className="flex flex-col md:flex-row h-[600px]">
          <div className="w-full md:w-1/2 bg-gradient-to-br from-violet-600 via-indigo-700 to-blue-800 p-8 text-white relative overflow-hidden">
            {/* Animated background elements */}
            <div className="absolute inset-0 opacity-10">
              <TwinklingStars />
            </div>
            
            <div className="absolute top-4 left-4 flex space-x-2">
              {steps.map((_, index) => (
                <motion.div 
                  key={index}
                  initial={{ scale: 0.8, opacity: 0.5 }}
                  animate={{ 
                    scale: currentStep === index ? 1 : 0.8,
                    opacity: currentStep === index ? 1 : 0.5
                  }}
                  whileHover={{ scale: 0.9 }}
                  onClick={() => {
                    setDirection(index > currentStep ? 1 : -1);
                    setCurrentStep(index);
                  }}
                  className={`w-3 h-3 rounded-full cursor-pointer transition-all ${
                    currentStep === index ? 'bg-white' : 'bg-white/40'
                  }`}
                />
              ))}
            </div>
            
            <div className="h-full flex flex-col justify-center relative z-10 pt-8">
              <AnimatePresence custom={direction} initial={false}>
                <motion.div
                  key={currentStep}
                  custom={direction}
                  variants={slideVariants}
                  initial="enter"
                  animate="center"
                  exit="exit"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  className="absolute inset-0 flex flex-col justify-center p-8 pt-16"
                >
                  <FloatingElement delay={0.1} className="mb-6">
                    <motion.div 
                      className="p-4 bg-white/10 rounded-full w-fit mb-4"
                      whileHover={{ rotate: 5, scale: 1.05 }}
                    >
                      {steps[currentStep].icon}
                    </motion.div>
                  </FloatingElement>
                  
                  <h3 className="text-3xl font-bold mb-4">
                    <AnimatedText text={steps[currentStep].title} delay={0.2} />
                  </h3>
                  
                  <p className="text-white/80 text-lg mb-8">
                    <AnimatedText text={steps[currentStep].description} delay={0.3} />
                  </p>
                </motion.div>
              </AnimatePresence>
              
              <div className="flex space-x-3 mt-auto relative z-20">
                {currentStep > 0 && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    <Button 
                      variant="outline" 
                      className="border-white/20 text-white hover:bg-white/10 hover:scale-105 transform transition-all"
                      onClick={prevStep}
                    >
                      Previous Step
                    </Button>
                  </motion.div>
                )}
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <Button 
                    onClick={nextStep}
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white hover:scale-105 transform transition-all shadow-lg"
                  >
                    {currentStep === steps.length - 1 ? 'Get Started' : 'Next Step'}
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </motion.div>
              </div>
            </div>
          </div>
          
          <div className="w-full md:w-1/2 flex items-center justify-center p-8 bg-gradient-to-b from-gray-100 to-white dark:from-gray-900 dark:to-gray-800 relative overflow-hidden">
            <div className="absolute inset-0">
              <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
            </div>
            
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, scale: 0.8, rotateY: 90 }}
                animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                exit={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                transition={{ type: "spring", damping: 20 }}
                className="relative z-10 bg-white dark:bg-gray-800 p-2 rounded-xl shadow-2xl"
              >
                <img 
                  src="/images/PANICSENSE PH.png" 
                  alt={steps[currentStep].title}
                  className="max-w-full max-h-[320px] rounded-lg"
                />
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  className="absolute -bottom-4 -right-4"
                >
                  <Badge className="bg-gradient-to-r from-pink-500 to-purple-500 text-white px-3 py-1">
                    Step {currentStep + 1}/{steps.length}
                  </Badge>
                </motion.div>
              </motion.div>
            </AnimatePresence>
          </div>
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
    <div className="min-h-screen overflow-hidden">
      {/* Header with Glass Effect */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/10 dark:bg-gray-900/20 backdrop-blur-md py-4 px-6 border-b border-white/10">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3"
          >
            <div className="relative">
              <motion.img 
                src="/images/PANICSENSE PH.png" 
                alt="PanicSense PH Logo" 
                className="h-12 rounded-full"
                whileHover={{ rotate: 10, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 500 }}
              />
              <motion.div 
                className="absolute -right-1 -top-1 w-4 h-4 bg-green-400 rounded-full"
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [1, 0.8, 1] 
                }}
                transition={{ 
                  repeat: Infinity,
                  duration: 2 
                }}
              />
            </div>
            
            <div>
              <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-500 to-pink-500">
                PanicSense PH
              </h1>
              <div className="flex items-center">
                <span className="text-xs px-1.5 py-0.5 bg-green-500/20 text-green-500 rounded-full">LIVE</span>
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">Monitoring active</span>
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex items-center space-x-4"
          >
            <Link href="/dashboard">
              <Button 
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all hover:scale-105"
              >
                Sign in to Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </header>
      
      {/* Hero Section with Animated Background */}
      <section className="relative min-h-screen flex items-center pt-20">
        <div className="absolute inset-0 bg-gradient-to-b from-gray-100 to-white dark:from-gray-900 dark:to-slate-900 overflow-hidden">
          <TwinklingStars />
          
          {/* Background geometric elements */}
          <div className="absolute bottom-0 left-0 w-full h-1/3 bg-grid-dark-pattern opacity-[0.03] dark:opacity-[0.07]"></div>
          
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
          
          <AnimatedGlobe />
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
                <Button 
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all hover:scale-105 px-8 py-6 text-lg"
                >
                  <Link href="/dashboard">
                    Get Started Now
                  </Link>
                </Button>
                
                <Button 
                  variant="outline" 
                  size="lg"
                  className="border-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all hover:scale-105 px-8 py-6 text-lg"
                  onClick={() => setShowTutorial(true)}
                >
                  Watch Tutorial
                </Button>
              </motion.div>
              
              <motion.div 
                className="mt-10 flex items-center space-x-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                <div className="flex -space-x-2">
                  {['/images/drei.jpg', '/images/julia.jpg', '/images/van.jpg'].map((src, i) => (
                    <motion.img 
                      key={i}
                      src={src}
                      alt={`User ${i+1}`}
                      className="w-8 h-8 rounded-full border-2 border-white dark:border-gray-900"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.8 + (i * 0.1) }}
                    />
                  ))}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  <span className="text-blue-600 dark:text-blue-400 font-semibold">180+</span> emergency responders actively using PanicSense
                </div>
              </motion.div>
            </div>
            
            <motion.div
              initial={{ opacity: 0, scale: 0.9, rotateY: 20 }}
              animate={{ opacity: 1, scale: 1, rotateY: 0 }}
              transition={{ duration: 0.8, delay: 0.3, type: "spring" }}
              className="rounded-2xl shadow-2xl overflow-hidden relative border border-gray-200 dark:border-gray-800"
            >
              <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/20 via-transparent to-purple-500/20"></div>
              <img 
                src="/images/PANICSENSE PH.png" 
                alt="PanicSense PH Dashboard Preview" 
                className="w-full h-auto relative z-10"
              />
              
              {/* Animated elements on top of the image */}
              <motion.div 
                className="absolute bottom-4 left-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 shadow-lg border border-gray-200 dark:border-gray-800 z-20 max-w-[200px]"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
              >
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                  <h4 className="font-semibold text-sm">Live Alert</h4>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">Earthquake detected in Mindanao region. Severity: 5.2</p>
              </motion.div>
              
              <FloatingElement 
                delay={0.5} 
                className="absolute top-4 right-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 shadow-lg border border-gray-200 dark:border-gray-800 z-20"
              >
                <div className="flex items-center">
                  <BellRing className="h-4 w-4 text-blue-500 mr-2" />
                  <span className="text-xs font-medium">Real-time analytics active</span>
                </div>
              </FloatingElement>
              
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
                title: "AI Sentiment Analysis",
                description: "Automatically detect emotions and sentiment in disaster-related text using advanced AI models.",
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
              { value: "30+", label: "Government Agencies Connected", icon: <Users /> },
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
      
      {/* Call to Action */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-white to-gray-100 dark:from-gray-900 dark:to-black">
          <div className="absolute inset-0 bg-grid-pattern opacity-[0.03] dark:opacity-[0.07]"></div>
        </div>
        
        <div className="max-w-5xl mx-auto px-6 relative z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-3xl p-12 md:p-16 text-center text-white shadow-2xl relative overflow-hidden"
          >
            <div className="absolute inset-0 overflow-hidden opacity-30">
              <motion.div 
                className="absolute top-0 -left-24 w-96 h-96 bg-gradient-to-r from-blue-500 to-transparent rounded-full blur-3xl"
                animate={{
                  y: [0, 50, 0],
                  x: [0, 30, 0],
                }}
                transition={{
                  repeat: Infinity,
                  duration: 10,
                  ease: "easeInOut"
                }}
              />
              <motion.div 
                className="absolute bottom-0 -right-24 w-96 h-96 bg-gradient-to-l from-purple-500 to-transparent rounded-full blur-3xl"
                animate={{
                  y: [0, -50, 0],
                  x: [0, -30, 0],
                }}
                transition={{
                  repeat: Infinity,
                  duration: 10,
                  delay: 0.5,
                  ease: "easeInOut"
                }}
              />
            </div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-6 relative">
                Ready to Experience <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-200 to-indigo-100">
                  PanicSense PH?
                </span>
              </h2>
              
              <p className="text-xl text-white/90 mb-10 max-w-2xl mx-auto">
                Explore our platform to see how advanced analytics can help improve disaster response and save lives.
              </p>
              
              <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
                <Link href="/dashboard">
                  <Button size="lg" className="bg-white text-indigo-700 hover:bg-gray-100 text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all hover:scale-105">
                    Access Dashboard
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </Link>
                
                <Button 
                  variant="outline" 
                  size="lg" 
                  className="border-white text-white hover:bg-white/10 text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all hover:scale-105"
                  onClick={() => setShowTutorial(true)}
                >
                  Watch Tutorial
                </Button>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16 relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-0 opacity-5">
            <TwinklingStars />
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
            <div className="col-span-1 md:col-span-2">
              <motion.div 
                className="flex items-center space-x-3 mb-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
              >
                <img src="/images/PANICSENSE PH.png" alt="PanicSense PH Logo" className="h-12 rounded-lg" />
                <div>
                  <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-300">
                    PanicSense PH
                  </h3>
                  <p className="text-sm text-gray-400">Disaster Intelligence Platform</p>
                </div>
              </motion.div>
              
              <motion.p 
                className="text-gray-400 mb-6 max-w-md"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                Advanced disaster monitoring and community resilience platform for the Philippines, leveraging cutting-edge AI to transform emergency response.
              </motion.p>
              
              <motion.div
                className="flex items-center space-x-4"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <ExternalLink className="h-5 w-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Heart className="h-5 w-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Info className="h-5 w-5" />
                </a>
              </motion.div>
            </div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <h4 className="font-semibold text-lg mb-4 text-white">Navigation</h4>
              <ul className="space-y-3">
                <li><Link href="/dashboard" className="text-gray-400 hover:text-white transition-colors">Dashboard</Link></li>
                <li><Link href="/geographic-analysis" className="text-gray-400 hover:text-white transition-colors">Geographic Analysis</Link></li>
                <li><Link href="/real-time" className="text-gray-400 hover:text-white transition-colors">Real-time Monitor</Link></li>
                <li><Link href="/about" className="text-gray-400 hover:text-white transition-colors">About</Link></li>
              </ul>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <h4 className="font-semibold text-lg mb-4 text-white">Contact</h4>
              <p className="text-gray-400 mb-3">Email: info@panicsenseph.com</p>
              <p className="text-gray-400">Phone: +63 (2) 8123 4567</p>
              
              <Button 
                className="mt-4 bg-white/10 hover:bg-white/20 text-white" 
                size="sm"
              >
                <Link href="/dashboard">
                  Contact Support
                </Link>
              </Button>
            </motion.div>
          </div>
          
          <motion.div 
            className="border-t border-gray-800 mt-16 pt-8 flex flex-col md:flex-row justify-between items-center"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <p className="text-gray-500 text-sm">© 2025 PanicSense PH. All rights reserved.</p>
            <div className="mt-4 md:mt-0">
              <p className="text-gray-500 text-sm">
                Built with <span className="text-red-500">♥</span> by Team PanicSense PH
              </p>
            </div>
          </motion.div>
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
      <style jsx global>{`
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