import React, { useState, useEffect } from 'react';
import { useTutorial } from '@/context/tutorial-context';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, X, HelpCircle, Upload, BarChart3, Database, LineChart, Clock, Globe } from 'lucide-react';

interface TutorialStep {
  title: string;
  description: string;
  element: string; // ID of the element to highlight
  position: 'top' | 'right' | 'bottom' | 'left';
  icon: React.ReactNode;
}

interface TutorialGuideProps {
  onClose: () => void;
  onComplete: () => void;
}

export const TutorialGuide: React.FC<TutorialGuideProps> = ({ onClose, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  
  // Define all tutorial steps
  const tutorialSteps: TutorialStep[] = [
    {
      title: 'Welcome to PanicSensePH',
      description: 'This quick guide will help you navigate through the dashboard. Click "Next" to begin the tour.',
      element: 'dashboard-main', // Main dashboard container
      position: 'top',
      icon: <HelpCircle className="h-5 w-5 text-blue-500" />
    },
    {
      title: 'Upload Data',
      description: 'Upload CSV files with social media posts to analyze sentiment patterns during disaster events.',
      element: 'upload-data-section', // File uploader component
      position: 'bottom',
      icon: <Upload className="h-5 w-5 text-purple-500" />
    },
    {
      title: 'Sentiment Dashboard',
      description: 'View aggregated sentiment analysis across all processed data. Monitor emotion trends during disaster events.',
      element: 'sentiment-stats', // Sentiment statistics card
      position: 'right',
      icon: <BarChart3 className="h-5 w-5 text-green-500" />
    },
    {
      title: 'Raw Data Access',
      description: 'Access and search through all processed posts with their sentiment classifications.',
      element: 'recent-posts', // Recent posts table
      position: 'top',
      icon: <Database className="h-5 w-5 text-amber-500" />
    },
    {
      title: 'Geographic Analysis',
      description: 'Visualize sentiment distribution across geographic locations to identify affected areas.',
      element: 'affected-areas', // Map or geographic visualization
      position: 'left',
      icon: <Globe className="h-5 w-5 text-indigo-500" />
    },
    {
      title: 'Timeline Analysis',
      description: 'Analyze how sentiment changes over time during disaster events.',
      element: 'timeline-link', // Timeline section or link
      position: 'bottom',
      icon: <Clock className="h-5 w-5 text-rose-500" />
    },
    {
      title: 'Comparative Analysis',
      description: 'Compare different disaster events or time periods to identify patterns.',
      element: 'comparison-link', // Comparison section or link
      position: 'bottom',
      icon: <LineChart className="h-5 w-5 text-cyan-500" />
    }
  ];
  
  // Find the target element based on step
  useEffect(() => {
    if (currentStep >= 0 && currentStep < tutorialSteps.length) {
      const step = tutorialSteps[currentStep];
      const element = document.getElementById(step.element);
      
      if (element) {
        setTargetElement(element);
        
        // Calculate position for the tooltip
        const rect = element.getBoundingClientRect();
        const position = calculateTooltipPosition(rect, step.position);
        setTooltipPosition(position);
        
        // Scroll the element into view with a smooth animation
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Add highlight class to the target element
        element.classList.add('tutorial-highlight');
        
        return () => {
          element.classList.remove('tutorial-highlight');
        };
      } else {
        // If element doesn't exist, use a fallback position
        setTargetElement(null);
        setTooltipPosition({
          top: window.innerHeight / 2,
          left: window.innerWidth / 2
        });
      }
    }
  }, [currentStep, tutorialSteps]);
  
  // Calculate the tooltip position based on the element's rect and desired position
  const calculateTooltipPosition = (rect: DOMRect, position: 'top' | 'right' | 'bottom' | 'left') => {
    const tooltipWidth = 320; // Approximate width of the tooltip
    const tooltipHeight = 200; // Approximate height of the tooltip
    const spacing = 16; // Spacing between element and tooltip
    
    switch (position) {
      case 'top':
        return {
          top: rect.top - tooltipHeight - spacing,
          left: rect.left + (rect.width / 2) - (tooltipWidth / 2)
        };
      case 'right':
        return {
          top: rect.top + (rect.height / 2) - (tooltipHeight / 2),
          left: rect.right + spacing
        };
      case 'bottom':
        return {
          top: rect.bottom + spacing,
          left: rect.left + (rect.width / 2) - (tooltipWidth / 2)
        };
      case 'left':
        return {
          top: rect.top + (rect.height / 2) - (tooltipHeight / 2),
          left: rect.left - tooltipWidth - spacing
        };
      default:
        return {
          top: rect.bottom + spacing,
          left: rect.left + (rect.width / 2) - (tooltipWidth / 2)
        };
    }
  };
  
  // Navigation functions
  const goToPrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  const goToNextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Last step completed
      onComplete();
    }
  };
  
  // Ensure tutorial stays within viewport
  useEffect(() => {
    const adjustTooltipPosition = () => {
      if (!tooltipPosition) return;
      
      let { top, left } = tooltipPosition;
      const tooltipWidth = 320;
      const tooltipHeight = 200;
      
      // Check right boundary
      if (left + tooltipWidth > window.innerWidth) {
        left = window.innerWidth - tooltipWidth - 16;
      }
      
      // Check left boundary
      if (left < 16) {
        left = 16;
      }
      
      // Check bottom boundary
      if (top + tooltipHeight > window.innerHeight) {
        top = window.innerHeight - tooltipHeight - 16;
      }
      
      // Check top boundary
      if (top < 16) {
        top = 16;
      }
      
      if (top !== tooltipPosition.top || left !== tooltipPosition.left) {
        setTooltipPosition({ top, left });
      }
    };
    
    adjustTooltipPosition();
    window.addEventListener('resize', adjustTooltipPosition);
    
    return () => {
      window.removeEventListener('resize', adjustTooltipPosition);
    };
  }, [tooltipPosition]);
  
  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        goToNextStep();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        goToPrevStep();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [currentStep, onClose]);
  
  if (currentStep < 0 || currentStep >= tutorialSteps.length) {
    return null;
  }
  
  const currentTutorialStep = tutorialSteps[currentStep];
  
  return (
    <AnimatePresence>
      {/* Tutorial overlay to dim the background */}
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[1000]" />
      
      {/* Tutorial tooltip */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.2 }}
        className="fixed z-[1001] w-80 bg-white rounded-lg shadow-xl"
        style={{
          top: tooltipPosition.top,
          left: tooltipPosition.left,
        }}
      >
        {/* Tooltip header */}
        <div className="flex items-center justify-between bg-blue-600 text-white p-3 rounded-t-lg">
          <div className="flex items-center gap-2">
            {currentTutorialStep.icon}
            <h3 className="font-semibold">{currentTutorialStep.title}</h3>
          </div>
          <button 
            onClick={onClose}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        {/* Tooltip body */}
        <div className="p-4">
          <p className="text-gray-700 mb-4">{currentTutorialStep.description}</p>
          
          {/* Progress indicators */}
          <div className="flex justify-center gap-1 mb-4">
            {tutorialSteps.map((_, index) => (
              <div 
                key={index}
                className={`h-1.5 rounded-full ${
                  index === currentStep ? 'w-4 bg-blue-600' : 'w-2 bg-gray-300'
                }`}
              />
            ))}
          </div>
          
          {/* Navigation buttons */}
          <div className="flex justify-between">
            <Button
              variant="outline"
              onClick={goToPrevStep}
              disabled={currentStep === 0}
              className="flex items-center gap-1"
            >
              <ChevronLeft className="h-4 w-4" />
              Back
            </Button>
            
            <Button
              onClick={goToNextStep}
              className="flex items-center gap-1"
            >
              {currentStep < tutorialSteps.length - 1 ? (
                <>
                  Next
                  <ChevronRight className="h-4 w-4" />
                </>
              ) : (
                'Finish'
              )}
            </Button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};