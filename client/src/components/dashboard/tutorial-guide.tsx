import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, X, FileText, BarChart3, MapPin, Clock, Database, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';

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

const TutorialGuide: React.FC<TutorialGuideProps> = ({ onClose, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);
  
  const steps: TutorialStep[] = [
    {
      title: "Upload Data",
      description: "Start by uploading a CSV file containing disaster text messages for analysis.",
      element: "upload-button",
      position: "bottom",
      icon: <FileText className="h-5 w-5" />
    },
    {
      title: "Sentiment Analysis",
      description: "View emotion classifications and sentiment scores for each message processed.",
      element: "sentiment-analysis",
      position: "right",
      icon: <BarChart3 className="h-5 w-5" />
    },
    {
      title: "Geographic Mapping",
      description: "See disaster locations plotted on maps to identify affected areas.",
      element: "geo-mapping",
      position: "left",
      icon: <MapPin className="h-5 w-5" />
    },
    {
      title: "Real-time Monitoring",
      description: "Track incoming disaster reports as they arrive for immediate response.",
      element: "realtime-monitor",
      position: "top",
      icon: <Clock className="h-5 w-5" />
    },
    {
      title: "Data Management",
      description: "Access and manage your uploaded files and processed data.",
      element: "data-management",
      position: "bottom",
      icon: <Database className="h-5 w-5" />
    }
  ];
  
  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };
  
  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  useEffect(() => {
    // Scroll to and highlight the current element
    const targetElement = document.getElementById(steps[currentStep].element);
    if (targetElement) {
      targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      // Add a highlight effect
      const originalStyle = targetElement.style.cssText;
      targetElement.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.5)';
      targetElement.style.position = 'relative';
      targetElement.style.zIndex = '50';
      targetElement.style.transition = 'box-shadow 0.3s ease';
      
      return () => {
        // Remove highlight effect when changing steps
        targetElement.style.cssText = originalStyle;
      };
    }
  }, [currentStep, steps]);
  
  if (!isVisible) return null;
  
  // Calculate position based on the target element
  const getPosition = () => {
    const targetElement = document.getElementById(steps[currentStep].element);
    if (!targetElement) return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
    
    const rect = targetElement.getBoundingClientRect();
    const position = steps[currentStep].position;
    
    switch (position) {
      case 'top':
        return {
          top: `${rect.top - 200}px`,
          left: `${rect.left + rect.width / 2}px`,
          transform: 'translateX(-50%)'
        };
      case 'right':
        return {
          top: `${rect.top + rect.height / 2}px`,
          left: `${rect.right + 20}px`,
          transform: 'translateY(-50%)'
        };
      case 'bottom':
        return {
          top: `${rect.bottom + 20}px`,
          left: `${rect.left + rect.width / 2}px`,
          transform: 'translateX(-50%)'
        };
      case 'left':
        return {
          top: `${rect.top + rect.height / 2}px`,
          left: `${rect.left - 300}px`,
          transform: 'translateY(-50%)'
        };
      default:
        return {
          top: `${rect.bottom + 20}px`,
          left: `${rect.left + rect.width / 2}px`,
          transform: 'translateX(-50%)'
        };
    }
  };
  
  const position = getPosition();
  
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.9 }}
          className="fixed z-50 w-[300px] bg-card shadow-xl rounded-lg overflow-hidden"
          style={{
            top: position.top,
            left: position.left,
            transform: position.transform
          }}
        >
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500"></div>
          
          <div className="bg-gradient-to-br from-blue-600 to-indigo-700 p-4 text-white">
            <button 
              onClick={onClose}
              className="absolute top-3 right-3 text-white/80 hover:text-white p-1 rounded-full hover:bg-white/10"
            >
              <X size={18} />
            </button>
            
            <div className="flex items-center space-x-3 mb-2">
              <div className="p-2 bg-white/20 rounded-full">
                {steps[currentStep].icon}
              </div>
              <h3 className="font-bold">{steps[currentStep].title}</h3>
            </div>
            
            <div className="flex space-x-1 mb-2">
              {steps.map((_, index) => (
                <div 
                  key={index}
                  className={`h-1 rounded-full ${
                    index === currentStep ? 'bg-white w-4' : 'bg-white/30 w-2'
                  } transition-all`}
                />
              ))}
            </div>
          </div>
          
          <div className="p-4">
            <p className="text-sm text-foreground mb-4">
              {steps[currentStep].description}
            </p>
            
            <div className="flex justify-between items-center">
              {currentStep > 0 ? (
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={prevStep}
                >
                  Previous
                </Button>
              ) : (
                <div /> // Empty div to maintain layout
              )}
              
              <Button 
                size="sm" 
                onClick={nextStep}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white"
              >
                {currentStep === steps.length - 1 ? (
                  <>
                    Finish
                    <Check className="ml-1 h-4 w-4" />
                  </>
                ) : (
                  <>
                    Next
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default TutorialGuide;