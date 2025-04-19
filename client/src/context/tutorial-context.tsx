import React, { createContext, useContext, useState, useEffect } from 'react';

interface TutorialContextType {
  showTutorial: boolean;
  completedTutorial: boolean;
  startTutorial: () => void;
  closeTutorial: () => void;
  completeTutorial: () => void;
}

const TutorialContext = createContext<TutorialContextType | undefined>(undefined);

const STORAGE_KEY = 'panicsense-tutorial-completed';

export const TutorialProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [showTutorial, setShowTutorial] = useState(false);
  const [completedTutorial, setCompletedTutorial] = useState(false);
  const [initialCheck, setInitialCheck] = useState(false);
  
  // On first mount, check local storage to see if tutorial was completed
  useEffect(() => {
    const completed = localStorage.getItem(STORAGE_KEY) === 'true';
    setCompletedTutorial(completed);
    
    // Show tutorial automatically on first visit if not completed
    if (!completed && !initialCheck) {
      // Delay showing the tutorial to ensure page loads first
      const timer = setTimeout(() => {
        setShowTutorial(true);
      }, 2000);
      
      setInitialCheck(true);
      return () => clearTimeout(timer);
    }
  }, [initialCheck]);
  
  const startTutorial = () => {
    setShowTutorial(true);
  };
  
  const closeTutorial = () => {
    setShowTutorial(false);
  };
  
  const completeTutorial = () => {
    setShowTutorial(false);
    setCompletedTutorial(true);
    localStorage.setItem(STORAGE_KEY, 'true');
  };
  
  return (
    <TutorialContext.Provider
      value={{
        showTutorial,
        completedTutorial,
        startTutorial,
        closeTutorial,
        completeTutorial
      }}
    >
      {children}
    </TutorialContext.Provider>
  );
};

export const useTutorial = () => {
  const context = useContext(TutorialContext);
  if (context === undefined) {
    throw new Error('useTutorial must be used within a TutorialProvider');
  }
  return context;
};