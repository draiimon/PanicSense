import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { resetUploadSessions } from "@/lib/api";

export function EmergencyResetButton() {
  const [isHidden, setIsHidden] = useState(true);
  const [isResetting, setIsResetting] = useState(false);
  const { setIsUploading, setUploadProgress } = useDisasterContext();
  
  // Check if shift key is pressed 5 times in 3 seconds to show the button
  useEffect(() => {
    let shiftCount = 0;
    let lastShiftTime = 0;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        const now = Date.now();
        if (now - lastShiftTime < 3000) {
          shiftCount++;
          if (shiftCount >= 5) {
            setIsHidden(false);
            shiftCount = 0;
          }
        } else {
          shiftCount = 1;
        }
        lastShiftTime = now;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  const resetLocalStorage = async () => {
    if (isResetting) return;
    
    setIsResetting(true);
    try {
      // Clear context state immediately
      setIsUploading(false);
      setUploadProgress({
        processed: 0,
        total: 0,
        stage: '',
      });
      
      // Clear all upload-related localStorage
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastDatabaseCheck');
      localStorage.removeItem('serverRestartProtection');
      localStorage.removeItem('serverRestartTimestamp');
      localStorage.removeItem('cooldownActive');
      localStorage.removeItem('cooldownStartedAt');
      localStorage.removeItem('lastTabSync');
      
      // Clean up any existing EventSource connections
      if (window._activeEventSources) {
        Object.values(window._activeEventSources).forEach(source => {
          try {
            source.close();
          } catch (e) {
            // Ignore errors on close
          }
        });
        // Reset the collection
        window._activeEventSources = {};
      }
      
      // Use the API function to reset all upload sessions
      const result = await resetUploadSessions();
      
      if (result.success) {
        alert('🧹 Upload modal cleared! All data has been reset.');
        // Hide the button after successful reset
        setIsHidden(true);
      } else {
        alert(`❌ Error: ${result.message}`);
      }
    } catch (error) {
      console.error('Error during emergency reset:', error);
      alert('❌ Error during reset. Check console for details.');
    } finally {
      setIsResetting(false);
    }
  };
  
  if (isHidden) return null;
  
  return (
    <div 
      style={{
        position: 'fixed',
        bottom: '10px',
        right: '10px',
        zIndex: 10000,
      }}
    >
      <Button 
        size="sm" 
        variant="destructive"
        onClick={resetLocalStorage}
        disabled={isResetting}
      >
        {isResetting ? (
          <>
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Fixing...
          </>
        ) : (
          <>
            <AlertCircle className="mr-2 h-4 w-4" />
            Fix Stuck Upload Modal
          </>
        )}
      </Button>
    </div>
  );
}