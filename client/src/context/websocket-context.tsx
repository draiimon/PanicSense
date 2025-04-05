import { createContext, useContext, useEffect, useState, useCallback, ReactNode, useRef } from 'react';

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: any | null;
  sendMessage: (message: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  isConnected: false,
  lastMessage: null,
  sendMessage: () => {}
});

export function useWebSocket() {
  return useContext(WebSocketContext);
}

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any | null>(null);
  
  // Add timestamps to track updates and prevent flicker
  const lastUpdateTimestampRef = useRef<number>(0);
  const pendingUpdatesRef = useRef<{[key: string]: any}>({});
  const debounceTimerRef = useRef<number | null>(null);
  
  // Storage event listener for cross-tab synchronization
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      // Only respond to specific keys we care about
      if (event.key === 'uploadProgress' && event.newValue) {
        try {
          // Parse the new value
          const newProgress = JSON.parse(event.newValue);
          
          // Check if this update has the tabSyncTimestamp or officialDbUpdate flag
          // These are special flags indicating a database-sourced update
          if (newProgress && (newProgress.tabSyncTimestamp || newProgress.officialDbUpdate)) {
            console.log('📱 CROSS-TAB SYNC: Received progress update from another tab!');
            console.log(`📱 Count sync: ${newProgress.processed}/${newProgress.total}`);
            
            // Update local state with the cross-tab synchronized data
            setLastMessage({
              type: 'progress',
              progress: newProgress,
              fromTabSync: true,
              timestamp: Date.now()
            });
          }
        } catch (e) {
          // Silently ignore parsing errors in cross-tab communication
        }
      }
      
      // Also handle special sync keys
      if (event.key === 'lastTabSync' && event.newValue) {
        try {
          const syncInfo = JSON.parse(event.newValue);
          console.log(`📱 Tab sync broadcast received! Count: ${syncInfo.processed}`);
          
          // Force a resync with database to ensure consistent display
          // This is especially important for mobile where connection might be unstable
          fetch('/api/active-upload-session')
            .then(response => {
              if (response.ok) return response.json();
              throw new Error('Failed to fetch session');
            })
            .then(data => {
              if (data && data.sessionId && data.progress) {
                // Update with the database-validated progress
                setLastMessage({
                  type: 'progress',
                  progress: data.progress,
                  fromTabSync: true,
                  forcedSync: true,
                  timestamp: Date.now()
                });
                
                // Update localStorage with validated data
                // CRITICAL - add both markers to ensure all tabs recognize this
                const syncedProgress = {
                  ...data.progress,
                  timestamp: Date.now(),
                  savedAt: Date.now(),
                  officialDbUpdate: true,
                  tabSyncTimestamp: Date.now(),
                  forcedDatabaseSync: true
                };
                
                // Save to localStorage with special flags
                localStorage.setItem('uploadProgress', JSON.stringify(syncedProgress));
                console.log('📱 Tab sync: Updated localStorage with database-validated data');
              }
            })
            .catch(err => {
              console.error('Error in tab sync database validation:', err);
            });
        } catch (e) {
          // Silently ignore parsing errors
        }
      }
    };
    
    // Add the storage event listener for cross-tab communication
    window.addEventListener('storage', handleStorageChange);
    
    // Clean up on unmount
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  // Create a debounced update function to prevent flickering
  const processPendingUpdates = useCallback(() => {
    // Get the most recent update based on timestamp
    const updates = pendingUpdatesRef.current;
    const keys = Object.keys(updates);
    
    if (keys.length === 0) return;
    
    // Find the most recent update
    let mostRecentKey = keys[0];
    let mostRecentTimestamp = updates[mostRecentKey].timestamp || 0;
    
    keys.forEach(key => {
      const update = updates[key];
      if (update.timestamp && update.timestamp > mostRecentTimestamp) {
        mostRecentKey = key;
        mostRecentTimestamp = update.timestamp;
      }
    });
    
    // Apply the most recent update
    const mostRecentUpdate = updates[mostRecentKey];
    setLastMessage(mostRecentUpdate);
    
    // Clear all pending updates
    pendingUpdatesRef.current = {};
    
    // Update the last update timestamp
    lastUpdateTimestampRef.current = Date.now();
    
    // Clear the debounce timer
    debounceTimerRef.current = null;
  }, []);

  // WebSocket connection setup
  useEffect(() => {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    // Connection opened
    ws.addEventListener('open', () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
    });

    // Enhanced WebSocket message handler with improved synchronization (2025 version)
    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Check for the enhanced broadcast properties (added in server/routes.ts)
        const broadcastTimestamp = data.timestamp;
        const broadcastId = data.broadcastId;
        
        // Log broadcast receipt for debugging
        if (broadcastId) {
          console.log(`📡 Received broadcast ${broadcastId.substring(0, 6)}... from server`);
        }
        
        // If this is a progress update message, ensure localStorage is synchronized
        if (data && data.type === 'progress' && data.progress) {
          const storedSessionId = localStorage.getItem('uploadSessionId');
          const activeSession = data.sessionId || (data.progress.sessionId);
          
          // Enhanced progressive update with cooldown detection
          const isCoolingDown = data.progress.coolingDown;
          
          // Check if this update is for the active session
          const isForActiveSession = storedSessionId === activeSession || !storedSessionId;
          
          if (isForActiveSession) {
            try {
              // Parse existing progress if available
              const storedProgress = localStorage.getItem('uploadProgress');
              if (storedProgress) {
                const localProgress = JSON.parse(storedProgress);
                const newProgress = data.progress;
                
                // Enhanced count and timestamp comparison logic
                const isNewerUpdate = broadcastTimestamp > (localProgress.timestamp || 0);
                const hasMoreProgress = newProgress.processed > localProgress.processed;
                const isFromDatabase = data.fromDatabase || data.progress.officialDbUpdate;
                
                // Decide whether to update based on multiple factors
                const shouldUpdate = isNewerUpdate || hasMoreProgress || isFromDatabase;
                
                if (shouldUpdate) {
                  // Create an enhanced progress object with all necessary synchronization flags
                  const syncedProgress = {
                    ...newProgress,
                    timestamp: broadcastTimestamp || Date.now(),
                    broadcastId: broadcastId,
                    savedAt: Date.now(),
                    websocketUpdate: true,
                    coolingDown: isCoolingDown, // Explicit cooldown flag
                    tabSyncTimestamp: Date.now() // Enable cross-tab syncing
                  };
                  
                  // Log the update decision reason for debugging
                  if (hasMoreProgress) {
                    console.log('📡 WebSocket progress ahead of local - updating localStorage', 
                      `WS: ${newProgress.processed}, Local: ${localProgress.processed}`);
                  } else if (isNewerUpdate) {
                    console.log('📡 WebSocket has newer timestamp - updating localStorage');
                  } else if (isFromDatabase) {
                    console.log('📡 Update is from database (official source) - updating localStorage');
                  }
                  
                  // Update localStorage with synchronized data
                  localStorage.setItem('uploadProgress', JSON.stringify(syncedProgress));
                  
                  // Broadcast to other tabs that we've updated
                  localStorage.setItem('lastTabSync', JSON.stringify({
                    timestamp: Date.now(),
                    broadcastId: broadcastId,
                    processed: newProgress.processed,
                    source: 'websocket'
                  }));
                } else {
                  console.log('📡 Ignoring WebSocket update (not newer or higher count)');
                }
              } else {
                // No existing progress, just store the new progress
                const syncedProgress = {
                  ...data.progress,
                  timestamp: broadcastTimestamp || Date.now(),
                  savedAt: Date.now(),
                  websocketUpdate: true,
                  coolingDown: isCoolingDown,
                  tabSyncTimestamp: Date.now()
                };
                localStorage.setItem('uploadProgress', JSON.stringify(syncedProgress));
                console.log('📡 No existing localStorage progress - initializing from WebSocket');
              }
            } catch (e) {
              // Error handling localStorage, log and continue
              console.error('❌ Error handling WebSocket progress in localStorage', e);
            }
          }
        }
        
        // Add to pending updates with deduplication
        const updateKey = `${data.type}_${broadcastId || Date.now()}`;
        pendingUpdatesRef.current[updateKey] = data;
        
        // Debounce the updates to prevent UI flickering
        if (debounceTimerRef.current === null) {
          debounceTimerRef.current = window.setTimeout(() => {
            processPendingUpdates();
          }, 50); // 50ms debounce to smooth updates
        }
      } catch (error) {
        console.error('❌ Failed to parse WebSocket message:', error);
      }
    });

    // Connection closed
    ws.addEventListener('close', () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
    });

    // Store socket instance
    setSocket(ws);

    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, []);

  // Send message function
  const sendMessage = useCallback((message: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }, [socket]);

  return (
    <WebSocketContext.Provider value={{ isConnected, lastMessage, sendMessage }}>
      {children}
    </WebSocketContext.Provider>
  );
}
