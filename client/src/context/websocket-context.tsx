import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';

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

    // Listen for messages
    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // If this is a progress update message, ensure localStorage is synchronized
        if (data && data.type === 'progress' && data.sessionId && data.progress) {
          const storedSessionId = localStorage.getItem('uploadSessionId');
          
          // If this progress update is for our active upload session, update localStorage
          if (storedSessionId === data.sessionId) {
            try {
              // Parse existing progress if available
              const storedProgress = localStorage.getItem('uploadProgress');
              if (storedProgress) {
                const localProgress = JSON.parse(storedProgress);
                const newProgress = data.progress;
                
                // Only update if server has newer/greater count than local
                if (newProgress.processed > localProgress.processed) {
                  // Add timestamp and mark as WebSocket update
                  const syncedProgress = {
                    ...newProgress,
                    timestamp: Date.now(),
                    savedAt: Date.now(),
                    websocketUpdate: true
                  };
                  
                  // Update localStorage with newer progress from WebSocket
                  localStorage.setItem('uploadProgress', JSON.stringify(syncedProgress));
                  console.log('WebSocket progress ahead of local - updating localStorage', 
                    `WS: ${newProgress.processed}, Local: ${localProgress.processed}`);
                }
              }
            } catch (e) {
              // Error handling localStorage, just continue
              console.error('Error handling WebSocket progress in localStorage', e);
            }
          }
        }
        
        // Set the last message received for context consumers
        setLastMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
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
