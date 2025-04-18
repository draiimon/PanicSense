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
        console.log('📡 WebSocket message received:', data.type);
        
        // Special handling for upload complete messages
        if (data.type === 'UPLOAD_COMPLETE') {
          console.log('🌟 UPLOAD_COMPLETE WebSocket message received!');
          
          // Save completion state to localStorage for all tabs
          localStorage.setItem('uploadCompleted', 'true');
          localStorage.setItem('uploadCompletedTimestamp', Date.now().toString());
          
          // Force a broadcast to all tabs via the BroadcastChannel API
          try {
            const bc = new BroadcastChannel('upload_status');
            bc.postMessage({
              type: 'upload_complete',
              progress: data.progress,
              timestamp: Date.now()
            });
            
            // Also use the dedicated completion channel
            const cc = new BroadcastChannel('upload_completion');
            cc.postMessage({
              type: 'analysis_complete',
              timestamp: Date.now()
            });
            
            // Close the channels after sending
            setTimeout(() => {
              try { bc.close(); } catch (e) { /* ignore */ }
              try { cc.close(); } catch (e) { /* ignore */ }
            }, 1000);
          } catch (e) {
            console.error('Error broadcasting completion via BroadcastChannel:', e);
          }
          
          // Clear any upload state after a delay
          setTimeout(() => {
            localStorage.removeItem('isUploading');
            localStorage.removeItem('uploadProgress');
            localStorage.removeItem('uploadSessionId');
          }, 3000);
        }
        
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
