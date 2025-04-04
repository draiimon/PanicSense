/**
 * Broadcast Service for cross-tab/cross-window communication
 * 
 * This service uses the BroadcastChannel API to synchronize upload states
 * across multiple browser tabs/windows, ensuring a consistent experience
 * when the user has multiple instances of the application open.
 */

import { UploadProgress } from './api';

// Channel names for different types of messages
const UPLOAD_STATE_CHANNEL = 'panicsense-upload-state';
const UPLOAD_PROGRESS_CHANNEL = 'panicsense-upload-progress';
const UPLOAD_COMMAND_CHANNEL = 'panicsense-upload-command';

// Type definitions for our messages
export type UploadStateMessage = {
  isUploading: boolean;
  sessionId?: string | null;
  timestamp: number; // For message ordering
};

export type UploadCommandMessage = {
  command: 'cancel' | 'forceClose' | 'cleanup';
  sessionId?: string | null;
  timestamp: number;
};

// Class to handle broadcast channel communications
class BroadcastService {
  private stateChannel: BroadcastChannel | null = null;
  private progressChannel: BroadcastChannel | null = null;
  private commandChannel: BroadcastChannel | null = null;
  private initialized = false;

  // Callbacks that will be called when messages are received
  private stateCallbacks: ((state: UploadStateMessage) => void)[] = [];
  private progressCallbacks: ((progress: UploadProgress) => void)[] = [];
  private commandCallbacks: ((command: UploadCommandMessage) => void)[] = [];

  constructor() {
    this.init();
  }

  // Initialize the channels if the browser supports BroadcastChannel
  private init() {
    if (typeof BroadcastChannel !== 'undefined') {
      try {
        // Create channels
        this.stateChannel = new BroadcastChannel(UPLOAD_STATE_CHANNEL);
        this.progressChannel = new BroadcastChannel(UPLOAD_PROGRESS_CHANNEL);
        this.commandChannel = new BroadcastChannel(UPLOAD_COMMAND_CHANNEL);

        // Set up listeners
        this.stateChannel.onmessage = (event) => {
          const message: UploadStateMessage = event.data;
          this.stateCallbacks.forEach((callback) => callback(message));
        };

        this.progressChannel.onmessage = (event) => {
          const message: UploadProgress = event.data;
          this.progressCallbacks.forEach((callback) => callback(message));
        };

        this.commandChannel.onmessage = (event) => {
          const message: UploadCommandMessage = event.data;
          this.commandCallbacks.forEach((callback) => callback(message));
        };

        this.initialized = true;
      } catch (error) {
        console.error('Failed to initialize broadcast channels:', error);
      }
    } else {
      // BroadcastChannel API not supported, cross-tab sync will be disabled
    }
  }

  // Add a callback for upload state changes
  public onUploadStateChange(callback: (state: UploadStateMessage) => void) {
    this.stateCallbacks.push(callback);
    return () => {
      this.stateCallbacks = this.stateCallbacks.filter((cb) => cb !== callback);
    };
  }

  // Add a callback for upload progress updates
  public onUploadProgressUpdate(callback: (progress: UploadProgress) => void) {
    this.progressCallbacks.push(callback);
    return () => {
      this.progressCallbacks = this.progressCallbacks.filter((cb) => cb !== callback);
    };
  }

  // Add a callback for upload commands
  public onUploadCommand(callback: (command: UploadCommandMessage) => void) {
    this.commandCallbacks.push(callback);
    return () => {
      this.commandCallbacks = this.commandCallbacks.filter((cb) => cb !== callback);
    };
  }

  // Broadcast an upload state change to all tabs
  public broadcastUploadState(state: Omit<UploadStateMessage, 'timestamp'>) {
    if (!this.initialized || !this.stateChannel) return;

    const message: UploadStateMessage = {
      ...state,
      timestamp: Date.now(),
    };

    try {
      this.stateChannel.postMessage(message);
    } catch (error) {
      console.error('Error broadcasting upload state:', error);
    }
  }

  // Broadcast upload progress to all tabs
  public broadcastUploadProgress(progress: UploadProgress) {
    if (!this.initialized || !this.progressChannel) return;

    // Add timestamp if it doesn't exist (by extending the interface dynamically)
    const message = {
      ...progress,
      timestamp: (progress as any).timestamp || Date.now(),
    };

    try {
      this.progressChannel.postMessage(message);
    } catch (error) {
      console.error('Error broadcasting upload progress:', error);
    }
  }

  // Broadcast an upload command to all tabs
  public broadcastUploadCommand(command: Omit<UploadCommandMessage, 'timestamp'>) {
    if (!this.initialized || !this.commandChannel) return;

    const message: UploadCommandMessage = {
      ...command,
      timestamp: Date.now(),
    };

    try {
      this.commandChannel.postMessage(message);
    } catch (error) {
      console.error('Error broadcasting upload command:', error);
    }
  }

  // Close all channels
  public cleanup() {
    if (this.stateChannel) {
      this.stateChannel.close();
      this.stateChannel = null;
    }
    if (this.progressChannel) {
      this.progressChannel.close();
      this.progressChannel = null;
    }
    if (this.commandChannel) {
      this.commandChannel.close();
      this.commandChannel = null;
    }
    this.initialized = false;
  }
}

// Create a singleton instance of the broadcast service
export const broadcastService = new BroadcastService();

// Export helper functions for common operations

// Broadcast a cancelation command to all tabs
export function broadcastCancelUpload(sessionId?: string) {
  broadcastService.broadcastUploadCommand({
    command: 'cancel',
    sessionId,
  });
  
  // Also clean up localStorage to ensure consistent state
  localStorage.removeItem('isUploading');
  localStorage.removeItem('uploadProgress');
  localStorage.removeItem('uploadSessionId');
  localStorage.removeItem('lastProgressTimestamp');
  localStorage.removeItem('serverRestartProtection');
  localStorage.removeItem('serverRestartTimestamp');
}