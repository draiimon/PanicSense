/**
 * Upload Persistence Module
 * 
 * This module handles persistence of upload state across page refreshes
 * using localStorage. It provides a consistent interface for saving and
 * retrieving upload progress information.
 */

// Types
export interface UploadProgressStorage {
  processed: number;
  total: number;
  stage: string;
  error?: string;
  batchNumber?: number;
  totalBatches?: number;
  batchProgress?: number;
  currentSpeed?: number;  // Records per second
  timeRemaining?: number; // Seconds
  processingStats?: {
    successCount: number;
    errorCount: number;
    lastBatchDuration: number;
    averageSpeed: number;
  };
}

// Storage keys
const UPLOAD_SESSION_KEY = 'uploadSessionId';
const UPLOAD_PROGRESS_KEY = 'uploadProgress';
const IS_UPLOADING_KEY = 'isUploading';

// Helper function to safely access localStorage
function safelyAccessStorage<T>(
  operation: () => T,
  fallback: T
): T {
  try {
    return operation();
  } catch (error) {
    console.warn('LocalStorage access failed:', error);
    return fallback;
  }
}

/**
 * Check if an upload is in progress according to localStorage
 */
export function isUploading(): boolean {
  return safelyAccessStorage(() => localStorage.getItem(IS_UPLOADING_KEY) === 'true', false);
}

/**
 * Get the current upload session ID if available
 */
export function getUploadSessionId(): string | null {
  return safelyAccessStorage(() => localStorage.getItem(UPLOAD_SESSION_KEY), null);
}

/**
 * Set the current upload session ID
 */
export function setUploadSessionId(sessionId: string): void {
  safelyAccessStorage(() => localStorage.setItem(UPLOAD_SESSION_KEY, sessionId), undefined);
}

/**
 * Get the current upload progress if available
 */
export function getUploadProgress(): UploadProgressStorage | null {
  return safelyAccessStorage(() => {
    const storedProgress = localStorage.getItem(UPLOAD_PROGRESS_KEY);
    if (storedProgress) {
      return JSON.parse(storedProgress) as UploadProgressStorage;
    }
    return null;
  }, null);
}

/**
 * Save the current upload progress
 */
export function saveUploadProgress(progress: UploadProgressStorage): void {
  safelyAccessStorage(() => {
    localStorage.setItem(UPLOAD_PROGRESS_KEY, JSON.stringify(progress));
    localStorage.setItem(IS_UPLOADING_KEY, 'true');
  }, undefined);
}

/**
 * Start tracking an upload with the given session ID
 */
export function startTrackingUpload(sessionId: string, initialProgress: UploadProgressStorage): void {
  safelyAccessStorage(() => {
    localStorage.setItem(UPLOAD_SESSION_KEY, sessionId);
    localStorage.setItem(UPLOAD_PROGRESS_KEY, JSON.stringify(initialProgress));
    localStorage.setItem(IS_UPLOADING_KEY, 'true');
  }, undefined);
}

/**
 * Clear all upload state (used after completion or cancellation)
 */
export function clearUploadState(): void {
  safelyAccessStorage(() => {
    localStorage.removeItem(UPLOAD_SESSION_KEY);
    localStorage.removeItem(UPLOAD_PROGRESS_KEY);
    localStorage.removeItem(IS_UPLOADING_KEY);
  }, undefined);
}