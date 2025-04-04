// upload-persistence.ts
// This file handles persistent state management for upload operations across browser sessions

import { UploadProgress as ApiUploadProgress } from './api';

// Define our own version of UploadProgress to avoid circular dependencies
export interface UploadProgress {
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
function getStorageItem<T>(key: string): T | null {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    console.error(`Error reading ${key} from localStorage:`, error);
    return null;
  }
}

// Helper function to safely write to localStorage
function setStorageItem(key: string, value: any): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Error writing ${key} to localStorage:`, error);
    return false;
  }
}

// Helper function to remove items from localStorage
function removeStorageItem(key: string): boolean {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error(`Error removing ${key} from localStorage:`, error);
    return false;
  }
}

// Save upload session data
export function saveUploadSession(sessionId: string): boolean {
  return setStorageItem(UPLOAD_SESSION_KEY, sessionId);
}

// Get upload session ID
export function getUploadSessionId(): string | null {
  const rawValue = localStorage.getItem(UPLOAD_SESSION_KEY);
  return rawValue; // Don't parse as JSON since it's a string
}

// Save upload progress
export function saveUploadProgress(progress: UploadProgress): boolean {
  return setStorageItem(UPLOAD_PROGRESS_KEY, progress);
}

// Get upload progress
export function getUploadProgress(): UploadProgress | null {
  return getStorageItem<UploadProgress>(UPLOAD_PROGRESS_KEY);
}

// Save uploading state
export function saveUploadingState(isUploading: boolean): boolean {
  return setStorageItem(IS_UPLOADING_KEY, isUploading);
}

// Get uploading state
export function isUploading(): boolean {
  return getStorageItem<boolean>(IS_UPLOADING_KEY) || false;
}

// Clear all upload state
export function clearUploadState(): void {
  removeStorageItem(UPLOAD_SESSION_KEY);
  removeStorageItem(UPLOAD_PROGRESS_KEY);
  removeStorageItem(IS_UPLOADING_KEY);
}

// Function to register a callback for storage events
// This allows other tabs to detect when upload state changes
export function registerStorageEventListener(callback: (event: StorageEvent) => void): void {
  window.addEventListener('storage', callback);
}

// Function to remove storage event listener
export function removeStorageEventListener(callback: (event: StorageEvent) => void): void {
  window.removeEventListener('storage', callback);
}

// Check if there's an in-progress upload based on localStorage
export function hasActiveUpload(): boolean {
  return isUploading() && !!getUploadSessionId();
}

// Function to store all upload state at once
export function saveUploadState(sessionId: string, progress: UploadProgress, uploading: boolean = true): boolean {
  try {
    saveUploadSessionId(sessionId);
    saveUploadProgress(progress);
    saveUploadingState(uploading);
    return true;
  } catch (error) {
    console.error('Error saving upload state:', error);
    return false;
  }
}

// Save just the session ID to localStorage
export function saveUploadSessionId(sessionId: string | null): boolean {
  if (sessionId === null) {
    removeStorageItem(UPLOAD_SESSION_KEY);
    return true;
  }
  try {
    localStorage.setItem(UPLOAD_SESSION_KEY, sessionId);
    return true;
  } catch (error) {
    console.error('Error saving upload session ID:', error);
    return false;
  }
}

// Get all upload state at once
export function getUploadState(): {
  sessionId: string | null;
  progress: UploadProgress | null;
  isUploading: boolean;
} {
  return {
    sessionId: getUploadSessionId(),
    progress: getUploadProgress(),
    isUploading: isUploading()
  };
}