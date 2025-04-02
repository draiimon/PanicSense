/**
 * Platform Adapter for PanicSense
 * 
 * This utility normalizes behavior between different deployment environments
 * (Replit, Render, Local) to ensure consistent application behavior.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import { log } from '../vite';

// Platform detection
export const isReplit = Boolean(process.env.REPL_ID);
export const isRender = Boolean(process.env.RENDER);
export const isProduction = process.env.NODE_ENV === 'production';
export const platform = isRender ? 'Render' : isReplit ? 'Replit' : 'Local';

// Log platform information
log(`Platform detection: ${platform}`, 'platform');
log(`Production mode: ${isProduction ? 'Yes' : 'No'}`, 'platform');

// Normalize environment variables across platforms
export function normalizeEnv(): void {
  // Ensure PORT is set correctly
  if (!process.env.PORT) {
    process.env.PORT = '5000';
  }
  
  // Ensure DATABASE_URL exists
  if (!process.env.DATABASE_URL) {
    log('WARNING: DATABASE_URL is not set', 'platform');
  }
  
  // Normalize platform-specific paths
  if (isReplit) {
    // Replit-specific environment adjustments
    process.env.REPLIT = 'true';
    
    // On Replit, ensure assets directory exists and is properly linked
    ensureAssetsDirectory();
  } else if (isRender) {
    // Render-specific environment adjustments
    process.env.RENDER = 'true';
    
    // On Render, always set to production
    process.env.NODE_ENV = 'production';
  }
  
  log('Environment normalized for platform compatibility', 'platform');
}

// Function to ensure the assets directory exists and is properly set up
function ensureAssetsDirectory(): void {
  const assetsSourceDir = path.join(process.cwd(), 'attached_assets');
  const assetsTargetDir = path.join(process.cwd(), 'assets');
  
  if (!fs.existsSync(assetsSourceDir)) {
    log('attached_assets directory does not exist, creating it', 'platform');
    fs.mkdirSync(assetsSourceDir, { recursive: true });
  }
  
  // If assets directory doesn't exist, create link or copy from attached_assets
  if (!fs.existsSync(assetsTargetDir)) {
    try {
      // Try to create a symbolic link first
      fs.symlinkSync(assetsSourceDir, assetsTargetDir, 'dir');
      log('Created symbolic link for assets directory', 'platform');
    } catch (e: any) {
      log(`Could not create symbolic link: ${e?.message || 'Unknown error'}`, 'platform');
      log('Falling back to directory copy', 'platform');
      
      // If symlink fails, create directory and copy files
      fs.mkdirSync(assetsTargetDir, { recursive: true });
      
      try {
        const files = fs.readdirSync(assetsSourceDir);
        for (const file of files) {
          const sourcePath = path.join(assetsSourceDir, file);
          const targetPath = path.join(assetsTargetDir, file);
          fs.copyFileSync(sourcePath, targetPath);
        }
        log(`Copied ${files.length} files from attached_assets to assets`, 'platform');
      } catch (copyError: any) {
        log(`Error copying files: ${copyError?.message || 'Unknown error'}`, 'platform');
      }
    }
  }
}

// Get optimal temp directory based on platform
export function getOptimalTempDir(suffix: string = 'disaster-sentiment'): string {
  // Use RAM disk on Linux for better performance if available
  if (process.platform === 'linux' && fs.existsSync('/dev/shm')) {
    return path.join('/dev/shm', suffix);
  }
  
  // Replit and Render both have good temp directories
  return path.join(os.tmpdir(), suffix);
}

// Get optimal Python binary path
export function getPythonBinary(): string {
  if (process.env.PYTHON_BINARY) {
    return process.env.PYTHON_BINARY;
  }
  
  // Default to python3 which works on most platforms
  return 'python3';
}

// Initialize the platform adapter
export function initPlatformAdapter(): void {
  log('Initializing platform adapter...', 'platform');
  normalizeEnv();
  
  const tempDir = getOptimalTempDir();
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
    log(`Created temp directory: ${tempDir}`, 'platform');
  }
  
  log('Platform adapter initialized successfully', 'platform');
}