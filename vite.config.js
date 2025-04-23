// vite.config.js - Simple version for Render.com
import { defineConfig } from 'vite';

export default defineConfig({
  // Force production mode
  mode: 'production',
  
  // Basic build config
  build: {
    outDir: 'dist/public',
    emptyOutDir: true,
    minify: true,
    sourcemap: false,
  },
  
  // Fast startup
  optimizeDeps: {
    exclude: ['@replit/vite-plugin-cartographer', '@replit/vite-plugin-runtime-error-modal']
  },
  
  // Don't use plugins in Render (they might not be compatible)
  plugins: []
});