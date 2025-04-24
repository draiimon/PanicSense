import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Special compatibility config for Render deployment
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'client/dist',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    port: process.env.PORT || 5000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), 'client', 'src'),
      '@shared': path.resolve(process.cwd(), 'shared'),
    },
  },
});