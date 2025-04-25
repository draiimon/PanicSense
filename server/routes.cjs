/**
 * CommonJS Routes for Render Deployment
 * This file provides API routes in CommonJS format for maximum compatibility
 */

const path = require('path');
const fs = require('fs');
const express = require('express');
const multer = require('multer');

// Create uploads directory if it doesn't exist
const uploadDir = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Use a unique filename with original extension
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB file size limit
});

/**
 * Register all API routes for the application
 * @param {express.Application} app - Express application instance
 */
async function registerRoutes(app) {
  console.log('📝 Registering API routes (CJS version)...');

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      version: '1.0.0',
      mode: 'production',
      timestamp: new Date().toISOString()
    });
  });

  // Fallback API endpoint
  app.get('/api/*', (req, res) => {
    res.status(404).json({
      error: 'API endpoint not found',
      message: 'This API endpoint is not implemented in the minimal CJS version',
      path: req.path
    });
  });

  return app;
}

module.exports = { registerRoutes };