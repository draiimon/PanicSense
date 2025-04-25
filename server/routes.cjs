/**
 * Pure CommonJS version of the routes file for PanicSense
 * For use with Render.com deployment
 */

// CommonJS imports
const express = require('express');
const path = require('path');
const fs = require('fs');
const { Pool } = require('@neondatabase/serverless');

// Initialize database connection
const databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;
const pool = new Pool({ 
  connectionString: databaseUrl,
  ssl: { rejectUnauthorized: false }
});

// Simple function to broadcast updates
function broadcastUpdate(data) {
  console.log('🔄 [SERVER] Broadcasting update:', data?.stage || 'unknown stage');
}

// Register all routes for the application
async function registerRoutes(app) {
  console.log('📝 Registering API routes...');

  // Health check API
  app.get('/api/health', (req, res) => {
    res.json({ 
      status: 'ok',
      time: new Date().toISOString(),
      message: 'PanicSense API is operational!',
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development'
    });
  });

  // Get all sentiment posts
  app.get('/api/posts', async (req, res) => {
    try {
      const result = await pool.query(
        'SELECT * FROM sentiment_posts ORDER BY timestamp DESC LIMIT 100'
      );
      res.json(result.rows);
    } catch (error) {
      console.error('❌ Error fetching posts:', error);
      res.status(500).json({ error: 'Failed to fetch posts' });
    }
  });

  // Get active upload session
  app.get('/api/active-upload-session', async (req, res) => {
    try {
      // Check for active upload session in database
      const result = await pool.query(
        'SELECT * FROM upload_sessions WHERE status = $1 ORDER BY created_at DESC LIMIT 1',
        ['active']
      );
      
      const sessionId = result.rows.length > 0 ? result.rows[0].id : null;
      console.log('⭐ No active sessions found in database');
      
      res.json({ sessionId });
    } catch (error) {
      console.error('❌ Error checking for active upload session:', error);
      res.status(500).json({ error: 'Failed to check for active upload session' });
    }
  });

  // Stats API
  app.get('/api/stats', async (req, res) => {
    try {
      // Get total posts count
      const postsResult = await pool.query('SELECT COUNT(*) FROM sentiment_posts');
      const totalPosts = parseInt(postsResult.rows[0].count);
      
      // Get posts by source
      const sourceResult = await pool.query(
        'SELECT source, COUNT(*) FROM sentiment_posts GROUP BY source ORDER BY count DESC'
      );
      const sourceStats = sourceResult.rows;
      
      // Get posts by sentiment
      const sentimentResult = await pool.query(
        'SELECT sentiment, COUNT(*) FROM sentiment_posts GROUP BY sentiment ORDER BY count DESC'
      );
      const sentimentStats = sentimentResult.rows;
      
      // Get posts by language
      const languageResult = await pool.query(
        'SELECT language, COUNT(*) FROM sentiment_posts GROUP BY language ORDER BY count DESC'
      );
      const languageStats = languageResult.rows;
      
      res.json({
        totalPosts,
        sourceStats,
        sentimentStats,
        languageStats
      });
    } catch (error) {
      console.error('❌ Error fetching stats:', error);
      res.status(500).json({ error: 'Failed to fetch statistics' });
    }
  });

  console.log('✅ API routes registered successfully');
  return app;
}

// Export the function for use in the main server file
module.exports = { registerRoutes, broadcastUpdate };