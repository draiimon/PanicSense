
/**
 * DIRECT DELETE ALL DATA
 * Provides a direct endpoint to delete all data
 */

import express from 'express';
import { pool } from '../db.js';

const router = express.Router();

// Direct delete all data endpoint
router.delete('/api/direct-delete-all', async (req, res) => {
  try {
    console.log("⚠️ EXECUTING DIRECT DELETE ALL DATA ⚠️");
    
    // Delete all data from all tables
    await pool.query("DELETE FROM sentiment_feedback");
    await pool.query("DELETE FROM training_examples");
    await pool.query("DELETE FROM upload_sessions");
    await pool.query("DELETE FROM sentiment_posts");
    await pool.query("DELETE FROM disaster_events");
    await pool.query("DELETE FROM analyzed_files");
    
    console.log("✅ All data successfully deleted");
    
    res.json({ 
      success: true, 
      message: "All data has been deleted successfully"
    });
  } catch (error) {
    console.error("❌ Error deleting all data:", error);
    res.status(500).json({ 
      error: "Failed to delete all data",
      details: error.message || String(error)
    });
  }
});

// Check database status endpoint
router.get('/api/check-db-status', async (req, res) => {
  try {
    // Get counts from all tables
    const tables = ['sentiment_posts', 'disaster_events', 'analyzed_files', 'upload_sessions'];
    const counts = {};
    
    for (const table of tables) {
      const result = await pool.query(`SELECT COUNT(*) FROM ${table}`);
      counts[table] = parseInt(result.rows[0].count);
    }
    
    const result = await pool.query("SELECT NOW() AS server_time");
    const serverTime = result.rows[0].server_time;
    
    res.json({
      status: "connected",
      serverTime,
      counts,
      database_url_set: !!process.env.DATABASE_URL,
      database_url_length: process.env.DATABASE_URL ? process.env.DATABASE_URL.length : 0
    });
  } catch (error) {
    console.error("❌ Error checking database status:", error);
    res.status(500).json({
      status: "error",
      error: error.message || String(error)
    });
  }
});

export default router;
