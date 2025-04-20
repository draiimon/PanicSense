import { db } from '../db';
import { eq, and, or, lt, gt, sql } from 'drizzle-orm';
import { uploadSessions } from '@shared/schema';
import { SERVER_START_TIMESTAMP } from '../index';

// Helper function to ensure timestamp is always a string
function normalizeTimestamp(timestamp: string | number | Date | unknown): string {
  if (timestamp === null || timestamp === undefined) {
    return Date.now().toString(); // Default to current time
  }
  
  if (typeof timestamp === 'number') {
    return timestamp.toString();
  }
  
  if (timestamp instanceof Date) {
    return timestamp.getTime().toString();
  }
  
  // If it's already a string, return it
  if (typeof timestamp === 'string') {
    return timestamp;
  }
  
  // For any other types, convert to string safely
  try {
    return String(timestamp);
  } catch (e) {
    console.warn('Failed to normalize timestamp, using current time', e);
    return Date.now().toString();
  }
}

/**
 * Enhanced upload session manager to ensure clean uploads and prevent orphaned sessions
 */
export class UploadSessionManager {
  // Stale session threshold in milliseconds (30 minutes)
  private readonly STALE_SESSION_THRESHOLD = 30 * 60 * 1000;
  
  /**
   * Get an active upload session by ID
   */
  async getSessionById(sessionId: string) {
    try {
      const session = await db.select().from(uploadSessions)
        .where(eq(uploadSessions.sessionId, sessionId))
        .limit(1);
      
      return session.length > 0 ? session[0] : null;
    } catch (error) {
      console.error('Error getting upload session:', error);
      return null;
    }
  }
  
  /**
   * Find any active upload sessions
   * A session is considered active if it's in 'processing' status and not stale
   */
  async findActiveSession() {
    try {
      // Just use direct SQL to avoid schema issues with column names
      const result = await db.execute(sql`
        SELECT * FROM upload_sessions 
        WHERE status = 'processing' 
        LIMIT 1
      `);
      
      if (result.rows && result.rows.length > 0) {
        return result.rows[0];
      }
      
      return null;
    } catch (error) {
      console.error('Error finding active upload session:', error);
      return null;
    }
  }
  
  /**
   * Create a new upload session
   */
  async createSession(sessionId: string, fileId: number | null = null) {
    try {
      const now = new Date();
      
      await db.insert(uploadSessions).values({
        sessionId,
        status: 'processing',
        fileId,
        createdAt: now,
        updatedAt: now,
        serverStartTimestamp: normalizeTimestamp(SERVER_START_TIMESTAMP),
        progress: JSON.stringify({
          processed: 0,
          total: 0,
          stage: 'Initializing...',
          timestamp: Date.now()
        })
      });
      
      return true;
    } catch (error) {
      console.error('Error creating upload session:', error);
      return false;
    }
  }
  
  /**
   * Update an existing upload session
   */
  async updateSession(sessionId: string, status: string, progress: any) {
    try {
      // First check if the session exists
      const existingSession = await this.getSessionById(sessionId);
      if (!existingSession) {
        return false;
      }
      
      // Convert progress to string if it's an object
      const progressStr = typeof progress === 'object' 
        ? JSON.stringify(progress)
        : progress;
      
      await db.update(uploadSessions)
        .set({
          status,
          progress: progressStr,
          updatedAt: new Date(),
          serverStartTimestamp: normalizeTimestamp(SERVER_START_TIMESTAMP)
        })
        .where(eq(uploadSessions.sessionId, sessionId));
      
      return true;
    } catch (error) {
      console.error('Error updating upload session:', error);
      return false;
    }
  }
  
  /**
   * Mark a session as complete
   */
  async completeSession(sessionId: string) {
    return this.updateSession(sessionId, 'completed', {
      stage: 'Analysis complete',
      timestamp: Date.now(),
      isComplete: true
    });
  }
  
  /**
   * Mark a session as error
   */
  async errorSession(sessionId: string, errorMessage: string) {
    return this.updateSession(sessionId, 'error', {
      stage: `Error: ${errorMessage}`,
      timestamp: Date.now(),
      error: errorMessage
    });
  }
  
  /**
   * Mark a session as canceled
   */
  async cancelSession(sessionId: string) {
    return this.updateSession(sessionId, 'canceled', {
      stage: 'Upload canceled',
      timestamp: Date.now(),
      isCanceled: true
    });
  }
  
  /**
   * Delete a session
   */
  async deleteSession(sessionId: string) {
    try {
      await db.delete(uploadSessions)
        .where(eq(uploadSessions.sessionId, sessionId));
      
      return true;
    } catch (error) {
      console.error('Error deleting upload session:', error);
      return false;
    }
  }
  
  /**
   * Clean up stale or error sessions
   * This should be called periodically to prevent orphaned sessions
   */
  async cleanupSessions() {
    try {
      // Use direct SQL to avoid schema issues with column names
      const result = await db.execute(sql`
        SELECT * FROM upload_sessions 
        WHERE status = 'error' OR status = 'canceled'
      `);
      
      const sessionsToClean = result.rows || [];
      
      // Delete all sessions that need cleaning
      if (sessionsToClean.length > 0) {
        const sessionIds = sessionsToClean.map(s => s.sessionId);
        
        await db.delete(uploadSessions)
          .where(
            sql`${uploadSessions.sessionId} IN (${sessionIds.join(',')})` as any
          );
        
        console.log(`Cleaned up ${sessionsToClean.length} stale or error upload sessions`);
      }
      
      return sessionsToClean.length;
    } catch (error) {
      console.error('Error cleaning up upload sessions:', error);
      return 0;
    }
  }
}

// Export a singleton instance
export const uploadSessionManager = new UploadSessionManager();