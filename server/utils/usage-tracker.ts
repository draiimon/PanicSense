import fs from 'fs';
import path from 'path';
import os from 'os';

// Interface for the usage data
interface UsageData {
  date: string;
  rowCount: number;
  lastReset: string;
}

class UsageTracker {
  private dataPath: string;
  private dailyLimit: number = 10000;
  
  constructor() {
    const dataDir = path.join(os.tmpdir(), 'disaster-sentiment');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    
    this.dataPath = path.join(dataDir, 'usage-data.json');
    this.initializeUsageData();
  }
  
  private initializeUsageData() {
    if (!fs.existsSync(this.dataPath)) {
      const initialData: UsageData = {
        date: this.getCurrentDate(),
        rowCount: 0,
        lastReset: new Date().toISOString()
      };
      
      fs.writeFileSync(this.dataPath, JSON.stringify(initialData, null, 2));
    }
  }
  
  private getCurrentDate(): string {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  }
  
  private getUsageData(): UsageData {
    try {
      const data = fs.readFileSync(this.dataPath, 'utf-8');
      return JSON.parse(data) as UsageData;
    } catch (error) {
      // If there's an error reading the file, initialize with fresh data
      const initialData: UsageData = {
        date: this.getCurrentDate(),
        rowCount: 0,
        lastReset: new Date().toISOString()
      };
      
      fs.writeFileSync(this.dataPath, JSON.stringify(initialData, null, 2));
      return initialData;
    }
  }
  
  private saveUsageData(data: UsageData) {
    fs.writeFileSync(this.dataPath, JSON.stringify(data, null, 2));
  }
  
  /**
   * Check if the current row count has reached the daily limit
   * @returns True if the limit has been reached, false otherwise
   */
  public hasReachedDailyLimit(): boolean {
    const currentData = this.getUsageData();
    const currentDate = this.getCurrentDate();
    
    // Reset the counter if it's a new day
    if (currentData.date !== currentDate) {
      currentData.date = currentDate;
      currentData.rowCount = 0;
      currentData.lastReset = new Date().toISOString();
      this.saveUsageData(currentData);
      return false;
    }
    
    return currentData.rowCount >= this.dailyLimit;
  }
  
  /**
   * Check if processing the given number of rows would exceed the daily limit
   * @param rowCount The number of rows to process
   * @returns The number of rows that can be processed without exceeding the limit, or 0 if the limit is already reached
   */
  public getProcessableRowCount(rowCount: number): number {
    const currentData = this.getUsageData();
    const currentDate = this.getCurrentDate();
    
    // Reset the counter if it's a new day
    if (currentData.date !== currentDate) {
      currentData.date = currentDate;
      currentData.rowCount = 0;
      currentData.lastReset = new Date().toISOString();
      this.saveUsageData(currentData);
      return Math.min(rowCount, this.dailyLimit);
    }
    
    const remainingQuota = this.dailyLimit - currentData.rowCount;
    return Math.max(0, Math.min(rowCount, remainingQuota));
  }
  
  /**
   * Get the current usage statistics
   * @returns The current usage data
   */
  public getUsageStats(): { used: number, limit: number, remaining: number, resetAt: string } {
    const currentData = this.getUsageData();
    const currentDate = this.getCurrentDate();
    
    // Reset the counter if it's a new day
    if (currentData.date !== currentDate) {
      currentData.date = currentDate;
      currentData.rowCount = 0;
      currentData.lastReset = new Date().toISOString();
      this.saveUsageData(currentData);
    }
    
    return {
      used: currentData.rowCount,
      limit: this.dailyLimit,
      remaining: Math.max(0, this.dailyLimit - currentData.rowCount),
      resetAt: currentData.lastReset
    };
  }
  
  /**
   * Increment the row count after processing rows
   * @param count The number of rows processed
   */
  public incrementRowCount(count: number) {
    const currentData = this.getUsageData();
    const currentDate = this.getCurrentDate();
    
    // Reset the counter if it's a new day
    if (currentData.date !== currentDate) {
      currentData.date = currentDate;
      currentData.rowCount = count;
      currentData.lastReset = new Date().toISOString();
    } else {
      currentData.rowCount += count;
    }
    
    this.saveUsageData(currentData);
  }
}

// Export a singleton instance
export const usageTracker = new UsageTracker();