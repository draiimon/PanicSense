import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import path from "path";
import { checkPortPeriodically, checkPort } from "./debug-port";
// Import emergency database fixes - both old and new strategy
import { applyEmergencyFixes } from "./emergency-db-fix";
// Import the simple fix (now ESM compatible)
import { simpleDbFix } from "./db-simple-fix";

const app = express();
app.use(express.json({ limit: '50mb' })); // Increased limit for better performance
app.use(express.urlencoded({ extended: false, limit: '50mb' }));

// Enhanced logging middleware with better performance metrics
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        const summary = JSON.stringify(capturedJsonResponse).slice(0, 100);
        logLine += ` :: ${summary}${summary.length > 99 ? '...' : ''}`;
      }
      log(logLine);
    }
  });

  next();
});

(async () => {
  try {
    console.log("========================================");
    console.log("Starting server initialization at: " + new Date().toISOString());
    console.log("========================================");
    
    // Apply emergency database fixes before anything else
    if (process.env.NODE_ENV === "production") {
      console.log("Running in production mode, applying emergency database fixes...");
      try {
        // First, try the simple fix (more reliable, simpler code)
        console.log("Trying simple database fix first...");
        const simpleFixSuccessful = await simpleDbFix();
        
        if (simpleFixSuccessful) {
          console.log("✅ Simple database fix successful!");
        } else {
          // If simple fix fails, try the more complex one
          console.log("Simple fix failed, trying complex emergency fix...");
          const fixesSuccessful = await applyEmergencyFixes();
          if (fixesSuccessful) {
            console.log("✅ Complex database fixes applied successfully!");
          } else {
            console.error("⚠️ All database fixes failed. App may not function correctly.");
          }
        }
      } catch (error) {
        console.error("Fatal error in database fix script:", error);
      }
    } else {
      // In development, also apply the simple fix (helps with local testing)
      try {
        console.log("Running simple database fix in development...");
        await simpleDbFix();
      } catch (error) {
        console.error("Development database fix failed:", error);
      }
    }
    
    const server = await registerRoutes(app);
    console.log("Routes registered successfully");

    // Enhanced error handling with structured error response
    app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
      const status = err.status || err.statusCode || 500;
      const message = err.message || "Internal Server Error";
      const details = err.stack || "";

      console.error(`[Error] ${status} - ${message}\n${details}`);
      res.status(status).json({ 
        error: true,
        message,
        timestamp: new Date().toISOString(),
        path: _req.path
      });
    });

    console.log("Current NODE_ENV:", process.env.NODE_ENV);
    
    if (process.env.NODE_ENV === "production") {
      console.log("Running in production mode, serving static files...");
      serveStatic(app);
      console.log("Static file serving setup complete");
    } else {
      console.log("Running in development mode, setting up Vite middleware...");
      await setupVite(app, server);
      console.log("Vite middleware setup complete");
    }

    // Use PORT environment variable with fallback to 5000 for local development
    const port = parseInt(process.env.PORT || "5000", 10);
    console.log(`Attempting to listen on port ${port}...`);
    
    server.listen(port, "0.0.0.0", () => {
      console.log(`========================================`);
      log(`🚀 Server running on port ${port}`);
      console.log(`Server listening at: http://0.0.0.0:${port}`);
      console.log(`Server ready at: ${new Date().toISOString()}`);
      console.log(`========================================`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
  }
})();