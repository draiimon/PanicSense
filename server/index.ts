import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import path from "path";
import { checkPortPeriodically, checkPort } from "./debug-port";
import compression from 'compression'; // Will add this package next
import cors from 'cors';

// Detect environment for optimized settings
const isProduction = process.env.NODE_ENV === 'production';
const isRender = Boolean(process.env.RENDER);
const isReplit = Boolean(process.env.REPL_ID);

console.log(`Running in ${isProduction ? 'PRODUCTION' : 'DEVELOPMENT'} mode`);
console.log(`Platform: ${isRender ? 'Render' : isReplit ? 'Replit' : 'Local'}`);

const app = express();

// Performance optimizations
app.use(compression()); // Compress all responses
app.use(cors()); // Enable CORS for API requests
app.use(express.json({ limit: '50mb' })); // Increased limit for better performance
app.use(express.urlencoded({ extended: false, limit: '50mb' }));

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// Enhanced logging middleware with better performance metrics
app.use((req, res, next) => {
  // Skip logging for static assets to reduce noise
  if (req.path.startsWith('/assets/') || req.path.endsWith('.ico')) {
    return next();
  }
  
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

    // Using just port 5000 as the main port
    const port = 5000;
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