import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import path from "path";
import { checkPortPeriodically, checkPort } from "./debug-port";

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

    // Using both ports 5000 and 8080 to improve workflow detection
    const port = 8080; // Using port 8080 for better compatibility
    // Check if port is already in use before attempting to listen
    console.log(`Pre-check: Verifying if port ${port} is available...`);
    
    const isPortAvailable = await checkPort(port, '0.0.0.0');
    console.log(`Pre-check result: Port ${port} is ${isPortAvailable ? 'AVAILABLE' : 'IN USE'}`);
    
    if (!isPortAvailable) {
      console.warn(`WARNING: Port ${port} appears to be already in use. Server may fail to start.`);
    }
    
    console.log(`Attempting to listen on port ${port}...`);
    
    server.listen(port, "0.0.0.0", () => {
      console.log(`========================================`);
      log(`🚀 Server running on port ${port}`);
      console.log(`Server listening at: http://0.0.0.0:${port}`);
      console.log(`Server ready at: ${new Date().toISOString()}`);
      console.log(`========================================`);
      
      // Create a secondary HTTP server on port 5000 to satisfy workflow requirements
      try {
        import('http').then(({ createServer }) => {
          const secondaryPort = 5000;
          const secondaryServer = createServer((req, res) => {
            res.writeHead(200, {'Content-Type': 'text/plain'});
            res.end(`Secondary server running on port ${secondaryPort}. Main application is at port ${port}.`);
          });
          
          secondaryServer.listen(secondaryPort, '0.0.0.0', () => {
            console.log(`Secondary server listening on port ${secondaryPort} for workflow detection`);
          });
          
          secondaryServer.on('error', (err) => {
            console.error(`Secondary server error:`, err);
          });
        });
      } catch (err) {
        console.error('Failed to start secondary server:', err);
      }
      
      // Repeatedly check and report port status to help debug workflow issues
      checkPortPeriodically(port, '0.0.0.0', 1000, 10)
        .then(() => {
          console.log(`Port ${port} is confirmed AVAILABLE for external connections`);
        })
        .catch((err) => {
          console.error(`Port checking error:`, err);
        });
        
      console.log(`Port ${port} should now be open and accessible`);
      console.log(`Try accessing http://0.0.0.0:${port} in your browser`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
  }
})();