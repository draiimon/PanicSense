import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

const startTime = Date.now();
log("Starting application initialization...", "startup");

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
    log("Initializing server components...", "startup");
    log(`Time since start: ${Date.now() - startTime}ms`, "startup");

    log("Registering routes...", "startup");
    const server = await registerRoutes(app);
    log(`Routes registered in ${Date.now() - startTime}ms`, "startup");

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

    log("Setting up application middleware...", "startup");
    if (app.get("env") === "development") {
      log("Initializing Vite middleware...", "startup");
      await setupVite(app, server);
      log(`Vite middleware setup complete in ${Date.now() - startTime}ms`, "startup");
    } else {
      log("Setting up static file serving...", "startup");
      serveStatic(app);
      log("Static file serving configured", "startup");
    }

    // ALWAYS serve the app on port 5000
    const port = 5000;
    try {
      log("Attempting to start server...", "startup");
      server.listen({
        port,
        host: "0.0.0.0",
        reusePort: true,
      }, () => {
        log(`🚀 Server ready and listening on port ${port}`, "startup");
        log(`Server running in ${app.get("env")} mode`, "startup");
        log(`Total startup time: ${Date.now() - startTime}ms`, "startup");
      });

      // Handle server errors
      server.on('error', (error: NodeJS.ErrnoException) => {
        if (error.code === 'EADDRINUSE') {
          console.error(`❌ Port ${port} is already in use`);
          process.exit(1);
        } else {
          console.error('❌ Server error:', error);
          process.exit(1);
        }
      });
    } catch (error) {
      console.error('❌ Failed to start server:', error);
      process.exit(1);
    }
  } catch (error) {
    console.error('❌ Fatal error during startup:', error);
    process.exit(1);
  }
})();