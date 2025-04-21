# STAGE 1: Install Node dependencies and build app
FROM node:20.19.0-slim AS node-builder
WORKDIR /app

# Copy package files and install dependencies (with production flag for lighter install)
COPY package*.json ./
RUN npm ci --production=false --no-audit --no-fund --prefer-offline

# Copy source files and build
COPY client ./client
COPY server ./server
COPY shared ./shared
COPY public ./public
COPY drizzle.config.ts tailwind.config.ts tsconfig.json vite.config.ts theme.json ./

# Build the application with optimizations
RUN npm run build
# Verify the build output structure
RUN ls -la dist/

# STAGE 2: Final image with Python + Node
FROM python:3.11-slim

# Install Node.js and only the required dependencies
# Optimized to reduce image size by combining commands and cleaning up
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    libpq-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && npm cache clean --force

# Create app directory
WORKDIR /app

# Copy built files from node-builder
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/public ./public
COPY --from=node-builder /app/node_modules ./node_modules

# Make a copy of dist to the public directory to match the vite.ts expectations
RUN if [ -d "dist" ]; then mkdir -p server/public && cp -r dist/* server/public/; fi

# Copy package files (only production files)
COPY package*.json ./

# Copy Python requirements and install
# Optimized Python installation to improve build speed and reduce image size
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --timeout=180 --retries=3 torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout=180 --retries=3 -r server/python/requirements.txt \
    && find /usr/local -name '*.pyc' -delete \
    && find /usr/local -name '__pycache__' -delete \
    && pip cache purge

# Copy only needed files for runtime
COPY migrations ./migrations
COPY server ./server
COPY shared ./shared
COPY index.js ./

# Create needed directories for file uploads
RUN mkdir -p uploads/temp && chmod 777 uploads/temp

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHON_PATH=python3
ENV TZ=Asia/Manila

# Expose the port
EXPOSE 5000

# Create a healthcheck script for more reliable health checks
RUN echo '#!/bin/bash\n\
# Comprehensive health check that verifies both the web server and database\n\
HEALTH_ENDPOINT="http://localhost:5000/api/health"\n\
\n\
# Check if the server is responding\n\
RESPONSE=$(curl -s -f -m 3 $HEALTH_ENDPOINT 2>/dev/null)\n\
if [ $? -ne 0 ]; then\n\
  echo "Health check failed: Server not responding"\n\
  exit 1\n\
fi\n\
\n\
# Make sure it returns a valid JSON with status ok\n\
if ! echo "$RESPONSE" | grep -q "\"status\":\"ok\""; then\n\
  echo "Health check failed: Invalid response from server"\n\
  exit 1\n\
fi\n\
\n\
# All checks passed\n\
echo "Health check passed: Application is running correctly"\n\
exit 0\n\
' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

# Add health check for better container stability
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /app/healthcheck.sh

# Adding database connection reliability script
COPY --from=node-builder /app/node_modules/.bin/drizzle-kit /app/node_modules/.bin/drizzle-kit

# Create startup script for better database connection handling
RUN echo '#!/bin/bash\n\
echo "🚀 Starting PanicSense PH with enhanced database reliability..."\n\
\n\
MAX_RETRIES=${DB_CONNECTION_RETRY_ATTEMPTS:-5}\n\
RETRY_DELAY=${DB_CONNECTION_RETRY_DELAY_MS:-3000}\n\
\n\
echo "📊 Database connection settings:"\n\
echo "  - Max retries: $MAX_RETRIES"\n\
echo "  - Retry delay: $RETRY_DELAY ms"\n\
\n\
echo "⏳ Waiting for database to be ready..."\n\
sleep 3\n\
\n\
echo "✅ Starting application..."\n\
node index.js\n\
' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]