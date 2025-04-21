# STAGE 1: Install Node dependencies and build app
FROM node:20.19.0-slim AS node-builder
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --production=false --no-audit --no-fund --prefer-offline

# Copy source files and config
COPY client ./client
COPY server ./server
COPY shared ./shared
COPY public ./public
COPY drizzle.config.ts tailwind.config.ts tsconfig.json vite.config.ts theme.json ./

# Build the application
RUN npm run build

# Prepare directories for runtime usage
RUN mkdir -p server/public && \
    cp -r dist/* server/public/ && \
    mkdir -p client/dist && \
    cp -r dist/* client/dist/

# STAGE 2: Final image with Python + Node
FROM python:3.11-slim
WORKDIR /app

# Install Node.js and required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    libpq-dev \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && npm cache clean --force

# Copy essential built artifacts and source code from node-builder
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/public ./public
COPY --from=node-builder /app/shared ./shared
COPY --from=node-builder /app/migrations ./migrations
COPY --from=node-builder /app/client ./client
COPY --from=node-builder /app/server ./server
COPY --from=node-builder /app/client/dist ./client/dist
COPY --from=node-builder /app/server/public ./server/public
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/index.js ./index.js

# Ensure static folders exist for runtime
RUN mkdir -p client/dist && \
    mkdir -p server/public && \
    mkdir -p uploads/temp && chmod 777 uploads/temp

# Copy package.json for reference
COPY package*.json ./

# Python requirements
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --timeout=180 --retries=3 torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --timeout=180 --retries=3 -r server/python/requirements.txt \
 && find /usr/local -name '*.pyc' -delete \
 && find /usr/local -name '__pycache__' -delete \
 && pip cache purge

# Environment variables
ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHON_PATH=python3
ENV TZ=Asia/Manila

# Expose port
EXPOSE 5000

# Health check script
RUN echo '#!/bin/bash\n\
HEALTH_ENDPOINT="http://localhost:5000/api/health"\n\
RESPONSE=$(curl -s -f -m 3 $HEALTH_ENDPOINT 2>/dev/null)\n\
if [ $? -ne 0 ]; then\n\
  echo "Health check failed: Server not responding"\n\
  exit 1\n\
fi\n\
if ! echo "$RESPONSE" | grep -q "\"status\":\"ok\""; then\n\
  echo "Health check failed: Invalid response from server"\n\
  exit 1\n\
fi\n\
echo "Health check passed: Application is running correctly"\n\
exit 0\n' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /app/healthcheck.sh

# Copy drizzle-kit binary
COPY --from=node-builder /app/node_modules/.bin/drizzle-kit /app/node_modules/.bin/drizzle-kit

# Startup script with static file verification
RUN echo '#!/bin/bash\n\
echo "🚀 Starting PanicSense PH with enhanced database reliability..."\n\
MAX_RETRIES=${DB_CONNECTION_RETRY_ATTEMPTS:-5}\n\
RETRY_DELAY=${DB_CONNECTION_RETRY_DELAY_MS:-3000}\n\
echo "📊 Database connection settings:"\n\
echo "  - Max retries: $MAX_RETRIES"\n\
echo "  - Retry delay: $RETRY_DELAY ms"\n\
echo "⏳ Waiting for database to be ready..."\n\
sleep 3\n\
echo "💡 Checking static directories..."\n\
if [ ! -d "/app/client/dist" ]; then echo "❌ Missing /app/client/dist"; exit 1; fi\n\
if [ ! -d "/app/server/public" ]; then echo "❌ Missing /app/server/public"; exit 1; fi\n\
if [ ! -d "/app/dist" ]; then echo "❌ Missing /app/dist"; exit 1; fi\n\
echo "✅ Starting application..."\n\
node index.js\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
