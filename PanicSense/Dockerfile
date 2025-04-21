# 1) Builder Stage: build client + install server deps
FROM node:20.19.0-slim AS builder
WORKDIR /app

# Copy and install client
COPY client/package*.json ./client/
WORKDIR /app/client
RUN npm ci --no-audit --no-fund --prefer-offline
# Copy client source files
COPY client/ ./
# Build React app
RUN npm run build

# Copy and install server dependencies
WORKDIR /app
COPY server/package*.json ./server/
WORKDIR /app/server
RUN npm ci --no-audit --no-fund --prefer-offline
# Copy server source files
COPY server/ ./

# Stage static output for Express
WORKDIR /app
RUN mkdir -p server/public \
 && cp -r client/dist/* server/public/

# 2) Final Stage: runtime with Python + Node
FROM python:3.11-slim
WORKDIR /app

# Install system deps + Node.js runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg libpq-dev \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && npm cache clean --force

# Copy server code + node_modules
COPY --from=builder /app/server /app/server
COPY --from=builder /app/server/node_modules /app/server/node_modules

# Copy built client dist directly to server/public
COPY --from=builder /app/client/dist /app/client/dist
COPY --from=builder /app/server/public /app/server/public

# Copy other needed files
COPY migrations ./migrations
COPY shared ./shared
COPY drizzle.config.ts ./
COPY package*.json ./
COPY server.js ./

# Copy start script
COPY start.sh ./
RUN chmod +x start.sh

# Install Python ML deps
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --timeout=180 --retries=3 \
      torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r server/python/requirements.txt \
 && find /usr/local -name '*.pyc' -delete \
 && find /usr/local -name '__pycache__' -delete \
 && pip cache purge

# Env & expose
ENV NODE_ENV=production \
    PORT=5000 \
    TZ=Asia/Manila
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
 CMD curl -sSf http://localhost:$PORT/api/health || exit 1

# Entry
CMD ["/bin/bash", "./start.sh"]