# ───────────────────────────────────────────────────
# STAGE 1: Build everything (client + server)
# ───────────────────────────────────────────────────
FROM node:20.19.0-slim AS builder
WORKDIR /app

# Copy only lockfiles/manifest for caching if you had one; here just copy all
COPY package*.json ./

# Install deps for both client & server
RUN npm install --no-audit --no-fund --prefer-offline

# Copy entire source & run unified build
COPY . .
RUN npm run build

# ───────────────────────────────────────────────────
# STAGE 2: Runtime with Python + Node
# ───────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Install system deps + Node runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg libpq-dev \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && npm cache clean --force

# Copy built artifacts & node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/server ./server
# The client/dist doesn't exist in this project structure - using dist/public instead
COPY --from=builder /app/dist/public ./client/dist
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json

# Copy other needed files
COPY --from=builder /app/migrations ./migrations
COPY --from=builder /app/shared ./shared
COPY --from=builder /app/drizzle.config.ts ./drizzle.config.ts
COPY --from=builder /app/server.js ./server.js

# Copy start script & make it executable
COPY start.sh ./
RUN chmod +x start.sh

# Install Python ML deps
COPY --from=builder /app/server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --timeout=180 --retries=3 \
      torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r server/python/requirements.txt \
 && find /usr/local -name '*.pyc' -delete \
 && find /usr/local -name '__pycache__' -delete \
 && pip cache purge

# Environment
ENV NODE_ENV=production \
    PORT=5000 \
    TZ=Asia/Manila
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sSf http://localhost:$PORT/api/health || exit 1

# Launch
CMD ["sh", "./start.sh"]