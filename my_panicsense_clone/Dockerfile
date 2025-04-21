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

# Python ML deps
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r server/python/requirements.txt \
 && find /usr/local -name '*.pyc' -delete \
 && find /usr/local -name '__pycache__' -delete

# Copy start script & make it executable
COPY start.sh ./
RUN chmod +x start.sh

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
