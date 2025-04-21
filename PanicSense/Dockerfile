# STAGE 1: Install Node dependencies and build app
FROM node:20.19.0-slim AS node-builder
WORKDIR /app

# 1. Copy package files and install all deps
COPY package*.json ./
RUN npm ci --production=false --no-audit --no-fund --prefer-offline

# 2. Copy source & config, then build
COPY client ./client
COPY server ./server
COPY shared ./shared
COPY public ./public
COPY drizzle.config.ts tailwind.config.ts tsconfig.json vite.config.ts theme.json ./
RUN npm run build

# 3. Verify and stage static files
RUN ls -la dist/ && echo "✅ dist/ exists" \
 && mkdir -p server/public client/dist \
 && cp -r dist/* server/public/ \
 && cp -r dist/* client/dist/ \
 && echo "✅ Static files staged"

# STAGE 2: Final image with Python + Node
FROM python:3.11-slim
WORKDIR /app

# 4. Install system deps + Node.js
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg libpq-dev \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && npm cache clean --force

# 5. Copy built artifacts & source code
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/public ./public
COPY --from=node-builder /app/server/public ./server/public
COPY --from=node-builder /app/client/dist ./client/dist

# Copy your runtime code
COPY migrations ./migrations
COPY server ./server
COPY shared ./shared
COPY index.js ./
COPY package*.json ./

# 6. Python dependencies
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --timeout=180 --retries=3 \
     torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r server/python/requirements.txt \
  && find /usr/local -name '*.pyc' -delete \
  && find /usr/local -name '__pycache__' -delete \
  && pip cache purge

# 7. Copy drizzle-kit for migrations
COPY --from=node-builder /app/node_modules/.bin/drizzle-kit ./node_modules/.bin/drizzle-kit

# 8. Copy & make your startup script executable
COPY start.sh ./
RUN chmod +x start.sh

# 9. Environment & ports
ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHON_PATH=python3
ENV TZ=Asia/Manila
EXPOSE 5000

# 10. Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /app/healthcheck.sh

# 11. Launch
CMD ["sh", "./start.sh"]
