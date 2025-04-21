# STAGE 1: Install Node dependencies and build app
FROM node:20.19.0-slim AS node-builder
WORKDIR /app

# Copy package files and install dependencies (with production flag for lighter install)
COPY package*.json ./
RUN npm ci --production=false

# Copy source files and build
COPY client ./client
COPY server ./server
COPY shared ./shared
COPY public ./public
COPY drizzle.config.ts tailwind.config.ts tsconfig.json vite.config.ts theme.json ./

# Build the application
RUN npm run build

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
COPY --from=node-builder /app/node_modules ./node_modules

# Copy package files (only production files)
COPY package*.json ./

# Copy Python requirements and install
# Combined pip commands to reduce layers and optimize caching
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r server/python/requirements.txt \
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

# Add health check for better container stability
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Expose the port
EXPOSE 5000

# Simple startup command
CMD ["node", "index.js"]