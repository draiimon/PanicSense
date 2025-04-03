## Stage 1: Build stage with full dependencies
FROM node:20-slim as builder

WORKDIR /build

# Install system dependencies first (Python and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /build/venv

# Copy Python requirements and install them
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r server/python/requirements.txt

# Install NLTK data
RUN pip install --no-cache-dir nltk && \
    python3 -c "import nltk; nltk.download('punkt', download_dir='/build/nltk_data')"

# Copy package files first (for better caching)
COPY package*.json ./
RUN npm ci

# Copy application code
COPY . .

# Build the application (both client and server)
RUN NODE_ENV=production NODE_OPTIONS=--max-old-space-size=1536 npm run build

## Stage 2: Production stage (smaller image)
FROM node:20-slim

WORKDIR /app

# Install only the runtime dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from build stage
COPY --from=builder /build/dist ./dist
COPY --from=builder /build/node_modules ./node_modules
COPY --from=builder /build/server/python ./server/python
COPY --from=builder /build/venv /app/venv
COPY --from=builder /build/nltk_data /usr/share/nltk_data
COPY --from=builder /build/migrations ./migrations
COPY --from=builder /build/server/db-simple-fix.js ./server/db-simple-fix.js
COPY --from=builder /build/server/emergency-db-fix.js ./server/emergency-db-fix.js

# Copy database fix scripts
COPY quick-fix.sh /app/quick-fix.sh
RUN chmod +x /app/quick-fix.sh

# Set environment variables
ENV NODE_ENV=production
ENV PATH="/app/venv/bin:$PATH"
ENV NLTK_DATA="/usr/share/nltk_data"
ENV PYTHONPATH="/app/server/python"

# Expose ports
EXPOSE 5000

# Apply database fixes and start the application
CMD ["/bin/bash", "-c", "/app/quick-fix.sh && node dist/index.js"]