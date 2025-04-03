# PanicSense Optimized Dockerfile for Render Deployment

# Build Stage
FROM node:20-slim AS builder

WORKDIR /build

# Install required system packages for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy and install Python requirements (with minimal dependencies)
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r server/python/requirements.txt

# Download only required NLTK data
RUN pip install --no-cache-dir nltk && \
    python3 -c "import nltk; nltk.download('punkt', download_dir='/build/nltk_data')"

# Copy package files and install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Copy source files
COPY . .

# Build the application
RUN NODE_ENV=production NODE_OPTIONS=--max-old-space-size=1536 npm run build

# Runtime Stage - Use a smaller image
FROM node:20-slim

WORKDIR /app

# Install only runtime dependencies (not dev dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy built application and dependencies
COPY --from=builder /build/dist ./dist
COPY --from=builder /build/node_modules ./node_modules
COPY --from=builder /build/server/python ./server/python
COPY --from=builder /build/venv /app/venv
COPY --from=builder /build/nltk_data /usr/share/nltk_data
COPY --from=builder /build/migrations ./migrations
COPY quick-fix.sh /app/quick-fix.sh

# Set proper permissions for scripts
RUN chmod +x /app/quick-fix.sh

# Configure environment variables
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH=/app/venv/lib/python3.*/site-packages
ENV NLTK_DATA=/usr/share/nltk_data
ENV NODE_ENV=production
ENV PORT=5000
ENV NODE_OPTIONS=--max-old-space-size=1536

# Expose the application port
EXPOSE 5000

# Run the application
CMD ["node", "dist/index.js"]