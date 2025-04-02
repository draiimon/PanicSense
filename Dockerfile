# PanicSense Multi-Stage Dockerfile
# Optimized for both local development and production deployment

# -------------------------
# Stage 1: Node.js Builder
# -------------------------
FROM node:20-slim AS node_builder

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# -------------------------
# Stage 2: Python Builder
# -------------------------
FROM python:3.11-slim AS python_builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python requirements
COPY server/python/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Stage 3: Final Image
# -------------------------
FROM node:20-slim

WORKDIR /app

# Install required system packages for Python and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js built files
COPY --from=node_builder /app/dist ./dist
COPY --from=node_builder /app/node_modules ./node_modules
COPY --from=node_builder /app/package*.json ./

# Copy Python environment
COPY --from=python_builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy necessary files for runtime
COPY .env.example ./.env
COPY server/python ./server/python
COPY migrations ./migrations

# Create any necessary directories
RUN mkdir -p uploads/temp

# Expose the application port
EXPOSE 5000

# Define environment variables
ENV NODE_ENV=production
ENV PORT=5000

# Set the command to run the application
CMD ["node", "dist/index.js"]