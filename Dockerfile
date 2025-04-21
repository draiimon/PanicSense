# Multi-stage build for PanicSense
# Stage 1: Base image with dependencies
FROM node:20-slim AS base

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY pyproject.toml ./

# Stage 2: Development dependencies
FROM base AS dependencies
# Install Node.js dependencies with development packages
RUN npm ci

# Set up Python environment and install dependencies
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir -e .

# Stage 3: Production build
FROM base AS builder
# Copy dependencies from the dependencies stage
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=dependencies /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Stage 4: Production runtime
FROM base AS runtime
ENV NODE_ENV=production
ENV PATH="/venv/bin:$PATH"

# Copy Python environment from dependencies stage
COPY --from=dependencies /venv /venv

# Copy built assets from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/server ./server
COPY --from=builder /app/python ./python
COPY --from=builder /app/shared ./shared
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/tsconfig.json ./
COPY --from=builder /app/server.js ./
COPY --from=builder /app/start.sh ./
COPY --from=builder /app/verify-db.js ./
COPY --from=builder /app/drizzle.config.ts ./

# Copy specific configuration files
COPY --from=builder /app/.env.example ./.env.example

# Make start script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 5000

# Set container startup command
CMD ["npm", "run", "dev"]