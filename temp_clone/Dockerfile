# Base image with multiple stages for optimization
FROM node:20-slim AS base
WORKDIR /app

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Build dependencies stage
FROM base AS deps
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install Node.js dependencies
RUN npm ci
RUN npm install -g ts-node

# Copy Python requirements
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; else pip install pandas numpy langdetect requests scikit-learn torch nltk openai pytz snscrape beautifulsoup4 tqdm; fi

# Build stage
FROM base AS builder
WORKDIR /app

# Copy dependencies
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/venv ./venv

# Copy source files
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM base AS runner
WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

# Copy from builder stage
COPY --from=builder /app/venv ./venv
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY --from=builder /app/python ./python

# Install ts-node for TypeScript execution
RUN npm install -g ts-node

# Copy necessary files for runtime
COPY package.json ./
COPY server ./server

# Expose port 5000 as the application uses this port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:5000/api/health || exit 1

# Set the entrypoint script
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command
CMD ["node", "--experimental-specifier-resolution=node", "--loader", "ts-node/esm", "server/index.ts"]