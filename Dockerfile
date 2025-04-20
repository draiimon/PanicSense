# STAGE 1: Install Node dependencies and build app
FROM node:20.19.0 AS node-builder
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

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

# Install Node.js and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential \
    libpq-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy built files from node-builder
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/node_modules ./node_modules

# Copy package files
COPY package*.json ./

# Copy Python requirements and install
COPY server/python/requirements.txt ./server/python/
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r server/python/requirements.txt

# Copy only needed files for runtime
COPY migrations ./migrations
COPY server ./server
COPY shared ./shared
COPY index.js render-setup.sh ./

# Make setup script executable
RUN chmod +x render-setup.sh

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000
ENV PYTHON_PATH=python3

# Expose the port
EXPOSE 5000

# The CMD is overridden by Render's startCommand in render.yaml
CMD ["npm", "run", "startRender"]