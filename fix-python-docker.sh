#!/bin/bash

# This script will update the necessary files and commit them to fix Docker build issues
# with Python virtual environment configuration

# Checkout main branch
git checkout main

# Update Dockerfile with virtual environment support
cat > Dockerfile << 'EOF'
# PanicSense Simple Dockerfile

FROM node:20-slim

WORKDIR /app

# Install required system packages for Python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files and install Node.js dependencies
COPY package*.json ./
RUN npm ci

# Set up Python virtual environment and install Python requirements
COPY server/python/requirements.txt ./
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads/temp

# Expose the application port
EXPOSE 5000

# Define environment variables
ENV PORT=5000

# Set NODE_ENV based on argument with development as default
ARG NODE_ENV=development
ENV NODE_ENV=${NODE_ENV}

# Use different command for development vs production
# For production (on Render), we'll build and then run from dist/
# For development (locally), we'll use the dev script
CMD if [ "$NODE_ENV" = "production" ]; then \
        npm run build && \
        node dist/index.js; \
    else \
        npm run dev; \
    fi
EOF

# Update python-service.ts to detect Docker environment and use virtual env
sed -i 's/this.pythonBinary = '"'"'python3'"'"';/  \/\/ Check if we'"'"'re running in Docker environment (where we use venv)\n    const isDocker = process.env.NODE_ENV === '"'"'production'"'"';\n    this.pythonBinary = isDocker ? '"'"'\/opt\/venv\/bin\/python3'"'"' : '"'"'python3'"'"';/' server/python-service.ts

# Commit and push changes
git add Dockerfile server/python-service.ts
git commit -m "Fix Docker build errors with Python virtual environment"
git push origin main

echo "Changes committed and pushed to main branch"