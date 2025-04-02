# PanicSense Simple Dockerfile

FROM node:20-slim

WORKDIR /app

# Install required system packages for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    && rm -rf /var/lib/apt/lists/*

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Set up Python virtual environment and install requirements
COPY server/python/requirements.txt ./
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

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