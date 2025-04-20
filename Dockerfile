FROM node:20.19.0-bullseye-slim

# Install Python 3.11 and required dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    pkg-config \
    libpq-dev \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci

# Set up Python virtual environment
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy Python requirements
COPY server/python/requirements.txt ./server/python/

# Install PyTorch CPU version specifically
RUN pip3 install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r server/python/requirements.txt

# Copy application files
COPY . .

# Build the application
RUN npm run build

# Perform database migrations
RUN npm run db:push

# Expose the port the app runs on
EXPOSE 5000

# Start the application
CMD ["npm", "run", "startRender"]