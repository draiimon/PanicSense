# Start with the official Python image that has Python.h included
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Install Node.js using the official script
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    make \
    python3-dev \
    libopenblas-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify versions
RUN node --version && python --version

# Install pnpm
RUN npm install -g pnpm@latest

# Copy package.json and package-lock.json for better caching
COPY package.json package-lock.json ./
# Force install nanoid to fix missing dependency issue
RUN pnpm install
RUN npm install nanoid

# Install prebuilt Python packages first to avoid compilation issues
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.26.4 && \
    pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Use a prebuilt wheel for scikit-learn
    pip install --only-binary=scikit-learn scikit-learn==1.3.2 && \
    pip install tqdm

# Copy Python requirements file
COPY server/python/requirements.txt server/python/requirements-lock.txt ./server/python/

# Install remaining Python dependencies with locked versions to prevent incompatibilities
RUN pip install -r server/python/requirements-lock.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application
COPY . .

# Build the application
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN NODE_ENV=production pnpm run build

# Expose port
EXPOSE 5000

# Set production environment
ENV NODE_ENV=production \
    PORT=5000

# Create a startup script that runs migrations before starting the app
RUN echo '#!/bin/bash\nnode /app/migrations/run-migrations.js\nnode /app/dist/index.js' > /app/start.sh && \
    chmod +x /app/start.sh

# Start the application with migration script
CMD ["/app/start.sh"]