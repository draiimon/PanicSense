# Base image with Python 3.11 and Node.js pre-installed
FROM nikolaik/python-nodejs:python3.11-nodejs20-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm globally
RUN npm install -g pnpm@10.7.1

# Copy package.json and install Node.js dependencies
COPY package.json ./
RUN pnpm install --frozen-lockfile

# Copy Python requirements
COPY server/python/requirements.txt ./server/python/

# Install Python dependencies strategically to avoid compilation issues
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    # Install numpy first (needed for scikit-learn)
    pip install numpy==1.26.4 && \
    # Install PyTorch with CPU only to reduce image size
    pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Install scikit-learn separately with a version known to work
    pip install scikit-learn==1.3.2 && \
    # Install remaining packages
    pip install -r server/python/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application
COPY . .

# Build the application
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN NODE_ENV=production pnpm run build

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables for production
ENV NODE_ENV=production \
    PORT=5000

# Start the application
CMD ["node", "dist/index.js"]