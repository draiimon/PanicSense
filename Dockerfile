FROM node:20-slim

# Install Python and required tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package files first to leverage Docker cache
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Setup Python virtual environment and install requirements
COPY python-requirements.txt ./
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r python-requirements.txt

# Create NLTK data directory and download required NLTK data
RUN mkdir -p /usr/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt vader_lexicon stopwords wordnet

# Copy application code
COPY . ./

# Create necessary directories
RUN mkdir -p uploads/temp attached_assets

# Run database migrations
RUN node migrations/run-migrations.js

# Build the application
RUN npm run build

# Expose the port the app runs on
ENV PORT=10000
EXPOSE 10000

# Set production environment
ENV NODE_ENV=production

# Command to run the application
CMD ["node", "dist/index.js"]