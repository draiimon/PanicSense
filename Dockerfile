FROM node:20-slim

# Install Python and required tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy package files first to leverage Docker cache
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install

# Copy Python requirements and install them
COPY python-requirements.txt ./
RUN pip3 install -r python-requirements.txt

# Create NLTK data directory and download required NLTK data
RUN mkdir -p /usr/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt vader_lexicon stopwords wordnet

# Copy application code
COPY . ./

# Build the application
RUN pnpm run build

# Expose the port the app runs on
ENV PORT=10000
EXPOSE 10000

# Command to run the application
CMD ["pnpm", "run", "start"]