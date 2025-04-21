FROM node:20-slim

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    curl \
    procps \
    wget \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install app dependencies with explicit production
RUN npm ci --only=production

# Install dev dependencies separately to ensure build tools are available
RUN npm ci --only=development

# Create Python virtual environment and install Python dependencies
COPY pyproject.toml .
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas numpy scikit-learn nltk torch beautifulsoup4 langdetect python-dotenv pytz requests tqdm snscrape openai

# Install NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy app source
COPY . .

# Build the application
RUN npm run build

# Set environment variables
ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=10000
ENV TZ=Asia/Manila
ENV PYTHON_PATH=python3
ENV PYTHON_SERVICE_ENABLED=true
ENV RUNTIME_ENV=render
ENV NODE_TLS_REJECT_UNAUTHORIZED=0

# Expose the port the app runs on - use Render's PORT variable
EXPOSE $PORT

# Create a directory for uploads and ensure proper permissions
RUN mkdir -p /app/uploads/data /app/uploads/profile_images /app/uploads/temp
RUN chmod -R 777 /app/uploads

# Copy and set execute permission for startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Add a basic healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application using the startup script
CMD ["/app/start.sh"]