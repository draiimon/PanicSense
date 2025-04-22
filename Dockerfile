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
ENV VERBOSE_LOGGING=true

# Expose the port the app runs on - use Render's PORT variable
EXPOSE $PORT

# Create a directory for uploads and ensure proper permissions
RUN mkdir -p /app/uploads/data /app/uploads/profile_images /app/uploads/temp
RUN chmod -R 777 /app/uploads

# Create special fix scripts for Render database issues
RUN echo '#!/bin/bash\n\
echo "🔧 Running Render-specific database fixes..."\n\
\n\
# Function to fix timestamp columns in tables\n\
fix_timestamp_columns() {\n\
  local table_name=$1\n\
  echo "🔧 Checking $table_name table structure..."\n\
  psql $DATABASE_URL -c "ALTER TABLE $table_name ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"\n\
  psql $DATABASE_URL -c "UPDATE $table_name SET timestamp = created_at WHERE timestamp IS NULL AND created_at IS NOT NULL;"\n\
  echo "✅ Fixed timestamp column in $table_name"\n\
}\n\
\n\
# Connect to database and run fixes\n\
if [ ! -z "$DATABASE_URL" ]; then\n\
  echo "🔄 Connecting to PostgreSQL database..."\n\
  psql $DATABASE_URL -c "SELECT NOW() as server_time;"\n\
  \n\
  # Fix disaster_events table\n\
  fix_timestamp_columns "disaster_events"\n\
  \n\
  # Fix analyzed_files table\n\
  fix_timestamp_columns "analyzed_files"\n\
  \n\
  # Create missing upload_sessions table if needed\n\
  echo "🔧 Creating upload_sessions table if needed..."\n\
  psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS upload_sessions (\n\
    id SERIAL PRIMARY KEY,\n\
    session_id TEXT UNIQUE NOT NULL,\n\
    file_id INTEGER,\n\
    status TEXT DEFAULT '\''active'\'',\n\
    progress JSONB DEFAULT '\''{}'\'' );"
else\n\
  echo "❌ DATABASE_URL not set, skipping database fixes"\n\
fi\n\
' > /app/render-db-fix.sh && chmod +x /app/render-db-fix.sh

# Copy and set execute permission for startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Add a basic healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application using the startup script
CMD ["/app/start.sh"]FROM node:20-slim

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
ENV VERBOSE_LOGGING=true

# Expose the port the app runs on - use Render's PORT variable
EXPOSE $PORT

# Create a directory for uploads and ensure proper permissions
RUN mkdir -p /app/uploads/data /app/uploads/profile_images /app/uploads/temp
RUN chmod -R 777 /app/uploads

# Create special fix scripts for Render database issues
RUN echo '#!/bin/bash\n\
echo "🔧 Running Render-specific database fixes..."\n\
\n\
# Function to fix timestamp columns in tables\n\
fix_timestamp_columns() {\n\
  local table_name=$1\n\
  echo "🔧 Checking $table_name table structure..."\n\
  psql $DATABASE_URL -c "ALTER TABLE $table_name ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"\n\
  psql $DATABASE_URL -c "UPDATE $table_name SET timestamp = created_at WHERE timestamp IS NULL AND created_at IS NOT NULL;"\n\
  echo "✅ Fixed timestamp column in $table_name"\n\
}\n\
\n\
# Connect to database and run fixes\n\
if [ ! -z "$DATABASE_URL" ]; then\n\
  echo "🔄 Connecting to PostgreSQL database..."\n\
  psql $DATABASE_URL -c "SELECT NOW() as server_time;"\n\
  \n\
  # Fix disaster_events table\n\
  fix_timestamp_columns "disaster_events"\n\
  \n\
  # Fix analyzed_files table\n\
  fix_timestamp_columns "analyzed_files"\n\
  \n\
  # Create missing upload_sessions table if needed\n\
  echo "🔧 Creating upload_sessions table if needed..."\n\
  psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS upload_sessions (\n\
    id SERIAL PRIMARY KEY,\n\
    session_id TEXT UNIQUE NOT NULL,\n\
    file_id INTEGER,\n\
    status TEXT DEFAULT '\''active'\'',\n\
    progress JSONB DEFAULT '\''{}'\'' );"
else\n\
  echo "❌ DATABASE_URL not set, skipping database fixes"\n\
fi\n\
' > /app/render-db-fix.sh && chmod +x /app/render-db-fix.sh

# Copy and set execute permission for startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Add a basic healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application using the startup script
CMD ["/app/start.sh"]
