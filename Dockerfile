FROM node:20-slim

# Set working directory
WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package.json and package-lock.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci

# Copy Python requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Build the frontend
RUN npm run build

# Set DATABASE_URL environment variable (this will be overridden by docker-compose)
ENV DATABASE_URL=postgres://postgres:postgres@db:5432/postgres
ENV NODE_ENV=production
ENV PORT=3000

# Expose the port
EXPOSE 3000

# CMD to start the application
CMD ["npm", "run", "start"]