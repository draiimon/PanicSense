FROM node:20-slim

WORKDIR /app

# Install Python and its dependencies for the AI processing
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm ci

# Copy application files
COPY . .

# Install Python dependencies
RUN pip3 install pandas nltk scikit-learn torch langdetect tqdm python-dotenv pytz

# Build the application
RUN npm run build

# Set environment variables
ENV NODE_ENV=production
ENV PORT=10000

# Verify build output
RUN if [ -d "dist/public" ]; then \
      echo "✅ Static files directory exists"; \
      ls -la dist/public; \
    else \
      echo "❌ ERROR: dist/public directory not found!"; \
      ls -la; \
    fi

# Expose the port
EXPOSE 10000

# Start the application
CMD ["node", "server-deploy.js"]