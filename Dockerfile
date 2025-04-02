FROM node:20-alpine AS build

# Set working directory
WORKDIR /app

# Install pnpm with specific version to avoid unexpected breaks
RUN npm install -g pnpm@10.7.1

# Install Python and required system packages for canvas and other dependencies
# Grouped into one layer to reduce image size
RUN apk add --no-cache \
    python3 \
    py3-pip \
    make \
    g++ \
    pkgconfig \
    pixman-dev \
    cairo-dev \
    pango-dev \
    jpeg-dev \
    giflib-dev

# Copy package files first (better layer caching)
COPY package.json ./

# Install dependencies with pnpm (consolidated for fewer layers)
RUN pnpm config set ignore-pnpmfile true && \
    pnpm config set auto-install-peers true && \
    pnpm install --prod --no-optional && \
    pnpm install -D esbuild

# Set up Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy Python requirements
COPY server/python/requirements.txt ./server/python/

# Install Python packages in the virtual environment
# Increased timeout for pip to avoid build failures
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip --default-timeout=100 install -r server/python/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"

# Copy source code
COPY . .

# Build the application with NODE_OPTIONS to avoid memory issues
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN NODE_ENV=production pnpm run build

# Use a smaller production image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install pnpm with specific version
RUN npm install -g pnpm@10.7.1

# Install Python and runtime dependencies for canvas - grouped to reduce layers
RUN apk add --no-cache \
    python3 \
    python3-dev \
    pixman \
    cairo \
    pango \
    jpeg \
    giflib

# Copy built app and dependencies from build stage
COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/server/python ./server/python
COPY --from=build /app/venv /app/venv
COPY --from=build /app/nltk_data /usr/local/share/nltk_data

# Set environment for Python virtual environment
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH=/app/venv/lib/python3.*/site-packages
ENV NLTK_DATA=/usr/local/share/nltk_data

# Expose the port the app runs on (this handles both HTTP and WebSocket)
EXPOSE 5000

# Define environment variables
ENV NODE_ENV production
ENV PORT 5000

# Run the application
CMD ["node", "dist/index.js"]