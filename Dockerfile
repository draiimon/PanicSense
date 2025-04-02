FROM node:20-alpine AS build

# Set working directory
WORKDIR /app

# Install Python and required system packages for canvas and other dependencies
RUN apk add --no-cache python3 py3-pip make g++ pkgconfig pixman-dev cairo-dev pango-dev jpeg-dev giflib-dev

# Copy package files first (better layer caching)
COPY package*.json ./

# Install dependencies with npm (modified to fix canvas build issues)
RUN npm ci --omit=dev

# Copy Python requirements
COPY server/python/requirements.txt ./server/python/

# Install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r server/python/requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Copy source code
COPY . .

# Build the application
RUN NODE_ENV=production npm run build

# Use a smaller production image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install Python and runtime dependencies for canvas
RUN apk add --no-cache python3 pixman cairo pango jpeg giflib

# Copy built app and dependencies from build stage
COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/server/python ./server/python
COPY --from=build /usr/local/share/nltk_data /usr/local/share/nltk_data
COPY --from=build /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages

# Expose the port the app runs on (this handles both HTTP and WebSocket)
EXPOSE 5000

# Define environment variables
ENV NODE_ENV production
ENV PORT 5000
ENV PYTHONPATH=/usr/local/lib/python3.*/site-packages

# Run the application
CMD ["node", "dist/index.js"]