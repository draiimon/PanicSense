FROM node:20-alpine as builder

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install Python and required packages
RUN apk add --no-cache python3 py3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install langdetect nltk numpy pandas pytz requests scikit-learn torch tqdm

# Copy built assets from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/python ./python

# Install production dependencies only
RUN npm ci --only=production

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable
ENV NODE_ENV production
ENV PORT 5000

# Run the application
CMD ["node", "dist/index.js"]