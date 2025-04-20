FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm ci

# Copy application files
COPY . .

# Build the application
RUN npm run build

# Expose the port
EXPOSE 8080

# Start the application
CMD ["node", "server-deploy.js"]