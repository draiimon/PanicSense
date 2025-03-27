FROM node:20

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Expose the port the app runs on
EXPOSE 5000

# Environment variables
ENV PORT=5000
ENV NODE_ENV=production

# Command to run the application
CMD ["node", "dist/index.js"]