FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install Python and required packages
RUN apk add --no-cache python3 py3-pip

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install langdetect nltk numpy pandas pytz requests scikit-learn torch tqdm

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Build the application
RUN npm run build

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables
ENV NODE_ENV production
ENV PORT 5000

# Run the application
CMD ["node", "dist/index.js"]