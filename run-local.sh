#!/bin/bash
# Script to run PanicSense locally with Docker

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Clean up old containers if they exist
echo "Stopping and removing any existing PanicSense containers..."
docker-compose down

# Build and start fresh containers
echo "Building and starting PanicSense..."
docker-compose up --build -d

# Display logs
echo "Displaying logs (press Ctrl+C to exit logs, but the app will continue running)..."
docker-compose logs -f

# When Ctrl+C is pressed, the script will continue here
echo "PanicSense is running at http://localhost:5000"
echo "To stop the application, run: docker-compose down"