#!/bin/bash

# This script helps verify locally that your Docker setup works correctly before pushing to Render
# Run this script after installing Docker on your machine

echo "=============================================="
echo "Docker Verification Script for PanicSense PH"
echo "$(date)"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

echo "Docker is installed: $(docker --version)"

# Build the Docker image
echo -e "\n1. Building Docker image..."
docker build -t panicsense-image .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed."
    exit 1
fi
echo "✓ Docker image built successfully!"

# Create a test network
echo -e "\n2. Creating test network..."
docker network create panicsense-test-network 2>/dev/null || true

# Run a test PostgreSQL container for local testing
echo -e "\n3. Starting a temporary PostgreSQL container..."
docker run --rm -d \
    --name postgres-test \
    --network panicsense-test-network \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=panicsense \
    postgres:15

echo "Waiting for PostgreSQL to start..."
sleep 5

# Run the container in test mode
echo -e "\n4. Starting a temporary PanicSense container..."
docker run --rm -d \
    --name panicsense-test \
    --network panicsense-test-network \
    -p 5000:5000 \
    -e DATABASE_URL="postgres://postgres:postgres@postgres-test:5432/panicsense" \
    -e NODE_ENV=production \
    -e PORT=5000 \
    -e PYTHON_PATH=python3 \
    -e SESSION_SECRET=test-secret \
    panicsense-image

echo "Waiting for PanicSense to start..."
sleep 10

# Check if the container is running
container_status=$(docker inspect -f {{.State.Status}} panicsense-test 2>/dev/null || echo "not_found")
if [ "$container_status" != "running" ]; then
    echo "ERROR: PanicSense container is not running. Check the logs below:"
    docker logs panicsense-test
    echo -e "\nCleaning up..."
    docker rm -f postgres-test panicsense-test 2>/dev/null || true
    docker network rm panicsense-test-network 2>/dev/null || true
    exit 1
fi

# Test the API
echo -e "\n5. Testing API health endpoint..."
response=$(curl -s http://localhost:5000/api/health)
if [[ $response == *"ok"* ]]; then
    echo "✓ API health check successful!"
else
    echo "ERROR: API health check failed. Response: $response"
    echo "Container logs:"
    docker logs panicsense-test
    echo -e "\nCleaning up..."
    docker rm -f postgres-test panicsense-test 2>/dev/null || true
    docker network rm panicsense-test-network 2>/dev/null || true
    exit 1
fi

# Clean up
echo -e "\n6. Cleaning up test containers..."
docker rm -f postgres-test panicsense-test 2>/dev/null || true
docker network rm panicsense-test-network 2>/dev/null || true

echo -e "\n=============================================="
echo "✓ Verification completed successfully!"
echo "Your Docker setup is ready for Render deployment."
echo "=============================================="
echo -e "\nNext steps:"
echo "1. Push your code to GitHub/GitLab"
echo "2. Connect your repository to Render"
echo "3. Render will automatically detect your configuration and deploy the application"
echo -e "\nRefer to RENDER_DEPLOYMENT.md for detailed deployment instructions."