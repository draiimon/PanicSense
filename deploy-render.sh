#!/bin/bash

# PanicSense Render Deployment Script
# This script helps prepare the project for deployment on Render.com

# Set environment to production
export NODE_ENV=production

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the application
echo "🏗️ Building application..."
npm run build

# Prepare for deployment
echo "🚀 Application built successfully!"
echo "Ready to deploy on Render.com"

# Display deployment instructions
echo ""
echo "=== RENDER DEPLOYMENT GUIDE ==="
echo "1. Create a new Web Service on Render"
echo "2. Connect your repository"
echo "3. Use these settings:"
echo "   - Environment: Node.js"
echo "   - Build Command: npm install && npm run build"
echo "   - Start Command: node render-start.js"
echo "4. Add environment variables:"
echo "   - NODE_ENV=production"
echo "   - DATABASE_URL=your_database_connection_string"
echo "   - SESSION_SECRET=your_session_secret"
echo "5. Click 'Create Web Service'"
echo ""
echo "Your application will be deployed at: https://your-service-name.onrender.com"