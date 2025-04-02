# PanicSense: Disaster Sentiment Analysis Platform

![PanicSense Logo](assets/icons/logo.png)

## Overview

PanicSense is an AI-powered disaster monitoring platform designed for the Philippines. It analyzes text data from social media and other sources to provide critical insights during natural disasters, helping emergency responders and communities with actionable intelligence.

Developed by **Mark Andrei R. Castillo**

## Key Features

- **Real-time Sentiment Analysis**: Analyze public sentiment during disasters
- **Batch CSV Processing**: Upload and analyze large datasets
- **Geospatial Visualization**: View affected regions on interactive maps
- **Disaster Event Tracking**: Monitor different types of disasters
- **Multi-language Support**: Process content in English and Filipino

## Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Shadcn UI
- **Backend**: Node.js, Express, Python
- **Database**: PostgreSQL with Drizzle ORM
- **AI**: Groq API, sentiment analysis models
- **Deployment**: Docker, Render, Replit

## Getting Started

### Requirements
- Node.js 20+
- Python 3.11+
- Docker (optional)

### Option 1: Quick Setup with Docker

```bash
# Clone repository
git clone https://github.com/draiimon/PanicSense.git
cd PanicSense

# Create .env file (copy from example)
cp .env.example .env
# Add your Groq API key to the .env file

# Build and start with Docker
docker-compose up --build
```

### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/draiimon/PanicSense.git
cd PanicSense

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r server/python/requirements.txt

# Set up environment variables
cp .env.example .env
# Add your Groq API key to the .env file

# Start the application
npm run dev
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY_1` | Groq API key for sentiment analysis | ✅ Yes |
| `DATABASE_URL` | PostgreSQL connection string | ✅ Yes |

## Deployment

### Deploy to Render

1. Fork the repository to your GitHub account
2. Sign up or log in to [Render](https://render.com)
3. Create a new Web Service and connect to your GitHub repo
4. Add your environment variables (GROQ_API_KEY_1)
5. Deploy the application

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/draiimon/PanicSense)

### Run on Replit

1. Visit [replit.com](https://replit.com)
2. Import from GitHub: github.com/draiimon/PanicSense
3. Add your Groq API key in the Secrets tab
4. Click Run to start the application

[![Run on Replit](https://replit.com/badge/github/draiimon/PanicSense)](https://replit.com/github/draiimon/PanicSense)

## Project Structure

```
PanicSense/
├── client/                  # Frontend React application
├── server/                  # Backend Express server
│   └── python/              # Python sentiment analysis
├── shared/                  # Shared code and types
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # Project documentation
```

## License

© 2025 Mark Andrei R. Castillo. All Rights Reserved.

**Made with ❤️ in the Philippines**