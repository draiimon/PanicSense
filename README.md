# PanicSense: Disaster Monitoring System

## Overview
PanicSense is an advanced AI-powered disaster monitoring and community resilience platform for the Philippines. It leverages cutting-edge natural language processing technology to analyze news content, detect disaster-related information, and provide real-time alerts about emergency situations.

## Features
- **Real-time Disaster News Monitoring**: Automatically collects and analyzes news from multiple Philippine sources
- **Keyword-based Disaster Detection**: Identifies disaster events using advanced keyword matching in both English and Filipino
- **Interactive Dashboard**: Visualizes disaster data, sentiment analysis, and geographic information
- **CSV Data Analysis**: Process and analyze disaster-related data from uploaded CSV files
- **Mobile-Responsive Design**: Works seamlessly across desktop and mobile devices

## Local Development Setup

### Prerequisites
- Docker and Docker Compose
- Git

### Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd panicsense
```

2. Make the run scripts executable:
```bash
chmod +x run-local.sh
chmod +x start.sh
```

3. Start the application using Docker:
```bash
./run-local.sh
```

The application will be available at http://localhost:5000

### Manual Setup (without Docker)

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
```

3. Make sure you have PostgreSQL running and create a database:
```bash
createdb panicsense
```

4. Push the database schema:
```bash
npm run db:push
```

5. Start the development server:
```bash
npm run dev
```

## Database Schema

PanicSense uses a PostgreSQL database with the following main tables:
- `sentiment_posts`: Stores analyzed text content with sentiment data
- `disaster_events`: Tracks detected disaster events
- `analyzed_files`: Maintains records of processed CSV files
- `users`: Stores user account information

## Configuration

Configuration is managed via environment variables. See `.env.example` for available options:

- `DATABASE_URL`: PostgreSQL connection string
- `GROQ_API_KEY`: API key for Groq (optional, used for enhanced AI analysis)
- `NEWS_REFRESH_INTERVAL`: How often to fetch news (in minutes)
- `DISASTER_KEYWORD_THRESHOLD`: Confidence threshold for keyword matching

## License
This project is proprietary software. All rights reserved.

## Acknowledgements
- Built with Node.js, React, and TypeScript
- Uses PostgreSQL with Drizzle ORM
- Incorporates advanced NLP algorithms
- Optimized for Philippine disaster monitoring