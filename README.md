# PanicSense: Disaster Monitoring System

## Overview
PanicSense is an advanced AI-powered disaster monitoring and community resilience platform for the Philippines. It leverages cutting-edge natural language processing technology to analyze news content, detect disaster-related information, and provide real-time alerts about emergency situations.

## Features
- **Real-time Disaster News Monitoring**: Automatically collects and analyzes news from multiple Philippine sources
- **Intelligent Keyword Filtering**: Identifies disaster events using advanced pattern matching in both English and Filipino
- **Interactive Dashboard**: Visualizes disaster data, sentiment analysis, and geographic information
- **CSV Data Analysis**: Process and analyze disaster-related data from uploaded CSV files
- **Mobile-Responsive Design**: Works seamlessly across desktop and mobile devices
- **Dual-language Support**: Full support for both English and Filipino disaster terminology

## Local Development Setup

### Prerequisites
- Node.js (v18+)
- PostgreSQL database
- Git

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd panicsense
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Make sure you have PostgreSQL running and create a database:
```bash
createdb panicsense
```

5. Push the database schema:
```bash
npm run db:push
```

6. Start the development server:
```bash
npm run dev
```

The application will be available at http://localhost:5000

## Performance Enhancements

In our latest update, we've significantly improved performance by:

1. **Keyword-First Processing**: News items are first filtered using an extensive keyword matching system before any AI validation occurs
2. **Improved Loading States**: Better loading placeholders to prevent showing "No Active Disaster Alerts" during data fetching
3. **Rate Limit Protection**: Intelligent request throttling to prevent API rate limits

## Database Schema

PanicSense uses a PostgreSQL database with the following main tables:
- `sentiment_posts`: Stores analyzed text content with sentiment data
- `disaster_events`: Tracks detected disaster events
- `analyzed_files`: Maintains records of processed CSV files
- `users`: Stores user account information

## Configuration

Configuration is managed via environment variables. See `.env.example` for available options:

- `DATABASE_URL`: PostgreSQL connection string
- `GROQ_API_KEY`: API key for Groq (used for enhanced AI analysis)
- `NEWS_REFRESH_INTERVAL`: How often to fetch news (in minutes)
- `DISASTER_KEYWORD_THRESHOLD`: Confidence threshold for keyword matching
- `ENABLE_KEYWORD_FILTERING`: Toggle keyword-based filtering on/off
- `ENABLE_AI_VALIDATION`: Toggle AI validation on/off

## License
This project is proprietary software. All rights reserved.

## Acknowledgements
- Built with Node.js, React, and TypeScript
- Uses PostgreSQL with Drizzle ORM
- Incorporates advanced NLP algorithms
- Optimized for Philippine disaster monitoring