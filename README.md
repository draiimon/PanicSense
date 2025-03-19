# Disaster Sentiment Analysis Platform

## Overview

The Disaster Sentiment Analysis Platform is an AI-powered tool designed to provide real-time emotional insights during natural calamities in the Philippines. The platform analyzes social media posts, news articles, and other text data to understand public sentiment during disasters, helping emergency responders and government agencies make more informed decisions.

## Features

- **Real-time Sentiment Analysis**: Analyze public sentiment from various data sources during disasters
- **Geospatial Visualization**: View sentiment data on interactive maps showing affected regions
- **Disaster Event Tracking**: Monitor and track different types of disasters (floods, typhoons, etc.)
- **Performance Metrics**: Evaluate the accuracy of sentiment analysis through comprehensive metrics
- **Multi-language Support**: Process content in different languages, with a focus on Filipino dialects
- **User Authentication**: Secure login system for authorized access to the platform

## Tech Stack

### Frontend
- **React**: UI library for building the user interface
- **TypeScript**: Type-safe JavaScript for improved developer experience
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Shadcn UI**: Component library built on Radix UI primitives
- **React Query**: Data fetching and state management library
- **Chart.js & Recharts**: Libraries for data visualization
- **Leaflet**: Interactive mapping library for geographical visualizations

### Backend
- **Node.js with Express**: Server framework for API endpoints
- **Python**: For advanced NLP and sentiment analysis processing
- **PostgreSQL**: Database for storing sentiment data and user information
- **Drizzle ORM**: Type-safe database toolkit for TypeScript
- **WebSockets**: For real-time data updates
- **Passport.js**: Authentication middleware

### Machine Learning & NLP
- **Langdetect**: Language detection library
- **Pandas & NumPy**: Data processing libraries
- **scikit-learn**: Machine learning tools for evaluation metrics

### DevOps
- **Docker**: Containerization for consistent deployment
- **Docker Compose**: Multi-container application orchestration
- **GitHub**: Version control and source code management

## Installation & Setup

### Prerequisites
- Node.js (v16+)
- Python (v3.11+)
- PostgreSQL
- Docker and Docker Compose (for containerized setup)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/disaster-sentiment-analysis.git
   cd disaster-sentiment-analysis
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   pip install pandas numpy langdetect scikit-learn
   ```

4. **Set up PostgreSQL database**
   - Create a PostgreSQL database
   - Set the `DATABASE_URL` environment variable to your database connection string

5. **Run database migrations**
   ```bash
   npm run db:push
   ```

6. **Start the development server**
   ```bash
   npm run dev
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   This will start the Node.js server, Python service, and PostgreSQL database in separate containers.

2. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Usage Guide

### Uploading Data
1. Navigate to the Dashboard
2. Click on "Upload CSV" to upload disaster-related text data
3. The system will process the data and provide sentiment analysis results

### Real-time Analysis
1. Go to the "Real-time" section
2. Enter text in the input field to analyze sentiment immediately
3. View the sentiment classification and confidence score

### Viewing Geographic Impact
1. Visit the "Geographic Analysis" page
2. Explore the interactive map showing sentiment distribution by region
3. Filter by disaster type or date range to focus on specific events

### Performance Evaluation
1. Access the "Evaluation" section to view model performance metrics
2. Check accuracy, precision, recall, and F1 scores
3. Review the confusion matrix for detailed error analysis

## Project Structure

```
disaster-sentiment-analysis/
├── client/                  # Frontend React application
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── context/         # Context providers
│   │   ├── hooks/           # Custom React hooks
│   │   ├── lib/             # Utility functions
│   │   └── pages/           # Page components
├── server/                  # Backend Express server
│   ├── python/              # Python NLP processing scripts
│   ├── routes.ts            # API routes
│   ├── storage.ts           # Database interactions
│   └── utils/               # Utility functions
├── shared/                  # Shared code between frontend and backend
│   └── schema.ts            # Database schema and types
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.