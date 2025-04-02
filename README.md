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

For detailed step-by-step instructions, see [LOCAL_SETUP.md](LOCAL_SETUP.md)

1. **Clone the repository**
   ```bash
   git clone https://github.com/draiimon/PanicSense.git
   cd PanicSense
   ```

2. **Option 1: Using pnpm (without Docker)**
   ```bash
   # Install Node.js dependencies
   pnpm install

   # Create and activate Python virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install Python dependencies
   pip install -r server/python/requirements.txt

   # Set up PostgreSQL database
   # Create a PostgreSQL database and update DATABASE_URL in .env

   # Run database migrations
   npm run db:push

   # Start the development server
   npm run dev
   ```

3. **Option 2: Using Docker (Recommended)**
   ```bash
   # Build and run with Docker
   docker-compose up --build
   ```

   This will start the Node.js server, Python service, and PostgreSQL database in separate containers.

### Environment Setup

1. **Create a .env file**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Make sure the DATABASE_URL in your .env file is set to:
   ```
   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/postgres
   ```
   
   Add your Groq API keys to the .env file:
   ```
   # For validation (used for sentiment feedback)
   VALIDATION_API_KEY=your_groq_api_key_here
   
   # For regular API calls with rotation
   GROQ_API_KEY_1=your_first_groq_api_key
   GROQ_API_KEY_2=your_second_groq_api_key
   GROQ_API_KEY_3=your_third_groq_api_key
   GROQ_API_KEY_4=your_fourth_groq_api_key
   ```
   
   You can add as many GROQ_API_KEY_N environment variables as needed. The system will automatically detect and use all available keys for rotation to prevent rate limiting.

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   This will start the Node.js server, Python service, and PostgreSQL database in separate containers.

3. **Run database migrations (first time only)**
   ```bash
   # In a new terminal window
   docker exec disaster-monitoring-app npm run db:push
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Deployment to Render

1. **Fork/Push this repository to your own GitHub account**

2. **Sign up for Render**
   - Create an account at [render.com](https://render.com)
   - Connect your GitHub account

3. **Deploy with Render Blueprint**
   - In the Render dashboard, click "New" and select "Blueprint"
   - Select the repository containing this project
   - Render will automatically detect the `render.yaml` file which uses `Dockerfile.new` for deployment

4. **Environment Variables**
   - The database connection string will be automatically configured
   - **Important**: You must add your Groq API keys in the Render dashboard:
     - Go to your web service in the Render dashboard
     - Click on "Environment" tab
     - Add the following environment variables:
       - `VALIDATION_API_KEY` - Single API key used for sentiment validation
       - `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc. - Multiple API keys for rotation
   - Using multiple API keys helps prevent rate limiting when processing large volumes of data

5. **Access Your Deployed Application**
   - Once deployment is complete, Render will provide a URL to access your application

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
│   ├── public/              # Static assets
│   └── src/
│       ├── components/      # UI components
│       ├── context/         # Context providers
│       ├── hooks/           # Custom React hooks
│       ├── lib/             # Utility functions
│       └── pages/           # Page components
├── server/                  # Backend Express server
│   ├── python/              # Python NLP processing scripts
│   │   ├── process.py       # Python processing script
│   │   └── requirements.txt # Python dependencies
│   ├── routes.ts            # API routes
│   ├── storage.ts           # Database interactions
│   ├── python-service.ts    # Python service integration
│   ├── db.ts                # Database connection
│   └── utils/               # Utility functions
├── shared/                  # Shared code between frontend and backend
│   └── schema.ts            # Database schema and types
├── migrations/              # Database migrations
├── .dockerignore            # Files to exclude from Docker image
├── .env                     # Environment variables (gitignored)
├── .env.example             # Example environment variables
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── render.yaml              # Render deployment configuration
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