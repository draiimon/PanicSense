# Disaster Monitoring Platform

An advanced AI-powered disaster monitoring and community resilience platform for the Philippines, leveraging cutting-edge technology to enhance emergency response and community preparedness.

## Running on Replit

The project is already configured to run on Replit. Simply click the "Run" button in the Replit interface to start the server. The application will be available at the provided Replit URL.

## Deployment on Render

This application is configured for deployment on Render.com with PostgreSQL database integration.

### Deployment Steps

1. Push this repository to GitHub
2. Log in to [Render](https://render.com)
3. Click "New +" and select "Blueprint"
4. Connect to your GitHub repository
5. Render will automatically detect the `render.yaml` file and set up the services

### Manual Deployment (Alternative)

If you prefer to set up services manually:

1. **Create a PostgreSQL Database:**
   - In Render dashboard, select "New +" > "PostgreSQL"
   - Configure your database (name: `disasterdb`, user: `disastermonitor`)
   - Note the internal connection string

2. **Create a Web Service:**
   - In Render dashboard, select "New +" > "Web Service"
   - Connect to your GitHub repository
   - Configure the service:
     - **Name:** disaster-monitoring-platform
     - **Environment:** Node
     - **Build Command:** `./render-build.sh`
     - **Start Command:** `./render-start.sh`
   - Add the environment variables:
     - `NODE_ENV`: `production`
     - `DATABASE_URL`: (Your database connection string from step 1)

### Environment Variables

- `NODE_ENV`: Set to `production` for production environment
- `DATABASE_URL`: PostgreSQL connection string - if using Neon database, get this from your Neon dashboard
- `PYTHON_API_KEYS`: Required for AI sentiment analysis functionality (if used)

## Key Features

- Advanced multilingual NLP processing
- Real-time social media and event monitoring
- Specialized Filipino language support
- Responsive mobile-first design with adaptive UI elements
- Interactive geospatial mapping capabilities

## Development

To run the application locally:

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

This will start both the backend and frontend development servers.