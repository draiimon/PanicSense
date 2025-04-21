# PanicSense PH - Disaster Monitoring Platform

![PanicSense PH](./public/logo.png)

An advanced AI-powered disaster monitoring and community resilience platform specifically designed for the Philippines, leveraging cutting-edge technology to enhance emergency preparedness and response.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/panicsense-ph.git
cd panicsense-ph

# The first-time setup script will run automatically after clone
# If it doesn't, run it manually:
./firstrun.sh

# Start the application in development mode
npm run dev
```

> **First-time setup:** The repository includes Git hooks that automatically run the setup script after cloning. This guides you through configuring environment variables, installing dependencies, and setting up the database. If the automatic setup doesn't trigger, just run `./firstrun.sh` manually.

> **For Repository Administrators:** If you're setting up the repository for others to clone, run `./git-clone-setup.sh` once to configure the Git hooks for automatic setup.

## ⚙️ System Requirements

- Node.js 20.x or later
- Python 3.11 or later
- PostgreSQL 15.x
- 2GB RAM minimum (4GB recommended)

## 🔧 Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Database
DATABASE_URL="postgres://username:password@localhost:5432/panicsense"

# Server
PORT=5000
NODE_ENV=development

# Python configuration
PYTHON_PATH=python3

# Session
SESSION_SECRET=your_secret_key
```

## 📋 Features

- **AI-Powered Sentiment Analysis**: Real-time analysis of social media and news for disaster signals
- **Multi-lingual Support**: Special focus on Filipino language processing
- **Interactive Maps**: Visualize disaster events and response coordination
- **Data Analytics**: Trend analysis and historical data comparison
- **Community Alerts**: Early warning system based on AI predictions
- **Admin Dashboard**: Comprehensive monitoring and management tools

## 🧰 Technology Stack

- **Frontend**: React with TypeScript
- **Backend**: Node.js (Express)
- **Database**: PostgreSQL with Drizzle ORM
- **AI/ML**: Python with PyTorch
- **UI Components**: Tailwind CSS with shadcn/ui
- **Deployment**: Docker, Render.com

## 🌐 API Endpoints

The application exposes various RESTful endpoints:

- `/api/health` - Server health check
- `/api/auth/*` - Authentication endpoints
- `/api/sentiment-posts/*` - Sentiment analysis data
- `/api/disaster-events/*` - Disaster event information
- `/api/analyzed-files/*` - Processed data files
- More endpoints detailed in our API documentation

## 🛠️ Development Guide

### Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start in production mode
npm run start

# Run database migrations
npm run db:push

# Run tests
npm run test
```

### Project Structure

```
├── client/           # Frontend React application
├── server/           # Backend Express server
│   └── python/       # Python ML components
├── shared/           # Shared code between frontend and backend
├── migrations/       # Database migrations
├── public/           # Static assets
└── uploads/          # User uploaded files
```

## 🐳 Docker Support

You can run this application using Docker:

```bash
# Build the Docker image
docker build -t panicsense-ph .

# Run the container
docker run -p 5000:5000 -e DATABASE_URL=your_database_url panicsense-ph
```

For a complete guide on using Docker with this project, see [DOCKER_README.md](./DOCKER_README.md).

## 🚀 Deployment to Render

This project is configured for easy deployment to Render.com. Follow these steps:

1. **Push your code to a Git repository** (GitHub, GitLab, etc.)
2. **Connect to Render.com**:
   - Create a new account or log in to Render
   - Click "New" and select "Web Service"
   - Connect to your Git repository
   - Select the repository with this project

3. **Configure the Web Service**:
   - Name: `panicsense-ph` (or your preferred name)
   - Runtime: `Node`
   - Build Command: `npm install && npm run build`
   - Start Command: `node index.js`
   - Instance Type: `Standard (1x CPU, 2GB RAM)` is recommended

4. **Configure Environment Variables**:
   - Once the service is created, go to the "Environment" tab
   - Add the following environment variables:
     ```
     DATABASE_URL=postgres://username:password@host:port/database_name
     DB_SSL_REQUIRED=true
     NODE_ENV=production
     PORT=10000
     TZ=Asia/Manila
     RUNTIME_ENV=render
     HOST=0.0.0.0
     PYTHON_SERVICE_ENABLED=true
     ENABLE_SOCIAL_SCRAPER=true
     SESSION_SECRET=your_random_secure_string
     ```
     
   - **Important**: Create a PostgreSQL database instance in Render and use its connection string for `DATABASE_URL`

5. **Set Auto-Deploy**:
   - Enable automatic deployments for the `render-deployment-fix` branch

6. **Troubleshooting Common Issues**:
   - If you encounter database-related errors containing "created_at" column references:
     1. Check server logs for details
     2. Verify that render-setup.js is being executed on deployment
     3. The system is designed to automatically create necessary tables through `server/db-setup.js`

7. **Wait for deployment to complete**:
   - Render will automatically build and deploy the application
   - You can monitor the build progress in the "Logs" tab

8. **Verify the Deployment**:
   - Once deployed, visit the provided Render URL
   - Test the application functionality including data uploads, real-time updates, and news display

> **IMPORTANT**: The `render-setup.js` script automatically handles database setup and static file preparation. It's designed to fix common deployment issues including the "created_at" column errors.

## 🔒 Security

- All sensitive data should be stored in environment variables
- API endpoints are protected with appropriate authentication
- User passwords are hashed using bcrypt
- Session data is securely stored and managed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please file an issue on the GitHub repository or contact the maintainers.