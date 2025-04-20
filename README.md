# PanicSense PH - Disaster Monitoring Platform

![PanicSense PH](./public/logo.png)

An advanced AI-powered disaster monitoring and community resilience platform specifically designed for the Philippines, leveraging cutting-edge technology to enhance emergency preparedness and response.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/panicsense-ph.git
cd panicsense-ph

# Copy environment variables template and set up your variables
cp .env.example .env

# Install dependencies
npm install

# Set up the database
npm run db:push

# Start the application in development mode
npm run dev
```

> **First-time setup:** When you first run the application, a setup script will automatically check if your environment is properly configured and guide you through the setup process.

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

This project is configured for easy deployment to Render.com. For detailed instructions, see [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md).

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