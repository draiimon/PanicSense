# PanicSense - Render Deployment Package

PanicSense is an advanced disaster intelligence platform providing comprehensive emergency insights and community safety coordination through intelligent AI-driven analysis.

This is a specialized deployment package designed to run on Render.com's free tier.

## Quick Start

1. Push these files to a GitHub repository
2. Create a new Web Service in Render
3. Connect to your GitHub repository
4. Set the following:
   - Build Command: `./render-build.sh`
   - Start Command: `node start-render.cjs`
5. Add the following Environment Variables:
   - `DATABASE_URL` - Your PostgreSQL database URL
   - `NODE_ENV` - Set to `production`
   - `SESSION_SECRET` - Any random secure string
   - `DEBUG` - Set to `true` for detailed logs (optional)

## Troubleshooting

If you encounter issues, see the [RENDER-DEPLOY-FIX.md](./RENDER-DEPLOY-FIX.md) file for detailed troubleshooting steps.

## Features

- Real-time disaster alerts and monitoring
- News aggregation and sentiment analysis
- Data visualization and reporting
- File upload and processing
- WebSocket support for real-time updates
- Automatic Python service management
- Database schema auto-detection

## Technology Stack

- Node.js backend with Express
- Python data processing and NLP
- PostgreSQL database
- WebSockets for real-time updates

## Contact

For more information, visit [draiimon/PanicSense](https://github.com/draiimon/PanicSense) on GitHub.