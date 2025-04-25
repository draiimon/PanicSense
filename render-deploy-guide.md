# Deploying PanicSense to Render

This guide provides instructions for deploying the PanicSense application to Render.com without using a render.yml file.

## Prerequisites

- A Render.com account
- Your PanicSense codebase pushed to a Git repository (GitHub, GitLab, etc.)
- Database connection string for Neon or PostgreSQL database

## Deployment Steps

### 1. Build Configuration

For deploying on Render, use the following configuration:

**Environment**: Node.js

**Build Command**:
```
npm install && npm run build
```

**Start Command**:
```
node render-start.js
```

### 2. Environment Variables

Set the following environment variables in the Render dashboard:

```
NODE_ENV=production
PORT=10000
DATABASE_URL=your_neon_postgres_connection_string
SESSION_SECRET=your_secure_session_secret
```

Add any additional environment variables needed by your application (API keys, etc.).

### 3. Deploy Process

1. Log in to your Render dashboard
2. Click "New" and select "Web Service"
3. Connect your Git repository
4. Configure the settings:
   - Name: "panicsense" (or your preferred name)
   - Environment: Node.js
   - Build Command: `npm install && npm run build`
   - Start Command: `node render-start.js`
   - Add the environment variables mentioned above
5. Click "Create Web Service"

Render will automatically build and deploy your application.

### 4. Database Connection

Ensure your DATABASE_URL environment variable contains a valid connection string to your Neon or PostgreSQL database. The application is already configured to use this connection string.

### 5. Custom Domains (Optional)

To set up a custom domain for your application:

1. Go to your service dashboard in Render
2. Navigate to "Settings" > "Custom Domain"
3. Follow the instructions to add and verify your domain

## Troubleshooting

### Building Issues

If you encounter issues during the build process, check:

- Node.js version compatibility (add an `.nvmrc` or `engines` field in package.json)
- Build script dependencies (ensure all required packages are in dependencies, not just devDependencies)

### Database Connection Issues

If the application cannot connect to the database:

- Verify the DATABASE_URL environment variable is correctly set
- Ensure the database server allows connections from Render IPs
- Check if the database service is running

### Static File Serving

The application is configured to automatically detect and serve static files from multiple possible directories. If your frontend is not showing:

- Make sure the build process correctly generates files in the expected locations
- Check the Render logs for any file path issues
- Verify that `npm run build` successfully builds both frontend and backend

## Additional Information

- Your server will be accessible at `https://your-service-name.onrender.com`
- Render automatically handles HTTPS certificates
- The application listens on the port specified by the PORT environment variable, but Render will route external traffic to your service regardless of the port value