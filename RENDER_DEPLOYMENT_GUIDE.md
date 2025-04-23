# Step-by-Step Guide: Deploying PanicSense on Render

This guide provides a comprehensive walkthrough for deploying PanicSense on Render's free tier.

## Prerequisites

1. GitHub account with the PanicSense repository
2. Render.com account
3. Neon.tech PostgreSQL database
4. Groq API key for AI functions

## Step 1: Set Up Your Render Account

1. Navigate to [render.com](https://render.com) and sign up/log in
2. Click "New" and select "Web Service"

## Step 2: Connect Your GitHub Repository

1. Click "Connect account" and choose GitHub
2. Find your PanicSense repository and connect it

## Step 3: Configure the Web Service

Enter the following settings:

- **Name**: PanicSense (or your preferred name)
- **Environment**: Node
- **Region**: Choose the closest to your location
- **Branch**: main

### CRITICAL SETTINGS:

- **Build Command**: `node render-setup.js`
- **Start Command**: `NODE_ENV=production node start-prod.js`
- **Plan**: Free (middle of page)

## Step 4: Set Environment Variables

Click "Advanced" and add the following:

- `NODE_ENV` = production
- `DATABASE_URL` = (your Neon PostgreSQL connection string)
- `GROQ_API_KEY` = (your Groq API key)
- `SESSION_SECRET` = (any random string)

## Step 5: Create Web Service

Click "Create Web Service"

## Step 6: Monitor Deployment

- Check the logs for any errors
- Once deployment is successful, your app will be available at <service-name>.onrender.com

## Step 7: Database Setup

If you haven't already:

1. Go to [neon.tech](https://neon.tech)
2. Create a new PostgreSQL database
3. Get the connection string and add it to your environment variables

## Step 8: Custom Domain (Optional)

If you want to use a custom domain:

1. Click "Settings" in the Render dashboard
2. Scroll down to the "Custom Domains" section
3. Click "Add Custom Domain"
4. Follow the instructions provided

## Troubleshooting

If you encounter errors:

1. Check the logs in the Render dashboard
2. Verify all environment variables are correctly set
3. If you see "vite not found" errors, the fallback mechanism will still work

## Key Files We Created

These are the most important files for your Render deployment:

1. `render-setup.js` - Ultra-simplified setup script
2. `start-prod.js` - Production startup script
3. `render.yaml` - Render configuration file

## Support

If you encounter issues, check the logs in the Render dashboard to see what's going wrong.

---

Congratulations! You now have your own PanicSense deployment!