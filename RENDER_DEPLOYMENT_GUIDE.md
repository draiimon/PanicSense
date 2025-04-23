# PanicSense - FREE TIER Render Deployment Guide

This guide is specifically for deploying PanicSense on Render's **FREE TIER** using the Web Service deployment method.

## Pre-Deployment Checklist

Before deploying to Render, make sure:
1. Your code is pushed to a GitHub repository
2. You have a Neon PostgreSQL database setup with connection details ready
3. You have your Groq API key available

## Quick Deployment Steps for FREE TIER

### 1. Sign Up for Render

If you don't have a Render account, sign up at [render.com](https://render.com)

### 2. Create a New Web Service (NOT Blueprint)

1. From the Render dashboard, click **New** and select **Web Service**
2. Connect your GitHub repository
3. Choose the repository containing your PanicSense application

### 3. Configure FREE TIER Settings

Enter the following information:

- **Name**: PanicSense (or your preferred name)
- **Environment**: Node
- **Region**: Choose the region closest to your users
- **Branch**: main (or your preferred branch)
- **Build Command**: `chmod +x ./build.sh && ./build.sh`
- **Start Command**: `NODE_ENV=production node server/index-wrapper.js`
- **Instance Type**: Free

### 4. Add Environment Variables

Click **Advanced** and add these environment variables:

| Variable | Value |
|---------|-------------|
| `NODE_ENV` | `production` |
| `DATABASE_URL` | Your Neon PostgreSQL connection string |
| `GROQ_API_KEY` | Your Groq API key |
| `SESSION_SECRET` | Any random string |

### 5. Create Free Tier Web Service

Click **Create Web Service** to start the deployment. The free tier has these limitations:
- 512 MB RAM
- Spins down after inactivity
- Slower cold starts
- Limited bandwidth

### 6. Monitor Deployment

Monitor the logs to ensure your application deploys successfully.

## FREE TIER Troubleshooting

If you encounter issues on the free tier:

1. **"Vite not found" Error**: If you see this error in your build logs, use one of these fixes:
   - Use our updated `build.sh` script which explicitly installs Vite globally
   - If that fails, try the Render web dashboard and change the start command to: `NODE_ENV=production node start.js`
   - For manual build, update the build command to: `npm install -g vite esbuild && chmod +x ./build.sh && ./build.sh`

2. **Resource Limits**: The free tier has limited resources (512MB RAM). If your app is crashing, it might be hitting memory limits.

3. **Cold Starts**: Your app will spin down after inactivity. The first request after inactivity will take longer.

4. **Database Connection**: Make sure your PostgreSQL database allows connections from Render's IPs.

5. **Build Timeout**: Free tier has a 15-minute build limit. If your build is timing out, optimize it.

## Important Notes for FREE TIER

1. Your app on free tier will sleep after 15 minutes of inactivity.

2. The free instance has limited processing power - complex operations may be slower.

3. Free tier web services have a shared CPU, so performance can vary.

4. There is a soft bandwidth limit of 100 GB/month on the free tier.

---

**Note**: The `build.sh` script and `Procfile` we've created are specifically designed to work well with Render's free tier.