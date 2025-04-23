# PanicSense - Render Deployment Guide

## 🚀 Step-by-Step Deployment sa Render.com

### Step 1: Setup sa Render Dashboard

Sa Render dashboard:
1. Click "New"
2. Piliin ang "Web Service"
3. I-connect ang GitHub repository mo o mag-upload ng code
4. I-set ang name at region

### Step 2: Configure ang Deployment (PINAKAMAHALAGA)

**GAMITIN MO ITO (100% RELIABLE METHOD):**

**Build Command:**
```
npm install
```

**Start Command:**
```
node run.cjs
```

**BAKIT ITO ANG BEST CHOICE:**
- ✅ Automatic copying ng frontend files mula sa client/dist folder
- ✅ Support para sa multiple file locations
- ✅ Auto-creation ng required directories
- ✅ Fallback placeholder page kung hindi available ang frontend
- ✅ Detailed logs para sa troubleshooting
- ✅ Supports Python dependencies kung kailangan

### Step 3: Environment Variables

I-add ang mga sumusunod na environment variables:

| Variable | Value | Requirement |
|----------|-------|-------------|
| `NODE_ENV` | `production` | **REQUIRED** |
| `DATABASE_URL` | [Your PostgreSQL URL] | **REQUIRED** |
| `GROQ_API_KEY` | [Your Groq API Key] | **REQUIRED** |
| `PORT` | `10000` (o iba) | Optional |
| `SESSION_SECRET` | Random string | Optional |

## ⚡ Paano Gumagana Ang Bagong Deployment System

### 1. render-setup.sh Script
Nag-aautomate ng pre-deployment setup:
- Creates all required directories
- Copies frontend files from client/dist to dist/public
- Sets up placeholder files kung wala ang frontend
- Installs Python dependencies if needed

### 2. Enhanced run.cjs
Nagha-handle ng complete deployment process:
- Executes render-setup.sh script
- Performs fallback manual setup kung hindi available ang script
- Checks multiple possible locations for frontend files
- Provides detailed logging for troubleshooting
- Creates clean placeholder page kung API-only mode

### 3. Improved Static File Serving
Sa server/index-wrapper.js:
- Checks multiple possible locations for frontend files (dist/public, client/dist, public)
- Intelligent fallback handling
- Shows helpful error messages kung walang frontend files

## 🔍 Troubleshooting

### Problem 1: "Application not properly built. Static files missing"
**Solution:**
- Check if index.html exists in client/dist
- Kung wala, i-rebuild ang frontend locally:
  ```
  cd client && npm run build
  ```
- Upload the built files to your repository

### Problem 2: Backend API is working but frontend is not
**Solution:**
- Our enhanced system should handle this automatically
- Kung may issues pa rin, check ang logs sa Render
- Try manual copying ng files to dist/public

### Problem 3: Database connection errors
**Solution:**
- Verify na correct ang DATABASE_URL
- Make sure na may access ang Render sa database IP mo
- Check firewall settings ng database

## 🚀 Final Checklist Bago I-deploy

1. ✅ Naka-commit at pushed ba ang latest code sa GitHub?
2. ✅ Mayroon bang pre-built frontend files sa client/dist?
3. ✅ Completo ba ang environment variables?
4. ✅ Naka-setup ba ang database sa Neon o iba?

---

## 📝 Technical Details (Advanced)

### Compatibility Features ng Deployment System

1. **CommonJS Compatibility**
   - run.cjs ay CommonJS format para compatible sa lahat ng Node.js environments
   - Hindi affected ng "type": "module" sa package.json

2. **No Build Tool Dependencies**
   - Hindi umaasa sa Vite, esbuild, o iba pang build tools
   - Plain Node.js lang ang requirement

3. **Multi-location File Serving**
   - Checks sa multiple locations para sa frontend files
   - Graceful fallback sa API-only mode kung needed

4. **Enhanced Logging**
   - Detailed logs for debugging
   - Clear error messages sa runtime

Lahat ng enhancements ay designed for maximum reliability at flexibility sa Render environment!