# PanicSense - Super Simple Render Deployment

## Deployment sa Render.com

### Step 1: Setup sa Render Dashboard

Sa Render dashboard, i-add ang new web service at i-point ito sa GitHub repository mo.

### Step 2: Configure ang deployment

**Build Command:**
```
npm install
```

**Start Command:**
```
node run.js
```

### Step 3: Environment Variables

I-add ang mga sumusunod na environment variables:

- `NODE_ENV` = `production`
- `DATABASE_URL` = [Your PostgreSQL URL]
- `GROQ_API_KEY` = [Your Groq API Key]

## Paano Gumagana Ang Deployment?

Gumawa tayo ng dalawang key files para sa deployment:

1. **render-setup.js**:
   - Nag-setup ng lahat ng dependencies (Node.js at Python)
   - Ginagawa ang file structure (python folder, client/dist folder, uploads, temp_files)
   - I-install ang Python packages na kailangan
   - May fallback na minimal packages kung may errors

2. **run.js**:
   - Tinatawag ang render-setup.js para sa setup
   - Tinatawag ang main index.js para patakbuhin ang application
   
Sa ganitong approach, napaka-simple lang ng deployment process:
- `npm install` para sa build command
- `node run.js` para sa start command

Lahat ng files ay makikita ni Render, walang hidden files o complex setup scripts!