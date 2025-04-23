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
node run.cjs
```

### Step 3: Environment Variables

I-add ang mga sumusunod na environment variables:

- `NODE_ENV` = `production`
- `DATABASE_URL` = [Your PostgreSQL URL]
- `GROQ_API_KEY` = [Your Groq API Key]

## Paano Gumagana Ang Deployment?

Gumawa tayo ng isang key file para sa deployment:

**run.cjs**:
   - Super simple na file na nagko-connect sa main application
   - Direkta na tinatawag ang main index.js para patakbuhin ang application
   - CommonJS format para compatible sa kahit anong environment
   - Ginagamit ang .cjs extension para masabi kay Node.js na iba ito sa ES modules
   - Compatible sa Render kahit may "type": "module" sa package.json
   
Sa ganitong approach, napaka-simple lang ng deployment process:
- `npm install` para sa build command
- `node run.cjs` para sa start command

Lahat ng files ay makikita ni Render, walang hidden files o complex setup scripts!